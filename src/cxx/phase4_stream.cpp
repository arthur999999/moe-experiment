// src/cxx/phase4_stream.cpp - Phase 4 (final, no O_DIRECT)
// =============================================================================
// PHASE 4 - Expert streaming with LRU CACHE + Strategy B
//
// - Expert reading via normal pread() (buffered; the OS page cache
//   keeps a copy, which is acceptable in this phase).
// - O_DIRECT was tested and REMOVED: GGUF aligns tensors to 32 bytes, offsets
//   are not multiples of 512/4096 -> EINVAL. Revisit in Phase 8 if needed.
// - LRU Cache: only hot experts stay in RAM; miss = pread from disk.
// - Strategy B: compact buffer (n_expert_used slots) + rewriting router
//   ids (ffn_moe_topk) to slots 0..7.
// =============================================================================

#include "llama.h"
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cerrno>
#include <string>
#include <vector>
#include <unordered_map>
#include <fcntl.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// cache LRU
// ---------------------------------------------------------------------------

// kind: 0 = gate, 1 = up, 2 = down
struct CacheKey {
    int layer;
    int kind;
    int eid;
    bool operator==(const CacheKey & o) const {
        return layer == o.layer && kind == o.kind && eid == o.eid;
    }
};
struct CacheKeyHash {
    size_t operator()(const CacheKey & k) const {
        size_t h = 0;
        h = h * 31 + (size_t)k.layer;
        h = h * 31 + (size_t)k.kind;
        h = h * 31 + (size_t)k.eid;
        return h;
    }
};

struct CacheEntry {
    std::vector<uint8_t> data;
    uint64_t last_used;
};

struct ExpertCache {
    size_t max_bytes;
    size_t used_bytes = 0;
    uint64_t clock = 0;
    std::unordered_map<CacheKey, CacheEntry, CacheKeyHash> map;
    uint64_t hits = 0, misses = 0;

    explicit ExpertCache(size_t max_bytes_) : max_bytes(max_bytes_) {}

    // procura; se hit, copia 'size' bytes para dst; retorna true
    bool lookup(const CacheKey & key, size_t size, uint8_t * dst) {
        auto it = map.find(key);
        if (it == map.end()) { ++misses; return false; }
        ++hits;
        it->second.last_used = ++clock;
        memcpy(dst, it->second.data.data(), size);
        return true;
    }

    void insert(const CacheKey & key, const uint8_t * data, size_t size) {
        if (map.count(key)) return;
        while (used_bytes + size > max_bytes && !map.empty()) {
            // evict least recently used (LRU)
            CacheKey victim;
            uint64_t min_ = UINT64_MAX;
            for (auto & kv : map) {
                if (kv.second.last_used < min_) { min_ = kv.second.last_used; victim = kv.first; }
            }
            used_bytes -= map[victim].data.size();
            map.erase(victim);
        }
        CacheEntry e;
        e.data.assign(data, data + size);
        e.last_used = ++clock;
        map[key] = std::move(e);
        used_bytes += size;
    }
};

// ---------------------------------------------------------------------------
// estruturas do streamer
// ---------------------------------------------------------------------------

struct ExpertTensor {
    struct ggml_tensor * t = nullptr;
    void * original = nullptr;
    const char * name = nullptr;
    size_t data_offset = 0;
    size_t n_bytes = 0;             // tensor fundido (64 experts)
    size_t bytes_per_expert = 0;
};

struct LayerCtx {
    ExpertTensor gate, up, down;
};

struct Streamer {
    std::vector<LayerCtx> layers;
    struct gguf_context * gg = nullptr;
    int fd = -1;
    size_t n_expert = 64;
    size_t n_expert_used = 8;
    bool warmed_up = false;
    bool enable_stream = false;

    // buffers compactos (Estratégia B): n_expert_used slots
    uint8_t * gate_buf = nullptr; size_t gate_cap = 0;
    uint8_t * up_buf   = nullptr; size_t up_cap   = 0;
    uint8_t * down_buf = nullptr; size_t down_cap = 0;

    ExpertCache * cache = nullptr;
    uint64_t n_expert_reads = 0;   // preads reais (disk)
};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static bool has_sub(const char * name, const char * sub) {
    return name && strstr(name, sub) != nullptr;
}
static void ensure_buf(uint8_t ** buf, size_t * cap, size_t need) {
    if (*cap < need) { delete[] *buf; *buf = new uint8_t[need]; *cap = need; }
}
static void pread_expert(const Streamer & s, const ExpertTensor & et,
                         int32_t eid, uint8_t * dst) {
    const size_t abs = et.data_offset + (size_t)eid * et.bytes_per_expert;
    const ssize_t n = pread(s.fd, dst, et.bytes_per_expert, (off_t)abs);
    if (n != (ssize_t)et.bytes_per_expert)
        fprintf(stderr, "ERROR pread expert=%d off=%zu got=%zd (%s)\n", eid, abs, n, strerror(errno));
}

static int moe_tensor_kind(const char * name) {
    if (has_sub(name, "ffn_gate_exps")) return 0;
    if (has_sub(name, "ffn_up_exps"))   return 1;
    if (has_sub(name, "ffn_down_exps")) return 2;
    return -1;
}

// ---------------------------------------------------------------------------
// callback
// ---------------------------------------------------------------------------

static bool moe_callback(struct ggml_tensor * t, bool ask, void * ud) {
    auto * s = static_cast<Streamer *>(ud);
    if (ask) return !s->warmed_up || has_sub(t->name, "ffn_moe_topk");

    // warm-up: descobre os tensores de expert nas fontes dos nós
    if (!s->warmed_up) {
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            struct ggml_tensor * src = t->src[i];
            if (!src || !src->name) break;
            int kind = moe_tensor_kind(src->name);
            if (kind < 0) continue;

            int layer = -1;
            if (sscanf(src->name, "blk.%d.", &layer) != 1) continue;
            if (layer < 0 || layer >= (int)s->layers.size()) continue;

            ExpertTensor * target =
                kind == 0 ? &s->layers[layer].gate :
                kind == 1 ? &s->layers[layer].up   :
                            &s->layers[layer].down;
            if (target->t) continue;

            target->t = src;
            target->original = src->data;
            target->name = src->name;
            target->n_bytes = ggml_nbytes(src);
            target->bytes_per_expert = target->n_bytes / s->n_expert;

            int tid = gguf_find_tensor(s->gg, src->name);
            if (tid >= 0)
                target->data_offset = gguf_get_data_offset(s->gg) + gguf_get_tensor_offset(s->gg, tid);
        }
        return true;
    }

    // geração: streaming com cache + Estratégia B
    if (s->enable_stream && has_sub(t->name, "ffn_moe_topk")) {
        int layer = -1;
        if (sscanf(t->name, "ffn_moe_topk-%d", &layer) != 1) return true;
        if (layer < 0 || layer >= (int)s->layers.size()) return true;
        if (t->ne[1] != 1) return true;   // só single-token decode

        LayerCtx & L = s->layers[layer];
        const int32_t * orig_ids = (const int32_t *)t->data;

        // garante buffers compactos: n_expert_used slots
        ensure_buf(&s->gate_buf, &s->gate_cap, s->n_expert_used * L.gate.bytes_per_expert);
        ensure_buf(&s->up_buf,   &s->up_cap,   s->n_expert_used * L.up.bytes_per_expert);
        ensure_buf(&s->down_buf, &s->down_cap, s->n_expert_used * L.down.bytes_per_expert);

        // preenche os slots compactos (cache lookup ou pread do disco)
        for (size_t k = 0; k < s->n_expert_used; ++k) {
            int32_t eid = orig_ids[k];
            if (eid < 0 || eid >= (int32_t)s->n_expert) continue;

            CacheKey kg{layer, 0, eid};
            uint8_t * dst_g = s->gate_buf + k * L.gate.bytes_per_expert;
            if (s->cache && s->cache->lookup(kg, L.gate.bytes_per_expert, dst_g)) {
                /* hit */
            } else {
                pread_expert(*s, L.gate, eid, dst_g);
                if (s->cache) s->cache->insert(kg, dst_g, L.gate.bytes_per_expert);
                ++s->n_expert_reads;
            }

            CacheKey ku{layer, 1, eid};
            uint8_t * dst_u = s->up_buf + k * L.up.bytes_per_expert;
            if (s->cache && s->cache->lookup(ku, L.up.bytes_per_expert, dst_u)) {
                /* hit */
            } else {
                pread_expert(*s, L.up, eid, dst_u);
                if (s->cache) s->cache->insert(ku, dst_u, L.up.bytes_per_expert);
                ++s->n_expert_reads;
            }

            CacheKey kd{layer, 2, eid};
            uint8_t * dst_d = s->down_buf + k * L.down.bytes_per_expert;
            if (s->cache && s->cache->lookup(kd, L.down.bytes_per_expert, dst_d)) {
                /* hit */
            } else {
                pread_expert(*s, L.down, eid, dst_d);
                if (s->cache) s->cache->insert(kd, dst_d, L.down.bytes_per_expert);
                ++s->n_expert_reads;
            }
        }

        // ESTRATÉGIA B: reescreve ids do router para slots compactos 0..7
        // O kernel mul_mat_id lê o expert no offset id*bytes_per_expert do src1.
        // Com ids[k]=k, ele lê o slot k do nosso buffer compacto.
        int32_t * ids = (int32_t *)t->data;   // escrevemos no buffer do router
        for (size_t k = 0; k < s->n_expert_used; ++k) ids[k] = (int32_t)k;

        // rebind ->data to compact buffers
        L.gate.t->data = s->gate_buf;
        L.up.t->data   = s->up_buf;
        L.down.t->data = s->down_buf;
    }
    return true;
}

static void restore_all(Streamer & s) {
    for (auto & L : s.layers) {
        if (L.gate.t) L.gate.t->data = L.gate.original;
        if (L.up.t)   L.up.t->data   = L.up.original;
        if (L.down.t) L.down.t->data = L.down.original;
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    std::string model_path = argc > 1 ? argv[1] : "./models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf";
    std::string prompt     = argc > 2 ? argv[2] : "Explain how mixture of experts routing works.";
    int n_pred    = argc > 3 ? atoi(argv[3]) : 128;
    int n_threads = argc > 4 ? atoi(argv[4]) : 4;
    uint32_t seed = argc > 5 ? (uint32_t)strtoul(argv[5], nullptr, 10) : 42;
    size_t cache_mb = argc > 6 ? (size_t)atoll(argv[6]) : 64;   // 0 = sem cache

    llama_backend_init();

    struct gguf_init_params ggpar = { true, nullptr };
    struct gguf_context * gg = gguf_init_from_file(model_path.c_str(), ggpar);
    if (!gg) { fprintf(stderr, "error gguf\n"); return 1; }

    auto gg_u32 = [&](const char * k, size_t d) { int id = gguf_find_key(gg, k); return id < 0 ? d : (size_t)gguf_get_val_u32(gg, id); };
    size_t n_layer = gg_u32("olmoe.block_count", 16);
    size_t n_expert = gg_u32("olmoe.expert_count", 64);
    size_t n_expert_used = gg_u32("olmoe.expert_used_count", 8);
    fprintf(stderr, "%zu layers, %zu experts, top-%zu, cache=%zu MiB\n",
            n_layer, n_expert, n_expert_used, cache_mb);

    llama_model_params mp = llama_model_default_params();
    mp.use_extra_bufts = false; mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "error loading model\n"); return 1; }

    int fd = open(model_path.c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "error open\n"); return 1; }

    Streamer st;
    st.gg = gg; st.fd = fd; st.n_expert = n_expert; st.n_expert_used = n_expert_used;
    st.layers.resize(n_layer);

    ExpertCache cache(cache_mb * (1u << 20));
    st.cache = (cache_mb > 0 ? &cache : nullptr);

    llama_context_params cp = llama_context_default_params();
    cp.n_threads = n_threads; cp.n_threads_batch = n_threads;
    cp.cb_eval = moe_callback; cp.cb_eval_user_data = &st;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "error context\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // ---- chat template (igual ao llama-cli) ----
    char tmpl[4096];
    int tmpl_len = llama_model_meta_val_str(model, "tokenizer.chat_template",
                                            tmpl, sizeof(tmpl));
    if (tmpl_len < 0) {
        fprintf(stderr, "[template] no template; using raw prompt\n");
    } else {
        llama_chat_message msgs[] = { { "user", prompt.c_str() } };
        std::vector<char> fmt(prompt.size() * 2 + 512);
        int flen = llama_chat_apply_template(tmpl, msgs, 1, true,
                                             fmt.data(), (int)fmt.size());
        if (flen < 0) {
            fprintf(stderr, "[template] error applying template; using raw prompt\n");
        } else {
            fmt.resize(flen);
            prompt.assign(fmt.data(), fmt.size());
            fprintf(stderr, "[template] formatted prompt (%d bytes)\n", flen);
        }
    }

    std::vector<llama_token> toks(prompt.size() + 64);
    int nt = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                            toks.data(), (int)toks.size(), true, false);
    if (nt < 0) nt = 0; toks.resize(nt);

    { llama_batch b = llama_batch_get_one(toks.data(), (int32_t)toks.size());
      if (llama_decode(ctx, b) != 0) { fprintf(stderr, "error prefill\n"); return 1; } }
    st.warmed_up = true; st.enable_stream = true;

    size_t found = 0;
    for (auto & L : st.layers) { if (L.gate.t) found++; if (L.down.t) found++; }
    fprintf(stderr, "warm-up: %zu/%zu experts discovered\n", found, n_layer * 2);

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // geração (cronometrada)
    int64_t t0 = llama_time_us();
    std::vector<llama_token> gen = toks;
    for (int i = 0; i < n_pred; ++i) {
        if ((int)gen.size() > (int)toks.size()) {
            llama_token tok = gen.back();
            llama_batch b = llama_batch_get_one(&tok, 1);
            if (llama_decode(ctx, b) != 0) { fprintf(stderr, "error decode\n"); break; }
            restore_all(st);
        }
        const llama_token new_tok = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_accept(smpl, new_tok);
        char piece[256];
        int np = llama_token_to_piece(vocab, new_tok, piece, (int)sizeof(piece), 0, false);
        if (np > 0) fwrite(piece, 1, (size_t)np, stdout);
        fflush(stdout);
        gen.push_back(new_tok);
        if (llama_vocab_is_eog(vocab, new_tok)) break;
    }
    int64_t t1 = llama_time_us();
    printf("\n");

    size_t n_generated = gen.size() - toks.size();
    double dt_s = (double)(t1 - t0) / 1e6;
    fprintf(stderr, "Phase 4: %zu tokens, %.2f s, %.2f tok/s\n",
            n_generated, dt_s, n_generated / dt_s);

    if (st.cache) {
        double hr = 100.0 * (double)st.cache->hits /
                    (double)(st.cache->hits + st.cache->misses + 1e-9);
        fprintf(stderr, "Cache: hits=%llu misses=%llu hit_rate=%.1f%% disk_reads=%llu cache_bytes=%zu\n",
                (unsigned long long)st.cache->hits, (unsigned long long)st.cache->misses,
                hr, (unsigned long long)st.n_expert_reads, st.cache->used_bytes);
    } else {
        fprintf(stderr, "Cache: disabled. disk_reads=%llu\n",
                (unsigned long long)st.n_expert_reads);
    }

    llama_sampler_free(smpl); llama_free(ctx); llama_model_free(model);
    llama_backend_free(); gguf_free(gg); if (fd >= 0) close(fd);
    delete[] st.gate_buf; delete[] st.up_buf; delete[] st.down_buf;
    return 0;
}