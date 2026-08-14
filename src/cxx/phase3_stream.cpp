// src/cxx/phase3_stream.cpp - Phase 3 (fixed: absolute GGUF offset)
#include "llama.h"
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cerrno>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

struct ExpertTensor {
    struct ggml_tensor * t = nullptr;
    void * original = nullptr;
    const char * name = nullptr;
    size_t data_offset = 0;
    size_t n_bytes = 0;
    size_t bytes_per_expert = 0;
};
struct LayerCtx { ExpertTensor gate, up, down; };
struct Streamer {
    std::vector<LayerCtx> layers;
    struct gguf_context * gg = nullptr;
    int fd = -1;
    size_t n_expert = 64;
    size_t n_expert_used = 8;
    bool warmed_up = false;
    bool enable_stream = false;
    uint8_t * gate_buf = nullptr; size_t gate_cap = 0;
    uint8_t * up_buf   = nullptr; size_t up_cap   = 0;
    uint8_t * down_buf = nullptr; size_t down_cap = 0;
    uint64_t n_expert_reads = 0;
    bool verif_done = false;
};

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
        fprintf(stderr, "ERROR pread expert=%d off=%zu got=%zd\n", eid, abs, n);
}

static bool moe_callback(struct ggml_tensor * t, bool ask, void * ud) {
    auto * s = static_cast<Streamer *>(ud);
    if (ask) return !s->warmed_up || has_sub(t->name, "ffn_moe_topk");

    if (!s->warmed_up) {
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            struct ggml_tensor * src = t->src[i];
            if (!src || !src->name) break;
            if (!has_sub(src->name, "ffn_gate_exps")
             && !has_sub(src->name, "ffn_up_exps")
             && !has_sub(src->name, "ffn_down_exps")) continue;
            int layer = -1;
            if (sscanf(src->name, "blk.%d.", &layer) != 1) continue;
            if (layer < 0 || layer >= (int)s->layers.size()) continue;
            ExpertTensor * target = nullptr;
            if      (has_sub(src->name, "ffn_gate_exps")) target = &s->layers[layer].gate;
            else if (has_sub(src->name, "ffn_up_exps"))   target = &s->layers[layer].up;
            else if (has_sub(src->name, "ffn_down_exps")) target = &s->layers[layer].down;
            if (!target || target->t) continue;
            target->t = src; target->original = src->data; target->name = src->name;
            target->n_bytes = ggml_nbytes(src);
            target->bytes_per_expert = target->n_bytes / s->n_expert;
            int tid = gguf_find_tensor(s->gg, src->name);
            if (tid >= 0)
                // FIX: absolute offset = data base + relative tensor offset
                target->data_offset = gguf_get_data_offset(s->gg) + gguf_get_tensor_offset(s->gg, tid);
        }
        return true;
    }

    if (s->enable_stream && has_sub(t->name, "ffn_moe_topk")) {
        int layer = -1;
        if (sscanf(t->name, "ffn_moe_topk-%d", &layer) != 1) return true;
        if (layer < 0 || layer >= (int)s->layers.size()) return true;
        if (t->ne[1] != 1) return true;

        LayerCtx & L = s->layers[layer];
        const int32_t * ids = (const int32_t *)t->data;

        if (layer == 0 && !s->verif_done) {
            fprintf(stderr, "[DIAG] layer0 ids = [%d %d %d %d %d %d %d %d]\n",
                    ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7]);
        }

        ensure_buf(&s->gate_buf, &s->gate_cap, L.gate.n_bytes);
        ensure_buf(&s->up_buf,   &s->up_cap,   L.up.n_bytes);
        ensure_buf(&s->down_buf, &s->down_cap, L.down.n_bytes);

        for (size_t k = 0; k < s->n_expert_used; ++k) {
            int32_t eid = ids[k];
            if (eid < 0 || eid >= (int32_t)s->n_expert) continue;
            pread_expert(*s, L.gate, eid, s->gate_buf + (size_t)eid * L.gate.bytes_per_expert);
            pread_expert(*s, L.up,   eid, s->up_buf   + (size_t)eid * L.up.bytes_per_expert);
            pread_expert(*s, L.down, eid, s->down_buf + (size_t)eid * L.down.bytes_per_expert);
            s->n_expert_reads += 3;
        }

        if (layer == 0 && !s->verif_done) {
            int eid = ids[0];
            const uint8_t * mmap = (const uint8_t *)L.gate.original;
            const uint8_t * prd  = s->gate_buf + (size_t)eid * L.gate.bytes_per_expert;
            int dc = 0;
            for (size_t b = 0; b < L.gate.bytes_per_expert && b < 64; ++b)
                if (mmap[(size_t)eid * L.gate.bytes_per_expert + b] != prd[b]) ++dc;
            if (dc == 0) fprintf(stderr, "[DIAG] OK: expert %d == mmap (64 bytes)\n", eid);
            else         fprintf(stderr, "[DIAG] FAIL: %d bytes differ (expert %d)\n", dc, eid);
            s->verif_done = true;
        }

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

int main(int argc, char ** argv) {
    std::string model_path = argc > 1 ? argv[1] : "./models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf";
    std::string prompt     = argc > 2 ? argv[2] : "Explain how mixture of experts routing works.";
    int n_pred    = argc > 3 ? atoi(argv[3]) : 128;
    int n_threads = argc > 4 ? atoi(argv[4]) : 4;
    uint32_t seed = argc > 5 ? (uint32_t)strtoul(argv[5], nullptr, 10) : 42;

    llama_backend_init();

    struct gguf_init_params ggpar = { true, nullptr };
    struct gguf_context * gg = gguf_init_from_file(model_path.c_str(), ggpar);
    if (!gg) { fprintf(stderr, "error gguf\n"); return 1; }

    auto gg_u32 = [&](const char * k, size_t d) { int id = gguf_find_key(gg, k); return id < 0 ? d : (size_t)gguf_get_val_u32(gg, id); };
    size_t n_layer = gg_u32("olmoe.block_count", 16);
    size_t n_expert = gg_u32("olmoe.expert_count", 64);
    size_t n_expert_used = gg_u32("olmoe.expert_used_count", 8);
    fprintf(stderr, "%zu layers, %zu experts, top-%zu\n", n_layer, n_expert, n_expert_used);

    llama_model_params mp = llama_model_default_params();
    mp.use_extra_bufts = false; mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "error loading model\n"); return 1; }

    int fd = open(model_path.c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "error open\n"); return 1; }

    Streamer st;
    st.gg = gg; st.fd = fd; st.n_expert = n_expert; st.n_expert_used = n_expert_used;
    st.layers.resize(n_layer);

    llama_context_params cp = llama_context_default_params();
    cp.n_threads = n_threads; cp.n_threads_batch = n_threads;
    cp.cb_eval = moe_callback; cp.cb_eval_user_data = &st;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "error context\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    char tmpl[4096];
    int tmpl_len = llama_model_meta_val_str(model, "tokenizer.chat_template",
                                            tmpl, sizeof(tmpl));
    if (tmpl_len < 0) {
        fprintf(stderr, "[template] no template; using raw prompt\n");
    } else {
        // 2) format as user -> assistant conversation
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
    int nt = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(), toks.data(), (int)toks.size(), true, false);
    if (nt < 0) nt = 0; toks.resize(nt);

    { llama_batch b = llama_batch_get_one(toks.data(), (int32_t)toks.size());
      if (llama_decode(ctx, b) != 0) { fprintf(stderr, "error prefill\n"); return 1; } }
    st.warmed_up = true; st.enable_stream = true;

    // DIAG: confirm absolute offset
    fprintf(stderr, "[DIAG] layer0.gate data_offset=%zu (expected 169841472)\n", st.layers[0].gate.data_offset);

    size_t found = 0;
    for (auto & L : st.layers) { if (L.gate.t) found++; if (L.down.t) found++; }
    fprintf(stderr, "warm-up: %zu/%zu experts discovered\n", found, n_layer * 2);

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

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
    printf("\n");
    fprintf(stderr, "Phase 3: %llu reads, %zu tokens\n", (unsigned long long)st.n_expert_reads, gen.size() - toks.size());

    llama_sampler_free(smpl); llama_free(ctx); llama_model_free(model);
    llama_backend_free(); gguf_free(gg); if (fd >= 0) close(fd);
    delete[] st.gate_buf; delete[] st.up_buf; delete[] st.down_buf;
    return 0;
}