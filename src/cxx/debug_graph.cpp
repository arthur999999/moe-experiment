// src/cxx/debug_graph.cpp
// Debug: prints names/types of router nodes and expert tensors
// from the llama.cpp graph at version (commit 0b1bad1).
#include "llama.h"
#include "ggml.h"
#include <cstdio>
#include <cstring>
#include <vector>

static bool debug_cb(struct ggml_tensor * t, bool ask, void *) {
    if (ask) return true; // observe everything
    if (t->name && strstr(t->name, "moe_topk")) {
        printf("[ROUTER] name='%s' type=%d ne=[%lld,%lld,%lld,%lld]\n",
               t->name, t->type,
               (long long)t->ne[0], (long long)t->ne[1],
               (long long)t->ne[2], (long long)t->ne[3]);
        printf("  src0='%s' src1='%s'\n",
               t->src[0] && t->src[0]->name ? t->src[0]->name : "(null)",
               t->src[1] && t->src[1]->name ? t->src[1]->name : "(null)");
    }
    if (t->name && strstr(t->name, "_exps.")) {
        printf("[EXPERT] name='%s' type=%d n_bytes=%zu ne=[%lld,%lld,%lld,%lld]\n",
               t->name, t->type, ggml_nbytes(t),
               (long long)t->ne[0], (long long)t->ne[1],
               (long long)t->ne[2], (long long)t->ne[3]);
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]); return 1; }
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.use_extra_bufts = false;   // ~ --no-repack

    llama_model * m = llama_model_load_from_file(argv[1], mp);
    if (!m) { fprintf(stderr, "ERROR: model\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_threads = 1; cp.n_threads_batch = 1;
    cp.cb_eval = debug_cb;

    llama_context * ctx = llama_init_from_model(m, cp);
    if (!ctx) { fprintf(stderr, "ERROR: context\n"); return 1; }

    const char * prompt = "Hello.";
    const llama_vocab * vocab = llama_model_get_vocab(m);
    std::vector<llama_token> toks(100);
    int nt = llama_tokenize(vocab, prompt, (int)strlen(prompt),
                            toks.data(), (int)toks.size(), true, false);
    if (nt < 0) nt = 0;
    toks.resize(nt);

    printf("=== DEBUG GRAPH ===\n");
    auto batch = llama_batch_get_one(toks.data(), (int)toks.size());
    llama_decode(ctx, batch);
    printf("=== END DEBUG ===\n");

    llama_free(ctx);
    llama_model_free(m);
    llama_backend_free();
    return 0;
}