// migrate-test: verify correctness and measure overhead of runtime ngl migration.
//
// Usage:
//   migrate-test -m <model.gguf> --ngl <initial> --ngl-targets <t1,t2,...>
//
// For each target ngl the tool:
//   1. Runs a fixed prompt with the current ngl and records output tokens.
//   2. Migrates to the target ngl, records migration time.
//   3. Re-runs the same prompt and checks that the output is identical.
//   4. Prints a summary table.

#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ---- helpers ----------------------------------------------------------------

static std::vector<llama_token> tokenize(const llama_model * model,
                                          const std::string & text,
                                          bool add_bos) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n = -llama_tokenize(vocab, text.c_str(), (int)text.size(),
                                  nullptr, 0, add_bos, true);
    std::vector<llama_token> tokens(n);
    llama_tokenize(vocab, text.c_str(), (int)text.size(),
                   tokens.data(), n, add_bos, true);
    return tokens;
}

// Run greedy decode for n_gen tokens starting from prompt tokens.
// Returns the generated token ids.
static std::vector<llama_token> run_greedy(llama_context * ctx,
                                            const llama_model * model,
                                            const std::vector<llama_token> & prompt,
                                            int n_gen) {
    llama_memory_clear(llama_get_memory(ctx), false);

    // prefill
    {
        llama_batch batch = llama_batch_get_one(
            const_cast<llama_token *>(prompt.data()), (int)prompt.size());
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "prefill failed\n");
            return {};
        }
    }

    std::vector<llama_token> out;
    out.reserve(n_gen);

    auto * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    llama_token cur = llama_sampler_sample(smpl, ctx, -1);
    llama_sampler_free(smpl);

    for (int i = 0; i < n_gen; ++i) {
        out.push_back(cur);
        if (llama_vocab_is_eog(llama_model_get_vocab(model), cur)) break;

        llama_batch batch = llama_batch_get_one(&cur, 1);
        if (llama_decode(ctx, batch) != 0) break;

        // greedy: pick the top logit
        const float * logits = llama_get_logits(ctx);
        const int n_vocab    = llama_vocab_n_tokens(llama_model_get_vocab(model));
        cur = (llama_token)(std::max_element(logits, logits + n_vocab) - logits);
    }
    return out;
}

// ---- main -------------------------------------------------------------------

int main(int argc, char ** argv) {
    // --- parse args ---
    std::string model_path;
    int         init_ngl = 20;
    std::vector<int> targets;
    int         n_gen    = 32;
    std::string prompt   = "The quick brown fox jumps over the lazy dog.";

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "--ngl") && i + 1 < argc) {
            init_ngl = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--ngl-targets") && i + 1 < argc) {
            char * s = argv[++i];
            char * tok = strtok(s, ",");
            while (tok) { targets.push_back(atoi(tok)); tok = strtok(nullptr, ","); }
        } else if (!strcmp(argv[i], "--n-gen") && i + 1 < argc) {
            n_gen = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) {
            prompt = argv[++i];
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "Usage: %s -m <model> --ngl <n> --ngl-targets <t1,t2,...>\n", argv[0]);
        return 1;
    }
    if (targets.empty()) { targets = {init_ngl - 10, init_ngl + 10}; }

    // --- load model ---
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = init_ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = 512;
    cparams.n_batch  = 512;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    const auto prompt_tokens = tokenize(model, prompt, true);

    // --- baseline run at init_ngl ---
    printf("\n=== Baseline run at ngl=%d ===\n", init_ngl);
    const auto baseline = run_greedy(ctx, model, prompt_tokens, n_gen);
    printf("Generated %d tokens.\n", (int)baseline.size());

    // --- migration loop ---
    printf("\n%-10s %-14s %-12s %-12s %-10s\n", "Target ngl", "Weight ms", "KV ms", "Warmup ms", "Match?");
    printf("%-10s %-14s %-12s %-12s %-10s\n", "----------", "---------", "-----", "---------", "------");

    int current_ngl = init_ngl;
    bool all_ok = true;

    for (int target : targets) {
        // migrate
        const double ms_w  = llama_model_migrate_weights(model, target);
        const double ms_kv = llama_context_migrate_kvcache(ctx);

        if (ms_w < 0.0 || ms_kv < 0.0) {
            printf("%-10d FAILED\n", target);
            all_ok = false;
            current_ngl = target;
            continue;
        }

        // Measure first-token latency after migration (includes CUDA graph warmup).
        double ms_warmup = 0.0;
        {
            llama_memory_clear(llama_get_memory(ctx), false);
            llama_batch batch = llama_batch_get_one(
                const_cast<llama_token *>(prompt_tokens.data()), (int)prompt_tokens.size());
            const int64_t t0 = ggml_time_us();
            llama_decode(ctx, batch);
            ms_warmup = (ggml_time_us() - t0) / 1000.0;
        }

        // verify (reuse the already-warmed graph)
        const auto result = run_greedy(ctx, model, prompt_tokens, n_gen);
        const bool match  = (result == baseline);
        if (!match) all_ok = false;

        printf("%-10d %-14.1f %-12.1f %-12.1f %-10s\n",
               target, ms_w, ms_kv, ms_warmup, match ? "YES" : "NO !!!");

        current_ngl = target;
    }

    printf("\n%s\n", all_ok ? "All outputs match baseline." : "WARNING: output mismatch detected!");

    llama_free(ctx);
    llama_model_free(model);
    return all_ok ? 0 : 1;
}
