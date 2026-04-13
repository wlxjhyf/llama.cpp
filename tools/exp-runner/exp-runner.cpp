// exp-runner: end-to-end timing for VRAM migration experiments.
//
// Runs phase1 → transition → phase2 in a single process with a single wall-clock.
// Transition is either:
//   --mode dynamic  : llama_model_migrate_weights() to ngl2, then first decode (warmup)
//   --mode static   : no migration, continue at ngl1 (ngl2 ignored)
//   --mode breakdown: measure each sub-component separately.
//
// Usage:
//   exp-runner -m <model.gguf> --ngl1 <n> --ngl2 <n> --mode <dynamic|static|breakdown>
//              --n-phase1 <tokens> --n-phase2 <tokens> [--n-prompt <tokens>]
//
// Output (stdout, one line):
//   dynamic/static: phase1_ms <f>  transition_ms <f>  phase2_ms <f>  total_ms <f>
//   breakdown:      model_load_ms <f>  ctx_init_ms <f>
//                   prefill_ms <f>  graph_capture_ms <f>
//                   per_token_ms <f>  steady_decode_ms <f>  total_ms <f>
//                   [migrate_weights_ms <f>  migrate_kvcache_ms <f>
//                    post_prefill_ms <f>  post_graph_capture_ms <f>
//                    post_per_token_ms <f>  post_steady_decode_ms <f>]
//
// breakdown field definitions:
//   prefill_ms        : llama_decode() for the full prompt batch (variable batch size)
//   graph_capture_ms  : first llama_decode() with batch_size=1 — CUDA graph is captured here
//   per_token_ms      : median decode latency for tokens 2..N (steady state, graph already warm)
//   steady_decode_ms  : total wall time for ALL single-token decodes (token 1 through N)
//   total_ms          : wall clock from just before model load to end of steady-state decode
//
// Reconciliation identity (should hold within ~1%):
//   total_ms ≈ model_load_ms + ctx_init_ms + prefill_ms + graph_capture_ms + steady_decode_ms
//
// Transition timing covers: migrate_weights + migrate_kvcache + first full decode
// (prefill + 1 generated token), so warmup is included on the same clock as migration.

#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ── helpers ──────────────────────────────────────────────────────────────────

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

// Prefill prompt then generate n_gen tokens. Returns wall-clock ms, or -1 on error.
// If t_first_out is non-null, also writes the time of the very first generated
// token (prefill + sample) into *t_first_out — used to split transition from phase2.
static double run_phase(llama_context * ctx,
                        const llama_model * model,
                        const std::vector<llama_token> & prompt,
                        int n_gen,
                        double * t_first_out = nullptr) {
    llama_memory_clear(llama_get_memory(ctx), false);

    const int64_t t0 = ggml_time_us();

    // prefill
    {
        llama_batch batch = llama_batch_get_one(
            const_cast<llama_token *>(prompt.data()), (int)prompt.size());
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "prefill failed\n");
            return -1.0;
        }
    }

    auto * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    llama_token cur = llama_sampler_sample(smpl, ctx, -1);
    llama_sampler_free(smpl);

    if (t_first_out) {
        *t_first_out = (ggml_time_us() - t0) / 1000.0;
    }

    for (int i = 0; i < n_gen - 1; ++i) {
        if (llama_vocab_is_eog(llama_model_get_vocab(model), cur)) break;
        llama_batch batch = llama_batch_get_one(&cur, 1);
        if (llama_decode(ctx, batch) != 0) break;
        const float * logits = llama_get_logits(ctx);
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        cur = (llama_token)(std::max_element(logits, logits + n_vocab) - logits);
    }

    return (ggml_time_us() - t0) / 1000.0;
}

// Like run_phase but also returns the last generated token.
// Used by dynamic_kv mode to hand off the token across migration.
static std::pair<double, llama_token> run_phase_ret(
        llama_context * ctx,
        const llama_model * model,
        const std::vector<llama_token> & prompt,
        int n_gen) {
    llama_memory_clear(llama_get_memory(ctx), false);
    const int64_t t0 = ggml_time_us();

    llama_batch batch = llama_batch_get_one(
        const_cast<llama_token *>(prompt.data()), (int)prompt.size());
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "run_phase_ret: prefill failed\n");
        return {-1.0, 0};
    }

    auto * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    llama_token cur = llama_sampler_sample(smpl, ctx, -1);
    llama_sampler_free(smpl);

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    for (int i = 0; i < n_gen - 1; ++i) {
        if (llama_vocab_is_eog(llama_model_get_vocab(model), cur)) break;
        llama_batch b = llama_batch_get_one(&cur, 1);
        if (llama_decode(ctx, b) != 0) break;
        const float * logits = llama_get_logits(ctx);
        cur = (llama_token)(std::max_element(logits, logits + n_vocab) - logits);
    }

    return {(ggml_time_us() - t0) / 1000.0, cur};
}

// Continue decoding from token `cur` for n_gen steps.
// Does NOT clear the KV cache or re-prefill — resumes from the current n_past.
// Used by dynamic_kv mode after weight+KV migration.
static double continue_phase(llama_context * ctx,
                             const llama_model * model,
                             llama_token cur,
                             int n_gen) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int64_t t0 = ggml_time_us();
    for (int i = 0; i < n_gen; ++i) {
        if (llama_vocab_is_eog(llama_model_get_vocab(model), cur)) break;
        llama_batch b = llama_batch_get_one(&cur, 1);
        if (llama_decode(ctx, b) != 0) break;
        const float * logits = llama_get_logits(ctx);
        cur = (llama_token)(std::max_element(logits, logits + n_vocab) - logits);
    }
    return (ggml_time_us() - t0) / 1000.0;
}

// ── breakdown helpers ─────────────────────────────────────────────────────────

// Run prefill (full prompt batch). Returns ms elapsed.
// Writes the token sampled from prefill logits to *cur_out.
// Does NOT run any batch_size=1 decode — CUDA graph is NOT captured here.
static double run_prefill(llama_context * ctx,
                          const std::vector<llama_token> & prompt,
                          llama_token * cur_out) {
    llama_memory_clear(llama_get_memory(ctx), false);
    const int64_t t0 = ggml_time_us();

    llama_batch batch = llama_batch_get_one(
        const_cast<llama_token *>(prompt.data()), (int)prompt.size());
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "prefill: llama_decode failed\n");
        return -1.0;
    }

    auto * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    *cur_out = llama_sampler_sample(smpl, ctx, -1);
    llama_sampler_free(smpl);

    return (ggml_time_us() - t0) / 1000.0;
}

// Run exactly one batch_size=1 decode step starting from *cur.
// This is where CUDA graph capture fires (first call after a context clear or ngl change).
// Returns ms elapsed. Writes the next token to *cur.
static double run_single_decode(llama_context * ctx,
                                const llama_model * model,
                                llama_token * cur) {
    if (llama_vocab_is_eog(llama_model_get_vocab(model), *cur)) return 0.0;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_batch batch = llama_batch_get_one(cur, 1);
    const int64_t t0 = ggml_time_us();
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "run_single_decode: llama_decode failed\n");
        return -1.0;
    }
    const float * logits = llama_get_logits(ctx);
    *cur = (llama_token)(std::max_element(logits, logits + n_vocab) - logits);
    return (ggml_time_us() - t0) / 1000.0;
}

// Run n_gen batch_size=1 decode steps starting from *cur.
// Returns {median_ms, total_ms}. Writes the final token to *cur.
// Assumes CUDA graph is already captured (no warmup overhead expected).
static std::pair<double, double> run_steady_state(llama_context * ctx,
                                                  const llama_model * model,
                                                  llama_token * cur,
                                                  int n_gen) {
    if (n_gen <= 0) return {0.0, 0.0};
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<double> times;
    times.reserve(n_gen);

    const int64_t t_total_start = ggml_time_us();
    for (int i = 0; i < n_gen; ++i) {
        if (llama_vocab_is_eog(llama_model_get_vocab(model), *cur)) break;
        llama_batch batch = llama_batch_get_one(cur, 1);
        const int64_t t0 = ggml_time_us();
        if (llama_decode(ctx, batch) != 0) break;
        const float * logits = llama_get_logits(ctx);
        *cur = (llama_token)(std::max_element(logits, logits + n_vocab) - logits);
        times.push_back((ggml_time_us() - t0) / 1000.0);
    }
    const double total_ms = (ggml_time_us() - t_total_start) / 1000.0;

    if (times.empty()) return {0.0, 0.0};
    std::sort(times.begin(), times.end());
    return {times[times.size() / 2], total_ms};
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    std::string model_path;
    std::string mode   = "dynamic";   // "dynamic" | "static"
    std::string prompt = "The quick brown fox jumps over the lazy dog. "
                         "In a world where technology advances rapidly, "
                         "artificial intelligence systems must adapt continuously.";
    int ngl1      = 40;
    int ngl2      = 60;
    int n_phase1  = 100;
    int n_phase2  = 100;
    int n_ctx     = 512;
    int n_prompt  = 0;   // 0 = use full prompt text; >0 = repeat/truncate to exactly N tokens

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "-m")         && i+1<argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--ngl1")     && i+1<argc) ngl1      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ngl2")     && i+1<argc) ngl2      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mode")     && i+1<argc) mode      = argv[++i];
        else if (!strcmp(argv[i], "--n-phase1") && i+1<argc) n_phase1  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n-phase2") && i+1<argc) n_phase2  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n-ctx")    && i+1<argc) n_ctx     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n-prompt") && i+1<argc) n_prompt  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prompt")   && i+1<argc) prompt    = argv[++i];
    }

    if (model_path.empty()) {
        fprintf(stderr,
            "Usage: %s -m <model> --ngl1 <n> --ngl2 <n> --mode <dynamic|static|breakdown>\n"
            "           --n-phase1 <tokens> --n-phase2 <tokens>\n", argv[0]);
        return 1;
    }

    // ── load model ────────────────────────────────────────────────────────────
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl1;

    // t_total_start covers the entire process lifetime measured by breakdown mode.
    const int64_t t_total_start = ggml_time_us();

    const int64_t t_load_start = ggml_time_us();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    const double model_load_ms = (ggml_time_us() - t_load_start) / 1000.0;
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    // Tokenize before context creation so we can size n_ctx correctly when
    // --n-prompt is specified.
    auto prompt_tokens = tokenize(model, prompt, true);
    if (n_prompt > 0 && n_prompt != (int)prompt_tokens.size()) {
        if (n_prompt < (int)prompt_tokens.size()) {
            prompt_tokens.resize(n_prompt);
        } else {
            std::vector<llama_token> extended;
            extended.reserve(n_prompt);
            while ((int)extended.size() < n_prompt) {
                for (llama_token t : prompt_tokens) {
                    if ((int)extended.size() >= n_prompt) break;
                    extended.push_back(t);
                }
            }
            prompt_tokens = std::move(extended);
        }
        // Auto-expand n_ctx to fit prompt + both decode phases.
        const int min_ctx = (int)prompt_tokens.size() + n_phase1 + n_phase2 + 4;
        if (n_ctx < min_ctx) n_ctx = min_ctx;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = n_ctx;
    cparams.n_batch = n_ctx;

    const int64_t t_ctx_start = ggml_time_us();
    llama_context * ctx = llama_init_from_model(model, cparams);
    const double ctx_init_ms = (ggml_time_us() - t_ctx_start) / 1000.0;
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    // ── breakdown mode (early exit before phase1) ─────────────────────────────
    if (mode == "breakdown") {
        // Timing breakdown with correct attribution:
        //
        //   prefill_ms       : llama_decode() over the full prompt (variable batch).
        //                      Does NOT trigger CUDA graph capture.
        //   graph_capture_ms : first llama_decode() with batch_size=1.
        //                      CUDA graph is captured on this call.
        //   per_token_ms     : median of tokens 2..N (steady state, graph already warm).
        //   steady_decode_ms : total wall time for ALL single-token decodes (token 1..N).
        //                      steady_decode_ms = graph_capture_ms + sum(tokens 2..N).
        //   total_ms         : wall clock from just before model_load to end.
        //
        // Reconciliation identity:
        //   total_ms ≈ model_load_ms + ctx_init_ms + prefill_ms
        //              + graph_capture_ms + steady_decode_ms
        //
        // n_phase1 = number of single-token decodes after prefill
        //            (token 1 = graph capture; tokens 2..n_phase1 = steady state)
        // n_phase2 = number of post-migration single-token decodes (0 = skip migration)

        llama_token cur;

        // Step 1: prefill — full prompt batch, no graph capture.
        const double prefill_ms = run_prefill(ctx, prompt_tokens, &cur);
        if (prefill_ms < 0) return 1;

        // Step 2: first batch_size=1 decode — CUDA graph is captured here.
        const double graph_capture_ms = run_single_decode(ctx, model, &cur);
        if (graph_capture_ms < 0) return 1;

        // Step 3: steady-state — tokens 2..n_phase1 (graph already warm).
        auto [per_token_ms, steady_tail_ms] = run_steady_state(ctx, model, &cur, n_phase1 - 1);
        const double steady_decode_ms = graph_capture_ms + steady_tail_ms;

        const double total_ms = (ggml_time_us() - t_total_start) / 1000.0;

        if (ngl2 == ngl1 || n_phase2 == 0) {
            printf("model_load_ms %.1f  ctx_init_ms %.1f"
                   "  prefill_ms %.1f  graph_capture_ms %.1f"
                   "  per_token_ms %.3f  steady_decode_ms %.1f  total_ms %.1f\n",
                   model_load_ms, ctx_init_ms,
                   prefill_ms, graph_capture_ms,
                   per_token_ms, steady_decode_ms, total_ms);
        } else {
            // migration path
            const double migrate_weights_ms = llama_model_migrate_weights(model, ngl2);
            const double migrate_kvcache_ms = llama_context_migrate_kvcache(ctx);

            if (migrate_weights_ms < 0.0 || migrate_kvcache_ms < 0.0) {
                fprintf(stderr, "migration failed\n");
                return 1;
            }

            // post-migration: prefill + graph capture + steady state at ngl2
            llama_token post_cur;
            const double post_prefill_ms = run_prefill(ctx, prompt_tokens, &post_cur);
            if (post_prefill_ms < 0) return 1;

            const double post_graph_capture_ms = run_single_decode(ctx, model, &post_cur);
            if (post_graph_capture_ms < 0) return 1;

            auto [post_per_token_ms, post_steady_tail_ms] =
                run_steady_state(ctx, model, &post_cur, n_phase2 - 1);
            const double post_steady_decode_ms = post_graph_capture_ms + post_steady_tail_ms;

            printf("model_load_ms %.1f  ctx_init_ms %.1f"
                   "  prefill_ms %.1f  graph_capture_ms %.1f"
                   "  per_token_ms %.3f  steady_decode_ms %.1f"
                   "  migrate_weights_ms %.1f  migrate_kvcache_ms %.1f"
                   "  post_prefill_ms %.1f  post_graph_capture_ms %.1f"
                   "  post_per_token_ms %.3f  post_steady_decode_ms %.1f"
                   "  total_ms %.1f\n",
                   model_load_ms, ctx_init_ms,
                   prefill_ms, graph_capture_ms,
                   per_token_ms, steady_decode_ms,
                   migrate_weights_ms, migrate_kvcache_ms,
                   post_prefill_ms, post_graph_capture_ms,
                   post_per_token_ms, post_steady_decode_ms,
                   total_ms);
        }

        llama_free(ctx);
        llama_model_free(model);
        return 0;
    }

    // ── phase 1 ───────────────────────────────────────────────────────────────
    const int64_t t_wall_start = ggml_time_us();

    // ── phase 1 ───────────────────────────────────────────────────────────────
    // dynamic_kv needs the last token from phase1 to resume decode after migration.
    // All other modes use run_phase which doesn't return the last token.
    llama_token phase1_last_tok = 0;
    double t_phase1;
    if (mode == "dynamic_kv") {
        auto [ms, tok] = run_phase_ret(ctx, model, prompt_tokens, n_phase1);
        if (ms < 0) return 1;
        t_phase1 = ms;
        phase1_last_tok = tok;
    } else {
        t_phase1 = run_phase(ctx, model, prompt_tokens, n_phase1);
        if (t_phase1 < 0) return 1;
    }

    // ── transition ────────────────────────────────────────────────────────────
    const int64_t t_trans_start = ggml_time_us();
    double t_transition = 0.0;

    if (mode == "dynamic_kv") {
        // KV-preserving migration: migrate weights + KV tensors, then resume
        // decode directly from the last token of phase1. No memory clear,
        // no re-prefill — the KV cache remains valid after migration.
        const double ms_w  = llama_model_migrate_weights(model, ngl2);
        const double ms_kv = llama_context_migrate_kvcache(ctx);
        if (ms_w < 0.0 || ms_kv < 0.0) {
            fprintf(stderr, "migration failed\n");
            return 1;
        }
        t_transition = (ggml_time_us() - t_trans_start) / 1000.0;

        // Phase 2: resume decode at new ngl without re-prefill.
        const int64_t t_phase2_start = ggml_time_us();
        const double t_phase2 = continue_phase(ctx, model, phase1_last_tok, n_phase2);
        const double t_total  = (ggml_time_us() - t_wall_start) / 1000.0;

        printf("phase1_ms %.1f  transition_ms %.1f  phase2_ms %.1f  total_ms %.1f\n",
               t_phase1, t_transition, t_phase2, t_total);
        (void)t_phase2_start;

    } else if (mode == "dynamic") {
        // Legacy dynamic mode: migrate weights+KV, then clear KV and re-prefill.
        const double ms_w  = llama_model_migrate_weights(model, ngl2);
        const double ms_kv = llama_context_migrate_kvcache(ctx);
        if (ms_w < 0.0 || ms_kv < 0.0) {
            fprintf(stderr, "migration failed\n");
            return 1;
        }

        llama_memory_clear(llama_get_memory(ctx), false);
        {
            llama_batch batch = llama_batch_get_one(
                const_cast<llama_token *>(prompt_tokens.data()), (int)prompt_tokens.size());
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "post-migration prefill failed\n");
                return 1;
            }
        }
        auto * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        llama_token cur = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_free(smpl);

        t_transition = (ggml_time_us() - t_trans_start) / 1000.0;

        const int64_t t_phase2_start = ggml_time_us();
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        for (int i = 0; i < n_phase2 - 1; ++i) {
            if (llama_vocab_is_eog(llama_model_get_vocab(model), cur)) break;
            llama_batch batch = llama_batch_get_one(&cur, 1);
            if (llama_decode(ctx, batch) != 0) break;
            const float * logits = llama_get_logits(ctx);
            cur = (llama_token)(std::max_element(logits, logits + n_vocab) - logits);
        }
        const double t_phase2 = (ggml_time_us() - t_phase2_start) / 1000.0;
        const double t_total  = (ggml_time_us() - t_wall_start) / 1000.0;

        printf("phase1_ms %.1f  transition_ms %.1f  phase2_ms %.1f  total_ms %.1f\n",
               t_phase1, t_transition, t_phase2, t_total);

    } else {
        // static: no migration, just continue at ngl1
        t_transition = 0.0;
        const double t_phase2 = run_phase(ctx, model, prompt_tokens, n_phase2);
        if (t_phase2 < 0) return 1;
        const double t_total = (ggml_time_us() - t_wall_start) / 1000.0;

        printf("phase1_ms %.1f  transition_ms %.1f  phase2_ms %.1f  total_ms %.1f\n",
               t_phase1, t_transition, t_phase2, t_total);
    }

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
