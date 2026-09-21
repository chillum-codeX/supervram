// svram-verify: deterministic greedy decoding gate for SuperVRAM storage modes.
//
// Loads a GGUF with a chosen expert-storage mode / GPU offload configuration, runs a fixed
// prompt followed by N greedy decode steps and emits, per step, the sampled token id and an
// FNV-1a hash of the full logits vector. Optionally dumps the raw logits so two runs can be
// compared within a numeric tolerance (CPU vs CUDA kernels are not bit-identical).
//
// usage: svram-verify --model M.gguf [--storage resident|mmap|cache] [--ngl N] [--cpu-moe]
//                     [--n-cpu-moe N] [--cache-mib N] [--cache-policy lru|lfu]
//                     [--n-predict N] [--prompt TEXT] [--dump-logits FILE] [--json FILE]

#include "ggml-backend.h"
#include "llama.h"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <random>
#include <iterator>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

uint64_t fnv1a(const float * data, size_t n) {
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, &data[i], sizeof(bits));
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= static_cast<uint8_t>(bits >> shift);
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct options {
    std::string model;
    std::string storage = "resident";
    int ngl = 999;
    bool cpu_moe = false;
    int n_cpu_moe = 0;
    int cache_mib = 0;
    std::string cache_policy = "lru";
    bool zerocopy = false;     // --zerocopy: experts read in place from pinned RAM through per-expert pointers
    bool pinned_moe = false;   // --pinned-moe: place expert tensors in pinned host RAM (CUDA_Host)
    bool direct_io = false;    // --direct-io: O_DIRECT reads for cache misses (also SVRAM_DIRECT_IO=1)
    int io_threads = 0;        // --io-threads N (0 = default)
    int staging_mib = 0;       // --staging-mib N (0 = default)
    std::string trace;         // --trace FILE: record per-token expert selections
    int n_predict = 32;
    int n_ctx = 1024;
    int n_batch = 512;
    int n_ubatch = 64;
    std::string prompt = "The tallest mountain in the world is";
    std::string prompt_file;   // --prompt-file: read the prompt text from a file
    int prompt_tokens = 0;     // --prompt-tokens N: use exactly N prompt tokens (truncate the file, then append the suffix)
    std::string prompt_suffix; // --prompt-suffix TEXT: appended after the truncated file text
    bool ignore_eos = false;   // --ignore-eos: never stop at an end-of-generation token (throughput tests)
    std::string progress;      // --progress FILE: append a line per prefill chunk
    float bias = 0.0f;         // --bias F: cache-aware routing (lossy)
    bool bias_mul = false;     // --bias-mul: multiplicative bias (relative margin)
    bool tokenize_only = false; // --tokenize-only: with --dump-prompt-tokens, stop after writing the tokens
    std::string dump_prompt;   // --dump-prompt-tokens FILE: write the tokenized prompt (ids, space separated)
    std::string warm;          // --warm FILE: fill the VRAM expert slots from a usage profile after the prompt, before decoding
    float temp = 0.0f;         // --temp F (0 = greedy); with --top-k/--top-p/--repeat-penalty/--seed: sampled output that does not loop
    int top_k = 40;
    float top_p = 0.95f;
    float repeat_penalty = 1.1f;
    int repeat_last_n = 256;
    unsigned seed = 1;
    std::string ctk = "f16";   // --ctk/--ctv: KV cache types (f16, q8_0, q4_0)
    std::string ctv = "f16";
    std::string dump_logits;
    std::string force_tokens; // teacher forcing: feed these token ids instead of the argmax
    std::string json;
    int threads = 12;
};

bool parse(int argc, char ** argv, options & o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](std::string & dst) { if (i + 1 >= argc) return false; dst = argv[++i]; return true; };
        std::string v;
        if (a == "--model") { if (!next(o.model)) return false; }
        else if (a == "--storage") { if (!next(o.storage)) return false; }
        else if (a == "--ngl") { if (!next(v)) return false; o.ngl = std::atoi(v.c_str()); }
        else if (a == "--cpu-moe") { o.cpu_moe = true; }
        else if (a == "--n-cpu-moe") { if (!next(v)) return false; o.n_cpu_moe = std::atoi(v.c_str()); }
        else if (a == "--cache-mib") { if (!next(v)) return false; o.cache_mib = std::atoi(v.c_str()); }
        else if (a == "--cache-policy") { if (!next(o.cache_policy)) return false; }
        else if (a == "--direct-io") { o.direct_io = true; }
        else if (a == "--zerocopy") { o.zerocopy = true; }
        else if (a == "--pinned-moe") { o.pinned_moe = true; }
        else if (a == "--io-threads") { if (!next(v)) return false; o.io_threads = std::atoi(v.c_str()); }
        else if (a == "--staging-mib") { if (!next(v)) return false; o.staging_mib = std::atoi(v.c_str()); }
        else if (a == "--trace") { if (!next(o.trace)) return false; }
        else if (a == "--n-predict") { if (!next(v)) return false; o.n_predict = std::atoi(v.c_str()); }
        else if (a == "--n-ctx") { if (!next(v)) return false; o.n_ctx = std::atoi(v.c_str()); }
        else if (a == "--n-batch") { if (!next(v)) return false; o.n_batch = std::atoi(v.c_str()); }
        else if (a == "--n-ubatch") { if (!next(v)) return false; o.n_ubatch = std::atoi(v.c_str()); }
        else if (a == "--threads") { if (!next(v)) return false; o.threads = std::atoi(v.c_str()); }
        else if (a == "--prompt") { if (!next(o.prompt)) return false; }
        else if (a == "--prompt-file") { if (!next(o.prompt_file)) return false; }
        else if (a == "--prompt-tokens") { if (!next(v)) return false; o.prompt_tokens = std::atoi(v.c_str()); }
        else if (a == "--prompt-suffix") { if (!next(o.prompt_suffix)) return false; }
        else if (a == "--ignore-eos") { o.ignore_eos = true; }
        else if (a == "--progress") { if (!next(o.progress)) return false; }
        else if (a == "--bias-mul") { o.bias_mul = true; }
        else if (a == "--bias") { if (!next(v)) return false; o.bias = (float) std::atof(v.c_str()); }
        else if (a == "--tokenize-only") { o.tokenize_only = true; }
        else if (a == "--dump-prompt-tokens") { if (!next(o.dump_prompt)) return false; }
        else if (a == "--warm") { if (!next(o.warm)) return false; }
        else if (a == "--temp") { if (!next(v)) return false; o.temp = (float) std::atof(v.c_str()); }
        else if (a == "--top-k") { if (!next(v)) return false; o.top_k = std::atoi(v.c_str()); }
        else if (a == "--top-p") { if (!next(v)) return false; o.top_p = (float) std::atof(v.c_str()); }
        else if (a == "--repeat-penalty") { if (!next(v)) return false; o.repeat_penalty = (float) std::atof(v.c_str()); }
        else if (a == "--repeat-last-n") { if (!next(v)) return false; o.repeat_last_n = std::atoi(v.c_str()); }
        else if (a == "--seed") { if (!next(v)) return false; o.seed = (unsigned) std::atoi(v.c_str()); }
        else if (a == "--ctk") { if (!next(o.ctk)) return false; }
        else if (a == "--ctv") { if (!next(o.ctv)) return false; }
        else if (a == "--dump-logits") { if (!next(o.dump_logits)) return false; }
        else if (a == "--force-tokens") { if (!next(o.force_tokens)) return false; }
        else if (a == "--json") { if (!next(o.json)) return false; }
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return false; }
    }
    return !o.model.empty();
}

} // namespace

int main(int argc, char ** argv) {
    options o;
    if (!parse(argc, argv, o)) {
        std::fprintf(stderr, "usage: %s --model M.gguf [--storage resident|mmap|cache] [--ngl N] [--cpu-moe] [--n-cpu-moe N] [--cache-mib N] [--n-predict N] [--prompt TEXT] [--dump-logits FILE] [--json FILE]\n", argv[0]);
        return 2;
    }
    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = o.ngl;
    if (o.storage == "resident") {
        mparams.moe_expert_storage = LLAMA_MOE_EXPERT_STORAGE_RESIDENT;
    } else if (o.storage == "mmap") {
        mparams.moe_expert_storage = LLAMA_MOE_EXPERT_STORAGE_MMAP;
#ifdef SVRAM_HAVE_EXPERT_CACHE
    } else if (o.storage == "cache") {
        mparams.moe_expert_storage = LLAMA_MOE_EXPERT_STORAGE_CACHE;
#endif
    } else {
        std::fprintf(stderr, "unsupported storage mode %s in this build\n", o.storage.c_str());
        return 2;
    }

    // --cpu-moe / --n-cpu-moe: keep expert tensors on the CPU buffer type (same regexes as common/arg.cpp)
    std::vector<std::string> patterns;
    std::vector<llama_model_tensor_buft_override> overrides;
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
    ggml_backend_buffer_type_t override_buft = cpu_buft;
    if (o.pinned_moe) {
        ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        override_buft = gpu ? ggml_backend_dev_host_buffer_type(gpu) : nullptr;
        if (override_buft == nullptr) {
            std::fprintf(stderr, "--pinned-moe needs a GPU with a pinned host buffer type\n");
            return 1;
        }
        patterns.push_back("\\.ffn_(up|down|gate)_(ch|)exps");
    } else if (o.cpu_moe) {
        patterns.push_back("\\.ffn_(up|down|gate)_(ch|)exps");
    } else if (o.n_cpu_moe > 0) {
        for (int i = 0; i < o.n_cpu_moe; ++i) {
            patterns.push_back("blk\\." + std::to_string(i) + "\\.ffn_(up|down|gate)_(ch|)exps");
        }
    }
    for (const auto & p : patterns) {
        overrides.push_back({ p.c_str(), override_buft });
    }
    if (!overrides.empty()) {
        overrides.push_back({ nullptr, nullptr });
        mparams.tensor_buft_overrides = overrides.data();
    }

    const double t_load0 = now_ms();
    llama_model * model = llama_model_load_from_file(o.model.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "failed to load model\n");
        return 1;
    }
    const double load_ms = now_ms() - t_load0;

    auto cparams = llama_context_default_params();
    cparams.n_ctx = o.n_ctx;
    cparams.n_batch = o.n_batch;
    cparams.n_ubatch = o.n_ubatch;
    auto kv_type = [](const std::string & n) { return n == "q8_0" ? GGML_TYPE_Q8_0 : n == "q4_0" ? GGML_TYPE_Q4_0 : GGML_TYPE_F16; };
    cparams.type_k = kv_type(o.ctk);
    cparams.type_v = kv_type(o.ctv);
    cparams.n_threads = o.threads;
    cparams.n_threads_batch = o.threads;
#ifdef SVRAM_HAVE_EXPERT_CACHE
    if (o.storage == "cache" || o.zerocopy) {
        cparams.moe_expert_zerocopy = o.zerocopy ? 1 : 0;
        cparams.moe_expert_cache_bias = o.bias;
        cparams.moe_expert_cache_bias_mul = o.bias_mul ? 1 : 0;
        cparams.moe_expert_cache_bytes = (size_t) o.cache_mib << 20;
        cparams.moe_expert_cache_policy = o.cache_policy == "lfu" ? LLAMA_MOE_EXPERT_CACHE_POLICY_LFU : LLAMA_MOE_EXPERT_CACHE_POLICY_LRU;
        cparams.moe_expert_direct_io = o.direct_io ? 1 : 0;
        cparams.moe_expert_io_threads = o.io_threads;
        cparams.moe_expert_staging_mib = o.staging_mib;
        cparams.moe_expert_trace = o.trace.empty() ? nullptr : o.trace.c_str();
    }
#endif
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "failed to create context\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    auto tokenize = [&](const std::string & text, bool add_special, bool parse_special) {
        std::vector<llama_token> out(text.size() + 16);
        int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), out.data(), (int32_t) out.size(), add_special, parse_special);
        if (n < 0) {
            out.resize(-n);
            n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), out.data(), (int32_t) out.size(), add_special, parse_special);
        }
        out.resize(std::max(n, 0));
        return out;
    };
    std::vector<llama_token> tokens;
    if (!o.prompt_file.empty()) {
        std::ifstream pf(o.prompt_file);
        std::string text((std::istreambuf_iterator<char>(pf)), std::istreambuf_iterator<char>());
        tokens = tokenize(text, true, false);
        if (o.prompt_tokens > 0) {
            const std::vector<llama_token> suffix = o.prompt_suffix.empty() ? std::vector<llama_token>() : tokenize(o.prompt_suffix, false, false);
            const size_t keep = (size_t) o.prompt_tokens > suffix.size() ? (size_t) o.prompt_tokens - suffix.size() : 0;
            if (tokens.size() < keep) {
                std::fprintf(stderr, "prompt file has only %zu tokens, %zu requested\n", tokens.size(), keep);
                return 1;
            }
            tokens.resize(keep);
            tokens.insert(tokens.end(), suffix.begin(), suffix.end());
        }
    } else {
        tokens = tokenize(o.prompt, true, true);
    }
    const int n_prompt = (int) tokens.size();
    if (!o.dump_prompt.empty()) {
        std::ofstream dp(o.dump_prompt);
        for (size_t i = 0; i < tokens.size(); ++i) dp << (i ? " " : "") << tokens[i];
    }
    if (o.tokenize_only) {
        return 0;
    }
    if (n_prompt + o.n_predict > o.n_ctx) {
        std::fprintf(stderr, "warning: prompt (%d) + n_predict (%d) exceeds n_ctx (%d)\n", n_prompt, o.n_predict, o.n_ctx);
    }
    std::vector<char> is_eog(n_vocab, 0);
    if (o.ignore_eos) {
        for (int32_t i = 0; i < n_vocab; ++i) is_eog[i] = llama_vocab_is_eog(vocab, i) ? 1 : 0;
    }

    std::ofstream dump;
    if (!o.dump_logits.empty()) {
        dump.open(o.dump_logits, std::ios::binary);
    }

    double warm_ms = -1.0;
#ifdef SVRAM_HAVE_EXPERT_CACHE
    if (!o.warm.empty()) {
        warm_ms = llama_moe_expert_cache_warm(ctx, o.warm.c_str()); // before the prompt: prompt batches then reuse the resident experts
        std::printf("warm_start_ms=%.0f (before prefill)\n", warm_ms);
    }
#endif
    std::vector<llama_token> forced;
    if (!o.force_tokens.empty()) {
        std::ifstream f(o.force_tokens);
        long long t;
        while (f >> t) forced.push_back((llama_token) t);
        if (forced.empty()) {
            std::fprintf(stderr, "no tokens in %s\n", o.force_tokens.c_str());
            return 1;
        }
    }
    std::vector<llama_token> argmax_tokens;
    std::vector<float> chosen_logprobs;
    std::vector<llama_token> generated;
    std::vector<std::string> hashes;
    std::vector<double> step_ms;

    const double t_pp0 = now_ms();
    std::vector<double> prefill_chunk_ms;
    for (int i0 = 0; i0 < n_prompt; i0 += o.n_batch) {
        const int n = std::min(o.n_batch, n_prompt - i0);
        const double tc = now_ms();
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + i0, n)) != 0) {
            std::fprintf(stderr, "prompt decode failed at token %d\n", i0);
            return 1;
        }
        llama_synchronize(ctx); // llama_decode returns once GPU work is queued; time the completed step
        prefill_chunk_ms.push_back(now_ms() - tc);
        if (!o.progress.empty()) {
            std::ofstream pg(o.progress, std::ios::app);
            pg << "prefill " << (i0 + n) << "/" << n_prompt << " chunk_ms=" << prefill_chunk_ms.back() << " elapsed_s=" << (now_ms() - t_pp0) / 1000.0 << "\n";
        }
    }
    const double prompt_ms = now_ms() - t_pp0;
#ifdef SVRAM_HAVE_EXPERT_CACHE
    if (!o.warm.empty() && warm_ms < 0.0) {
        warm_ms = llama_moe_expert_cache_warm(ctx, o.warm.c_str()); // the layout only exists after the first graph: retry after the prompt
        std::printf("warm_start_ms=%.0f (after prefill)\n", warm_ms);
    }
#endif

    for (int step = 0; step < o.n_predict; ++step) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (dump.is_open()) {
            dump.write(reinterpret_cast<const char *>(logits), (std::streamsize) n_vocab * sizeof(float));
        }
        char h[32];
        std::snprintf(h, sizeof(h), "%016llx", (unsigned long long) fnv1a(logits, n_vocab));
        hashes.emplace_back(h);
        int32_t best = -1, mx = 0;
        for (int32_t i = 0; i < n_vocab; ++i) {
            if (logits[i] > logits[mx]) mx = i;
            if (o.ignore_eos && is_eog[i]) continue;
            if (best < 0 || logits[i] > logits[best]) best = i;
        }
        if (!forced.empty() && (size_t) step >= forced.size()) {
            break;
        }
        int32_t chosen = forced.empty() ? best : forced[step];
        if (forced.empty() && o.temp > 0.0f) {
            // sampled output: repetition penalty, top-k, temperature, top-p (fixed seed, so runs are repeatable)
            static std::mt19937 rng(o.seed);
            std::vector<float> l(logits, logits + n_vocab);
            const int hist = std::min<int>((int) (tokens.size() + generated.size()), o.repeat_last_n);
            for (int h = 0; h < hist; ++h) {
                const size_t idx = tokens.size() + generated.size() - 1 - h;
                const llama_token t = idx < tokens.size() ? tokens[idx] : generated[idx - tokens.size()];
                l[t] = l[t] > 0.0f ? l[t] / o.repeat_penalty : l[t] * o.repeat_penalty;
            }
            if (o.ignore_eos) {
                for (int32_t i = 0; i < n_vocab; ++i) if (is_eog[i]) l[i] = -1e30f;
            }
            std::vector<int32_t> idx(n_vocab);
            for (int32_t i = 0; i < n_vocab; ++i) idx[i] = i;
            const int k = std::max(1, std::min(o.top_k, (int) n_vocab));
            std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int32_t a, int32_t b) { return l[a] > l[b]; });
            std::vector<double> pr(k);
            double z = 0.0;
            for (int i = 0; i < k; ++i) { pr[i] = std::exp((double) (l[idx[i]] - l[idx[0]]) / o.temp); z += pr[i]; }
            double cum = 0.0; int keep = k;
            for (int i = 0; i < k; ++i) { pr[i] /= z; cum += pr[i]; if (cum >= o.top_p) { keep = i + 1; break; } }
            double zz = 0.0;
            for (int i = 0; i < keep; ++i) zz += pr[i];
            double u = std::uniform_real_distribution<double>(0.0, zz)(rng);
            chosen = idx[keep - 1];
            for (int i = 0; i < keep; ++i) { u -= pr[i]; if (u <= 0.0) { chosen = idx[i]; break; } }
        }
        {
            double sum = 0.0;
            for (int32_t i = 0; i < n_vocab; ++i) sum += std::exp((double) logits[i] - (double) logits[mx]);
            chosen_logprobs.push_back((float) ((double) logits[chosen] - (double) logits[mx] - std::log(sum)));
            argmax_tokens.push_back(best);
        }
        generated.push_back(chosen);
        if (!o.ignore_eos && llama_vocab_is_eog(vocab, chosen)) {
            break;
        }
        const double t0 = now_ms();
        if (llama_decode(ctx, llama_batch_get_one(&chosen, 1)) != 0) {
            std::fprintf(stderr, "decode failed at step %d\n", step);
            return 1;
        }
        llama_synchronize(ctx);
        step_ms.push_back(now_ms() - t0);
    }

    // decoded text
    std::string text;
    for (llama_token t : generated) {
        char buf[256];
        int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
        if (n > 0) text.append(buf, n);
    }

    double decode_sum = 0.0;
    for (double v : step_ms) decode_sum += v;
    const double tps = step_ms.empty() ? 0.0 : 1000.0 * step_ms.size() / decode_sum;

    std::printf("mode=%s ngl=%d cpu_moe=%d n_cpu_moe=%d cache_mib=%d prompt_tokens=%d generated=%zu load_ms=%.0f prompt_ms=%.1f decode_tps=%.2f\n",
                o.storage.c_str(), o.ngl, (int) o.cpu_moe, o.n_cpu_moe, o.cache_mib, n_prompt, generated.size(), load_ms, prompt_ms, tps);
#ifdef SVRAM_HAVE_EXPERT_CACHE
    {
        ggml_backend_sched_expert_cache_stats st{};
        if (llama_get_moe_expert_cache_stats(ctx, &st)) {
            const double hit_rate = st.accesses ? 100.0 * st.hits / st.accesses : 0.0;
            std::printf("cache_stats accesses=%llu hits=%llu misses=%llu evictions=%llu hit_rate=%.1f bytes_h2d=%llu n_slots=%llu host_ms=%.1f\n",
                        (unsigned long long) st.accesses, (unsigned long long) st.hits, (unsigned long long) st.misses,
                        (unsigned long long) st.evictions, hit_rate, (unsigned long long) st.bytes_h2d,
                        (unsigned long long) st.n_slots_total, st.host_ms);
            std::printf("direct_io_stats bytes_ssd=%llu io_ms=%.1f\n", (unsigned long long) st.bytes_ssd, st.io_ms);
        }
    }
#endif
    std::printf("tokens=");
    for (size_t i = 0; i < generated.size(); ++i) std::printf("%s%d", i ? "," : "", generated[i]);
    std::printf("\nlogits_hash_step0=%s\ntext=%s\n", hashes.empty() ? "" : hashes[0].c_str(), text.c_str());

    if (!o.json.empty()) {
        std::ofstream out(o.json);
        out << "{\n  \"schema_version\": 1,\n  \"evidence_class\": \"measured_rtx3090\",\n";
        out << "  \"mode\": \"" << o.storage << "\", \"ngl\": " << o.ngl << ", \"cpu_moe\": " << (o.cpu_moe ? "true" : "false")
            << ", \"n_cpu_moe\": " << o.n_cpu_moe << ", \"cache_mib\": " << o.cache_mib << ", \"cache_policy\": \"" << o.cache_policy << "\",\n";
        out << "  \"model\": \"" << o.model << "\",\n  \"prompt_tokens\": " << n_prompt << ",\n";
        out << "  \"load_ms\": " << load_ms << ",\n  \"prompt_ms\": " << prompt_ms << ",\n  \"decode_tps\": " << tps << ",\n";
        out << "  \"tokens\": [";
        for (size_t i = 0; i < generated.size(); ++i) out << (i ? "," : "") << generated[i];
        out << "],\n  \"argmax_tokens\": [";
        for (size_t i = 0; i < argmax_tokens.size(); ++i) out << (i ? "," : "") << argmax_tokens[i];
        out << "],\n  \"chosen_logprobs\": [";
        for (size_t i = 0; i < chosen_logprobs.size(); ++i) out << (i ? "," : "") << chosen_logprobs[i];
        out << "],\n  \"logits_hashes\": [";
        for (size_t i = 0; i < hashes.size(); ++i) out << (i ? "," : "") << '"' << hashes[i] << '"';
        out << "],\n  \"warm_ms\": " << warm_ms << ",\n  \"prefill_chunk_ms\": [";
        for (size_t i = 0; i < prefill_chunk_ms.size(); ++i) out << (i ? "," : "") << prefill_chunk_ms[i];
        out << "],\n  \"step_ms\": [";
        for (size_t i = 0; i < step_ms.size(); ++i) out << (i ? "," : "") << step_ms[i];
        out << "]\n}\n";
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
