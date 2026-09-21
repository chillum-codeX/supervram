#include "llama.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

int main(int argc, char ** argv) {
    if (argc != 3 || (std::strcmp(argv[2], "resident") != 0 && std::strcmp(argv[2], "mmap") != 0)) {
        std::fprintf(stderr, "usage: %s MODEL.gguf resident|mmap\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.moe_expert_storage = std::strcmp(argv[2], "mmap") == 0 ? LLAMA_MOE_EXPERT_STORAGE_MMAP : LLAMA_MOE_EXPERT_STORAGE_RESIDENT;
    llama_model * model = llama_model_load_from_file(argv[1], params);
    if (!model) {
        std::fprintf(stderr, "failed to load model\n");
        llama_backend_free();
        return 1;
    }
    auto cparams = llama_context_default_params();
    cparams.n_ctx = 32;
    cparams.n_batch = 8;
    cparams.n_ubatch = 8;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    llama_token token = 1;
    if (llama_decode(ctx, llama_batch_get_one(&token, 1)) != 0) {
        std::fprintf(stderr, "decode failed\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    const float * logits = llama_get_logits_ith(ctx, 0);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    double checksum = 0.0;
    uint64_t hash = 1469598103934665603ULL;
    for (int32_t i = 0; i < n_vocab; ++i) {
        checksum += static_cast<double>(i + 1) * logits[i];
        uint32_t bits = 0;
        std::memcpy(&bits, &logits[i], sizeof(bits));
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= static_cast<uint8_t>(bits >> shift);
            hash *= 1099511628211ULL;
        }
    }
    std::printf("mode=%s layers=%d parameters=%lld logits_checksum=%.17g logits_hash=%016llx\n", argv[2], llama_model_n_layer(model), static_cast<long long>(llama_model_n_params(model)), checksum, static_cast<unsigned long long>(hash));
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return std::isfinite(checksum) ? 0 : 1;
}
