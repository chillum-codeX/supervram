# llama.cpp integration map

Pinned upstream revision: `ce8caa6e60a03093351d6016a818720e0d46f0fb` (2026-09-20).

## Qwen3 MoE tensors

- `src/models/qwen3moe.cpp::llama_model_qwen3moe::load_arch_tensors`
  - Router: `blk.<layer>.ffn_gate_inp.weight`, shape `{n_embd,n_expert}`.
  - Expert stacks: `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`; expert index is dimension 2 and each expert is a contiguous `nb[2]` byte slice.
- `src/models/qwen3moe.cpp::llama_model_qwen3moe::graph::graph`
  - Calls generic `build_moe_ffn` for each layer.
- `src/llama-graph.cpp::llm_graph_context::build_moe_ffn`
  - Router logits -> softmax -> top-k IDs -> three `MUL_MAT_ID` expert operations.
  - Up and gate precede SwiGLU; down can be fetched while they compute.
- `ggml/src/ggml-cuda/topk-moe.cu::ggml_cuda_op_topk_moe`
  - Produces routing weights and device-resident I32 expert IDs.
- `ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_mul_mat_id`
  - CUDA expert-matmul dispatch.
- `ggml/src/ggml-cuda/mmid.cu::mm_ids_helper`
  - Compacts token rows by selected expert.

## Loading and mappings

- `src/llama-model-loader.cpp::llama_model_loader::lazy_read::add`
  - Existing demand-paged tensor range registration.
- `src/llama-model-loader.cpp::llama_model_loader::init_mappings`
  - Builds split-GGUF-aware mappings.
- `src/llama-mmap.cpp::llama_mmap::impl`
  - Excludes lazy ranges from eager prefetch and applies random-access advice.
- `src/llama-model-loader.cpp::llama_model_loader::load_all_data`
  - Existing four-buffer pinned staging pipeline is load-time only, not inference-time paging.

## Scheduler insertion point for a true GPU cache

- `ggml/src/ggml-backend.cpp::ggml_backend_sched_compute_splits`, around the `GGML_OP_MUL_MAT_ID` host-weight special case.
  - It already downloads selected IDs and copies only selected expert ranges.
  - It still allocates a full-shaped destination tensor and synchronizes IDs, so it is not a bounded expert cache.
- Required future extension:
  1. Fixed compact GPU slots per layer and projection (gate/up/down).
  2. Logical expert -> slot map and ID remapping before MMID.
  3. Router-ready event, transfer stream, and expert-ready event.
  4. Separate gate/up and down readiness.
  5. Explicit CUDA graph compatibility handling.

## Current patch

`patches/0001-qwen3-moe-mmap-expert-storage.patch` adds an explicit, disabled-by-default first integration:

- `include/llama.h`: `llama_moe_expert_storage` and `llama_model_params::moe_expert_storage`.
- `src/llama-model.cpp`: resident default.
- `src/llama.cpp`: validates Qwen3/file-backed/mmap and forces lazy mode only when requested.
- `src/models/qwen3moe.cpp`: marks only gate/up/down expert stacks `TENSOR_READ_LAZY`.
- `common/common.h`, `common/common.cpp`, `common/arg.cpp`: `--moe-expert-storage resident|mmap` plumbing.

This path is SSD-backed CPU expert execution through demand-paged GGUF mappings. It is an honest baseline and functional precursor, not the claimed final GPU/NVMe cache.
