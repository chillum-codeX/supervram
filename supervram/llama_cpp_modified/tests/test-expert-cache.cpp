// Synthetic, backend-agnostic unit test for the SuperVRAM compact GPU expert cache
// (ggml_backend_sched_set_expert_cache et al., see ggml-backend.h / ggml-backend.cpp).
//
// This uses two fully synthetic "fake" backends (plain malloc'd memory, no real CPU/CUDA
// device involved) so the test is fast, portable, and isolates the *scheduler's* cache
// bookkeeping from any specific backend's kernels. The "dev" backend implements just enough
// of GGML_OP_MUL_MAT_ID (naively, in plain loops) to make the routed computation observable.
//
// It verifies:
//  1. a host-resident 3-D MoE expert weight, routed through a bounded 2-slot device cache
//     across 3 sequential decode-like steps, produces results that exactly match an
//     independently computed reference for whichever logical expert each step routes to
//     (i.e. the cache copies/serves the *correct* expert's bytes, and the id remap is correct)
//  2. cumulative hit/miss/eviction counters exactly match a hand-verified LRU trace
//  3. requesting more distinct experts in one step than there are cache slots fails loudly
//     (GGML_STATUS_FAILED) instead of corrupting memory or crashing

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

// identifies whether a fake buffer-type/device plays the "host" (CPU-typed, weight owning)
// or "dev" (GPU-typed, cache owning) role, and points back at its own buffer type so that
// ggml_backend_dev_buffer_type() has something to return
struct fake_role {
    bool is_host;
    ggml_backend_buffer_type_t buft = nullptr;
};

// ---- buffer (malloc-backed, both roles share the same trivial memcpy semantics) ----

void fake_buffer_free(ggml_backend_buffer_t buffer) {
    free(buffer->context);
}

void * fake_buffer_get_base(ggml_backend_buffer_t buffer) {
    return buffer->context;
}

void fake_buffer_set_tensor(ggml_backend_buffer_t, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *) tensor->data + offset, data, size);
}

void fake_buffer_get_tensor(ggml_backend_buffer_t, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *) tensor->data + offset, size);
}

void fake_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

const char * fake_buft_name(ggml_backend_buffer_type_t buft) {
    return ((fake_role *) buft->context)->is_host ? "fake-host" : "fake-dev";
}

ggml_backend_buffer_t fake_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * mem = malloc(size);
    memset(mem, 0, size);
    ggml_backend_buffer_i iface{};
    iface.free_buffer = fake_buffer_free;
    iface.get_base    = fake_buffer_get_base;
    iface.set_tensor  = fake_buffer_set_tensor;
    iface.get_tensor  = fake_buffer_get_tensor;
    iface.clear       = fake_buffer_clear;
    return ggml_backend_buffer_init(buft, iface, mem, size);
}

size_t fake_buft_get_alignment(ggml_backend_buffer_type_t) {
    return 32;
}

bool fake_buft_is_host(ggml_backend_buffer_type_t buft) {
    return ((fake_role *) buft->context)->is_host;
}

// ---- device ----

const char * fake_dev_get_name(ggml_backend_dev_t dev) {
    return ((fake_role *) dev->context)->is_host ? "fake-host-dev" : "fake-dev-dev";
}

const char * fake_dev_get_description(ggml_backend_dev_t dev) {
    return fake_dev_get_name(dev);
}

void fake_dev_get_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    *free = 0;
    *total = 0;
}

enum ggml_backend_dev_type fake_dev_get_type_host(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_CPU;
}

enum ggml_backend_dev_type fake_dev_get_type_dev(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

void fake_dev_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name = fake_dev_get_name(dev);
    props->description = fake_dev_get_name(dev);
}

bool fake_dev_supports_op(ggml_backend_dev_t, const struct ggml_tensor *) {
    return true;
}

bool fake_dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    // mirrors the real behavior of e.g. a CUDA backend: it only "supports" its own
    // buffer type, never the plain host buffer type of another backend
    return dev->context == buft->context;
}

ggml_backend_buffer_type_t fake_dev_get_buffer_type(ggml_backend_dev_t dev) {
    return ((fake_role *) dev->context)->buft;
}

// ---- backend (stream) ----

const char * fake_backend_get_name(ggml_backend_t backend) {
    return fake_dev_get_name(backend->device);
}

void fake_backend_free(ggml_backend_t) {}

enum ggml_status fake_host_graph_compute(ggml_backend_t, struct ggml_cgraph * cgraph) {
    // the host backend only ever owns leaves (weight / inputs) in this test, never a node
    if (cgraph->n_nodes != 0) {
        fprintf(stderr, "fake_host_graph_compute: unexpected node on host backend\n");
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

// c = ggml_mul_mat_id(as, b, ids); as[cols,rows,n_rows_avail], b[cols,n_used,n_tokens],
// ids[n_used,n_tokens] i32, c[rows,n_used,n_tokens]. Deliberately naive (no quantization,
// no SIMD): this test only cares about correctness of the *cache's* data placement and id
// remap, not about kernel performance.
void compute_mul_mat_id(struct ggml_tensor * node) {
    const struct ggml_tensor * as  = node->src[0];
    const struct ggml_tensor * b   = node->src[1];
    const struct ggml_tensor * ids = node->src[2];

    const int64_t cols    = as->ne[0];
    const int64_t rows    = as->ne[1];
    const int64_t n_used  = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];

    const char * as_data  = (const char *) as->data;
    const char * b_data   = (const char *) b->data;
    const char * ids_data = (const char *) ids->data;
    char *       dst_data = (char *) node->data;

    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t s = 0; s < n_used; ++s) {
            const int32_t e = *(const int32_t *) (ids_data + t * ids->nb[1] + s * ids->nb[0]);
            const float * w_e   = (const float *) (as_data + e * as->nb[2]);
            const float * b_col = (const float *) (b_data + t * b->nb[2] + s * b->nb[1]);
            float *       out   = (float *) (dst_data + t * node->nb[2] + s * node->nb[1]);
            for (int64_t j = 0; j < rows; ++j) {
                const float * w_row = (const float *) ((const char *) w_e + j * as->nb[1]);
                double sum = 0.0;
                for (int64_t i = 0; i < cols; ++i) {
                    sum += (double) w_row[i] * (double) b_col[i];
                }
                out[j] = (float) sum;
            }
        }
    }
}

enum ggml_status fake_dev_graph_compute(ggml_backend_t, struct ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (node->op == GGML_OP_MUL_MAT_ID) {
            compute_mul_mat_id(node);
        } else if (node->op != GGML_OP_NONE) {
            fprintf(stderr, "fake_dev_graph_compute: unexpected op %s\n", ggml_op_name(node->op));
            return GGML_STATUS_FAILED;
        }
    }
    return GGML_STATUS_SUCCESS;
}

// expected value of sum_i w[i,j,e] for the deterministic fill w[i,j,e] = e*100 + j*10 + i,
// with the test's all-ones activation vector (see fill_weights/run_step below)
double expected_row_sum(int64_t cols, int32_t e, int64_t j) {
    double sum = 0.0;
    for (int64_t i = 0; i < cols; ++i) {
        sum += (double) (e * 100 + j * 10 + i);
    }
    return sum;
}

struct step_result {
    enum ggml_status status;
    std::vector<float> out; // [rows * n_used], only valid if status == SUCCESS
};

step_result run_step(ggml_backend_sched_t sched, struct ggml_tensor * w, int64_t cols, int64_t rows,
                      const std::vector<int32_t> & ids_vals) {
    const int64_t n_used = (int64_t) ids_vals.size();

    ggml_init_params params{};
    params.mem_size = 16 * ggml_tensor_overhead() + ggml_graph_overhead();
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cols, n_used, 1);
    ggml_set_input(b);
    struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, 1);
    ggml_set_input(ids);
    struct ggml_tensor * y = ggml_mul_mat_id(ctx, w, b, ids);
    ggml_set_output(y);

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);

    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        ggml_free(ctx);
        return { GGML_STATUS_ALLOC_FAILED, {} };
    }

    std::vector<float> b_data(cols * n_used, 1.0f); // all-ones activations -> output is a row-sum
    ggml_backend_tensor_set(b, b_data.data(), 0, ggml_nbytes(b));
    ggml_backend_tensor_set(ids, ids_vals.data(), 0, ggml_nbytes(ids));

    const enum ggml_status status = ggml_backend_sched_graph_compute(sched, graph);

    step_result result;
    result.status = status;
    if (status == GGML_STATUS_SUCCESS) {
        result.out.resize((size_t) (rows * n_used));
        ggml_backend_tensor_get(y, result.out.data(), 0, ggml_nbytes(y));
    }

    ggml_free(ctx);
    return result;
}

void run(const char * name, void (*f)()) {
    printf("%s ", name);
    fflush(stdout);
    f();
    printf("PASSED\n");
}

// shared fixture: two fake backends (dev=priority 0, host=priority 1/last, CPU-typed as
// required by ggml_backend_sched_new), plus one persistent host-resident 3-D weight tensor
struct fixture {
    fake_role host_role{ true,  nullptr };
    fake_role dev_role { false, nullptr };

    ggml_backend_buffer_type host_buft{};
    ggml_backend_buffer_type dev_buft{};
    ggml_backend_device      host_device{};
    ggml_backend_device      dev_device{};
    ggml_backend             host_backend{};
    ggml_backend             dev_backend{};

    ggml_context * ctx_model = nullptr;
    ggml_tensor *  w = nullptr;
    ggml_backend_buffer_t w_buffer = nullptr;

    static constexpr int64_t cols     = 4;
    static constexpr int64_t rows     = 3;
    static constexpr int64_t n_expert = 6;

    fixture() {
        host_buft.iface.get_name      = fake_buft_name;
        host_buft.iface.alloc_buffer  = fake_buft_alloc_buffer;
        host_buft.iface.get_alignment = fake_buft_get_alignment;
        host_buft.iface.is_host       = fake_buft_is_host;
        host_buft.device              = &host_device;
        host_buft.context             = &host_role;
        host_role.buft                = &host_buft;

        dev_buft.iface = host_buft.iface; // identical vtable, only context differs
        dev_buft.device  = &dev_device;
        dev_buft.context = &dev_role;
        dev_role.buft    = &dev_buft;

        host_device.iface.get_name        = fake_dev_get_name;
        host_device.iface.get_description = fake_dev_get_description;
        host_device.iface.get_memory      = fake_dev_get_memory;
        host_device.iface.get_type        = fake_dev_get_type_host;
        host_device.iface.get_props       = fake_dev_get_props;
        host_device.iface.get_buffer_type = fake_dev_get_buffer_type;
        host_device.iface.supports_op     = fake_dev_supports_op;
        host_device.iface.supports_buft   = fake_dev_supports_buft;
        host_device.context               = &host_role;

        dev_device.iface = host_device.iface;
        dev_device.iface.get_type = fake_dev_get_type_dev;
        dev_device.context        = &dev_role;

        host_backend.iface.get_name      = fake_backend_get_name;
        host_backend.iface.free          = fake_backend_free;
        host_backend.iface.graph_compute = fake_host_graph_compute;
        host_backend.device              = &host_device;
        host_backend.context             = &host_role;

        dev_backend.iface = host_backend.iface;
        dev_backend.iface.graph_compute = fake_dev_graph_compute;
        dev_backend.device              = &dev_device;
        dev_backend.context             = &dev_role;

        ggml_init_params params{};
        params.mem_size = 4 * ggml_tensor_overhead();
        params.no_alloc = true;
        ctx_model = ggml_init(params);

        w = ggml_new_tensor_3d(ctx_model, GGML_TYPE_F32, cols, rows, n_expert);
        ggml_set_name(w, "expert_weight");
        w_buffer = ggml_backend_buft_alloc_buffer(&host_buft, ggml_nbytes(w));
        GGML_ASSERT(ggml_backend_tensor_alloc(w_buffer, w, ggml_backend_buffer_get_base(w_buffer)) == GGML_STATUS_SUCCESS);
        ggml_backend_buffer_set_usage(w_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        // deterministic fill: w[i,j,e] = e*100 + j*10 + i
        std::vector<float> host_w((size_t) (cols * rows * n_expert));
        for (int64_t e = 0; e < n_expert; ++e) {
            for (int64_t j = 0; j < rows; ++j) {
                for (int64_t i = 0; i < cols; ++i) {
                    host_w[(size_t) (e * rows * cols + j * cols + i)] = (float) (e * 100 + j * 10 + i);
                }
            }
        }
        ggml_backend_tensor_set(w, host_w.data(), 0, ggml_nbytes(w));
    }

    ~fixture() {
        ggml_backend_buffer_free(w_buffer);
        ggml_free(ctx_model);
    }

    ggml_backend_sched_t new_sched(int n_slots_min) {
        const size_t expert_size = (size_t) (cols * rows) * sizeof(float);
        ggml_backend_t backends[2] = { &dev_backend, &host_backend };
        ggml_backend_buffer_type_t bufts[2] = { &dev_buft, &host_buft };
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2, 64, /*parallel=*/false, /*op_offload=*/true);

        ggml_backend_sched_expert_cache_params xc_params{};
        xc_params.capacity_bytes = (size_t) n_slots_min * expert_size;
        xc_params.policy         = GGML_SCHED_EXPERT_CACHE_LRU;
        xc_params.n_slots_min    = n_slots_min;
        ggml_backend_sched_set_expert_cache(sched, &xc_params);

        return sched;
    }
};

void test_cache_correctness_and_stats() {
    fixture fx;
    ggml_backend_sched_t sched = fx.new_sched(/*n_slots_min=*/2);
    GGML_ASSERT(ggml_backend_sched_expert_cache_enabled(sched));

    // hand-verified LRU trace (see comment block below) against the real
    // ggml_backend_sched_expert_cache_load implementation:
    //   step1 ids={0,1}: cold, 2 misses, 0 hits, 0 evictions -> slot0=e0, slot1=e1
    //   step2 ids={0,2}: e0 hit, e2 miss evicting e1 (LRU)   -> slot0=e0, slot1=e2
    //   step3 ids={1,2}: e2 hit, e1 miss evicting e0 (LRU)   -> slot0=e1, slot1=e2
    // cumulative: accesses=6, hits=2, misses=4, evictions=2
    const std::vector<std::vector<int32_t>> steps = { {0, 1}, {0, 2}, {1, 2} };

    for (const auto & ids_vals : steps) {
        step_result result = run_step(sched, fx.w, fx.cols, fx.rows, ids_vals);
        GGML_ASSERT(result.status == GGML_STATUS_SUCCESS);
        for (size_t s = 0; s < ids_vals.size(); ++s) {
            for (int64_t j = 0; j < fx.rows; ++j) {
                const double expected = expected_row_sum(fx.cols, ids_vals[s], j);
                const double actual   = result.out[s * (size_t) fx.rows + (size_t) j];
                if (expected != actual) {
                    fprintf(stderr, "mismatch at expert=%d row=%lld: expected=%f actual=%f\n",
                            ids_vals[s], (long long) j, expected, actual);
                    GGML_ABORT("expert cache output mismatch");
                }
            }
        }
    }

    struct ggml_backend_sched_expert_cache_stats stats {};
    ggml_backend_sched_get_expert_cache_stats(sched, &stats);
    GGML_ASSERT(stats.accesses  == 6);
    GGML_ASSERT(stats.hits      == 2);
    GGML_ASSERT(stats.misses    == 4);
    GGML_ASSERT(stats.evictions == 2);
    GGML_ASSERT(stats.n_slots_total == 2);

    ggml_backend_sched_reset_expert_cache_stats(sched);
    struct ggml_backend_sched_expert_cache_stats stats_after_reset {};
    ggml_backend_sched_get_expert_cache_stats(sched, &stats_after_reset);
    GGML_ASSERT(stats_after_reset.accesses == 0);
    GGML_ASSERT(stats_after_reset.hits == 0);
    GGML_ASSERT(stats_after_reset.n_slots_total == 2); // layout-only counters survive a stats reset

    ggml_backend_sched_set_expert_cache(sched, nullptr);
    GGML_ASSERT(!ggml_backend_sched_expert_cache_enabled(sched));

    ggml_backend_sched_free(sched);
}

void test_cache_disabled_is_noop() {
    fixture fx;
    ggml_backend_t backends[2] = { &fx.dev_backend, &fx.host_backend };
    ggml_backend_buffer_type_t bufts[2] = { &fx.dev_buft, &fx.host_buft };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2, 64, false, true);
    GGML_ASSERT(!ggml_backend_sched_expert_cache_enabled(sched));

    struct ggml_backend_sched_expert_cache_stats stats {};
    ggml_backend_sched_get_expert_cache_stats(sched, &stats);
    GGML_ASSERT(stats.accesses == 0 && stats.n_slots_total == 0);

    ggml_backend_sched_free(sched);
}

// requesting more distinct experts in one step than the configured cache has slots for must
// fail loudly (a documented v1 limitation - see docs/IMPLEMENTATION_STATUS.md), not crash or
// silently corrupt the output
void test_oversized_step_fails_cleanly() {
    fixture fx;
    ggml_backend_sched_t sched = fx.new_sched(/*n_slots_min=*/2);

    // warm the cache up first so we know a real failure path is exercised, not just a cold start
    step_result warm = run_step(sched, fx.w, fx.cols, fx.rows, { 0, 1 });
    GGML_ASSERT(warm.status == GGML_STATUS_SUCCESS);

    step_result oversized = run_step(sched, fx.w, fx.cols, fx.rows, { 3, 4, 5 }); // 3 distinct > 2 slots
    GGML_ASSERT(oversized.status == GGML_STATUS_FAILED);
    GGML_ASSERT(oversized.out.empty());

    // the scheduler must remain usable afterwards for a step that fits again
    step_result recovered = run_step(sched, fx.w, fx.cols, fx.rows, { 0, 1 });
    GGML_ASSERT(recovered.status == GGML_STATUS_SUCCESS);
    for (int64_t j = 0; j < fx.rows; ++j) {
        GGML_ASSERT(recovered.out[0 * (size_t) fx.rows + (size_t) j] == (float) expected_row_sum(fx.cols, 0, j));
        GGML_ASSERT(recovered.out[1 * (size_t) fx.rows + (size_t) j] == (float) expected_row_sum(fx.cols, 1, j));
    }

    ggml_backend_sched_free(sched);
}

} // namespace

int main() {
    run("test_cache_correctness_and_stats", test_cache_correctness_and_stats);
    run("test_cache_disabled_is_noop", test_cache_disabled_is_noop);
    run("test_oversized_step_fails_cleanly", test_oversized_step_fails_cleanly);
    return 0;
}
