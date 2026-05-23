#include "mus_associative.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>

// ═══════════════════════════════════════════════════════════════
//  Device helpers
// ═══════════════════════════════════════════════════════════════

__device__ int d_find_idx(ConceptId* ids, int count, ConceptId target) {
    for (int i = 0; i < count; i++) {
        if (ids[i] == target) return i;
    }
    return -1;
}

__device__ int d_find_slot(ConceptId* slots, int count, ConceptId target) {
    for (int i = 0; i < count; i++) {
        if (slots[i] == target) return i;
    }
    return -1;
}

// ═══════════════════════════════════════════════════════════════
//  Add association (one-way, atomically fill first empty slot)
// ═══════════════════════════════════════════════════════════════

__global__ void link_kernel(ConceptId* assoc_slots, int* assoc_count,
    int slots_per_node, ConceptId src_id, ConceptId dst_id,
    ConceptId* node_ids, int node_count)
{
    int src_idx = d_find_idx(node_ids, node_count, src_id);
    if (src_idx < 0) return;

    int curr = atomicAdd(&assoc_count[src_idx], 0);
    if (curr >= slots_per_node) return;

    ConceptId* row = assoc_slots + src_idx * slots_per_node;
    for (int i = 0; i < slots_per_node; i++) {
        if (row[i] == dst_id) return;
    }

    int pos = atomicAdd(&assoc_count[src_idx], 1);
    if (pos < slots_per_node) {
        row[pos] = dst_id;
    }
}

// ═══════════════════════════════════════════════════════════════
//  Activation BFS (level-synchronous)
// ═══════════════════════════════════════════════════════════════

__global__ void reset_activation_kernel(float* act, int* visited, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        act[i] = 0.0f;
        visited[i] = 0;
    }
}

__global__ void bfs_init_kernel(ConceptId* node_ids, int node_count,
    ConceptId seed, int* visited, int* frontier, float* activation)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= node_count) return;
    if (node_ids[i] == seed) {
        visited[i] = 1;
        frontier[i] = 1;
        activation[i] = 1.0f;
    } else {
        frontier[i] = 0;
    }
}

__global__ void bfs_expand_kernel(ConceptId* node_ids, int node_count,
    ConceptId* assoc_slots, int* assoc_count, int slots_per_node,
    int* frontier, int* visited, float* activation,
    int* next_frontier, int depth_parity)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= node_count) return;

    if (frontier[i] == 1) {
        frontier[i] = 0;
        ConceptId* row = assoc_slots + i * slots_per_node;
        int cnt = assoc_count[i];
        if (cnt > slots_per_node) cnt = slots_per_node;

        for (int s = 0; s < cnt; s++) {
            ConceptId nbr_id = row[s];
            int nbr_idx = d_find_idx(node_ids, node_count, nbr_id);
            if (nbr_idx >= 0) {
                if (atomicCAS((int*)(visited + nbr_idx), 0, 1) == 0) {
                    next_frontier[nbr_idx] = 1;
                    activation[nbr_idx] = 1.0f;
                }
            }
        }
    }
}

__global__ void swap_frontier_kernel(int* frontier, int* next_frontier, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        frontier[i] = next_frontier[i];
        next_frontier[i] = 0;
    }
}

// ═══════════════════════════════════════════════════════════════
//  Coherence: count nodes with at least 1 association
// ═══════════════════════════════════════════════════════════════

__global__ void coherence_kernel(int* assoc_count, int node_count, int* out) {
    __shared__ int s[256];
    int tid = threadIdx.x;
    int connected = 0;
    for (int i = tid; i < node_count; i += blockDim.x) {
        if (assoc_count[i] > 0) connected++;
    }
    s[tid] = connected;
    __syncthreads();
    for (int sz = blockDim.x / 2; sz > 0; sz /= 2) {
        if (tid < sz) s[tid] += s[tid + sz];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, s[0]);
}

// ═══════════════════════════════════════════════════════════════
//  Saturation: total filled slots / total available slots
// ═══════════════════════════════════════════════════════════════

__global__ void saturation_kernel(int* assoc_count, int node_count, int slots_per_node, float* out) {
    __shared__ unsigned long long s[256];
    int tid = threadIdx.x;
    unsigned long long total = 0;
    for (int i = tid; i < node_count; i += blockDim.x) {
        total += assoc_count[i];
    }
    s[tid] = total;
    __syncthreads();
    for (int sz = blockDim.x / 2; sz > 0; sz /= 2) {
        if (tid < sz) s[tid] += s[tid + sz];
        __syncthreads();
    }
    if (tid == 0) {
        float denom = (float)node_count * slots_per_node;
        out[0] = (denom > 0.0f) ? (float)s[0] / denom : 0.0f;
    }
}

// ═══════════════════════════════════════════════════════════════
//  Hebbian learning: link all pairs in active set
// ═══════════════════════════════════════════════════════════════

__global__ void hebbian_kernel(ConceptId* assoc_slots, int* assoc_count,
    int slots_per_node, ConceptId* node_ids, int node_count,
    ConceptId* active_set, int active_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = active_len * (active_len - 1) / 2;

    int p = idx;
    if (p >= total_pairs) return;

    // Map linear pair index to (i, j) where 0 <= i < j < active_len
    int i = 0;
    while (p >= (active_len - 1 - i)) {
        p -= (active_len - 1 - i);
        i++;
    }
    int j = i + 1 + p;

    if (i >= active_len || j >= active_len) return;

    ConceptId a = active_set[i];
    ConceptId b = active_set[j];
    if (a == b) return;

    int src_idx = d_find_idx(node_ids, node_count, a);
    if (src_idx < 0) return;

    int curr = atomicAdd(&assoc_count[src_idx], 0);
    if (curr >= slots_per_node) return;

    ConceptId* row = assoc_slots + src_idx * slots_per_node;
    for (int s = 0; s < slots_per_node; s++) {
        if (row[s] == b) return;
    }

    int pos = atomicAdd(&assoc_count[src_idx], 1);
    if (pos < slots_per_node) {
        row[pos] = b;
    }

    // Also add reverse link
    int dst_idx = d_find_idx(node_ids, node_count, b);
    if (dst_idx < 0) return;

    curr = atomicAdd(&assoc_count[dst_idx], 0);
    if (curr >= slots_per_node) return;

    row = assoc_slots + dst_idx * slots_per_node;
    for (int s = 0; s < slots_per_node; s++) {
        if (row[s] == a) return;
    }

    pos = atomicAdd(&assoc_count[dst_idx], 1);
    if (pos < slots_per_node) {
        row[pos] = a;
    }
}

// ═══════════════════════════════════════════════════════════════
//  Active count
// ═══════════════════════════════════════════════════════════════

__global__ void active_count_kernel(float* activation, int n, int* out) {
    __shared__ int s[256];
    int tid = threadIdx.x;
    int cnt = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        if (activation[i] > 0.5f) cnt++;
    }
    s[tid] = cnt;
    __syncthreads();
    for (int sz = blockDim.x / 2; sz > 0; sz /= 2) {
        if (tid < sz) s[tid] += s[tid + sz];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, s[0]);
}

// ═══════════════════════════════════════════════════════════════
//  CPU wrappers
// ═══════════════════════════════════════════════════════════════

static cudaStream_t get_stream(void* s) {
    return (cudaStream_t)(uintptr_t)s;
}

AssociativeGraphGPU* assoc_graph_create(int capacity, int slots_per_node) {
    AssociativeGraphGPU* g = (AssociativeGraphGPU*)calloc(1, sizeof(AssociativeGraphGPU));
    if (!g) return NULL;

    g->capacity = capacity;
    g->slots_per_node = slots_per_node;
    g->node_count = 0;

    size_t ids_bytes = (size_t)capacity * sizeof(ConceptId);
    size_t label_bytes = (size_t)capacity * MAX_LABEL_LEN;
    size_t mod_bytes = (size_t)capacity * sizeof(int);
    size_t slots_bytes = (size_t)capacity * slots_per_node * sizeof(ConceptId);
    size_t count_bytes = (size_t)capacity * sizeof(int);
    size_t act_bytes = (size_t)capacity * sizeof(float);
    size_t flag_bytes = (size_t)capacity * sizeof(int);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    g->stream = (void*)(uintptr_t)stream;

    cudaMalloc(&g->node_ids, ids_bytes);
    cudaMalloc(&g->node_labels, label_bytes);
    cudaMalloc(&g->node_modality, mod_bytes);
    cudaMalloc(&g->assoc_slots, slots_bytes);
    cudaMalloc(&g->assoc_count, count_bytes);
    cudaMalloc(&g->activation, act_bytes);
    cudaMalloc(&g->frontier, flag_bytes);
    cudaMalloc(&g->visited, flag_bytes);

    cudaMemsetAsync(g->assoc_count, 0, count_bytes, stream);
    cudaMemsetAsync(g->activation, 0, act_bytes, stream);
    cudaMemsetAsync(g->frontier, 0, flag_bytes, stream);
    cudaMemsetAsync(g->visited, 0, flag_bytes, stream);

    return g;
}

void assoc_graph_destroy(AssociativeGraphGPU* g) {
    if (!g) return;
    cudaStreamSynchronize(get_stream(g->stream));
    cudaFree(g->node_ids);
    cudaFree(g->node_labels);
    cudaFree(g->node_modality);
    cudaFree(g->assoc_slots);
    cudaFree(g->assoc_count);
    cudaFree(g->activation);
    cudaFree(g->frontier);
    cudaFree(g->visited);
    cudaStreamDestroy(get_stream(g->stream));
    free(g);
}

int assoc_graph_add_node(AssociativeGraphGPU* g, ConceptId id, const char* label, int modality) {
    if (!g) return -1;
    cudaStream_t s = get_stream(g->stream);
    int idx = g->node_count;
    g->node_count = idx + 1;
    if (idx >= g->capacity) {
        g->node_count = idx;
        return -1;
    }
    cudaMemcpyAsync(g->node_ids + idx, &id, sizeof(ConceptId), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(g->node_labels + idx * MAX_LABEL_LEN, label, MAX_LABEL_LEN, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(g->node_modality + idx, &modality, sizeof(int), cudaMemcpyHostToDevice, s);
    return idx;
}

int assoc_graph_link(AssociativeGraphGPU* g, ConceptId a, ConceptId b) {
    if (!g || a == b) return 0;
    cudaStream_t str = get_stream(g->stream);

    int* d_src_idx;
    cudaMalloc(&d_src_idx, sizeof(int));
    cudaMemsetAsync(d_src_idx, -1, sizeof(int), str);

    int threads = 256;
    int blocks = (g->node_count + threads - 1) / threads;
    link_kernel<<<1, 1, 0, str>>>(g->assoc_slots, g->assoc_count,
        g->slots_per_node, a, b, g->node_ids, g->node_count);
    cudaFree(d_src_idx);
    return 0;
}

int assoc_graph_activate(AssociativeGraphGPU* g, ConceptId seed, int depth) {
    if (!g || g->node_count == 0) return 0;
    cudaStream_t str = get_stream(g->stream);
    int threads = 256;
    int blocks = (g->capacity + threads - 1) / threads;

    reset_activation_kernel<<<blocks, threads, 0, str>>>(g->activation, g->visited, g->capacity);

    bfs_init_kernel<<<blocks, threads, 0, str>>>(g->node_ids, g->node_count,
        seed, g->visited, g->frontier, g->activation);

    for (int d = 0; d < depth; d++) {
        bfs_expand_kernel<<<blocks, threads, 0, str>>>(
            g->node_ids, g->node_count,
            g->assoc_slots, g->assoc_count, g->slots_per_node,
            g->frontier, g->visited, g->activation,
            g->visited, 0
        );
        swap_frontier_kernel<<<blocks, threads, 0, str>>>(g->frontier, g->visited, g->capacity);
    }

    // Count activated nodes
    int* d_cnt;
    cudaMalloc(&d_cnt, sizeof(int));
    cudaMemsetAsync(d_cnt, 0, sizeof(int), str);
    active_count_kernel<<<blocks, threads, 0, str>>>(g->activation, g->node_count, d_cnt);
    int h_cnt;
    cudaMemcpyAsync(&h_cnt, d_cnt, sizeof(int), cudaMemcpyDeviceToHost, str);
    cudaStreamSynchronize(str);
    cudaFree(d_cnt);
    return h_cnt;
}

float assoc_graph_coherence(AssociativeGraphGPU* g) {
    if (!g || g->node_count < 2) return 1.0f;
    cudaStream_t str = get_stream(g->stream);
    int* d_cnt;
    cudaMalloc(&d_cnt, sizeof(int));
    cudaMemsetAsync(d_cnt, 0, sizeof(int), str);
    int threads = 256;
    int blocks = (g->capacity + threads - 1) / threads;
    coherence_kernel<<<blocks, threads, 0, str>>>(g->assoc_count, g->node_count, d_cnt);
    int h_cnt;
    cudaMemcpyAsync(&h_cnt, d_cnt, sizeof(int), cudaMemcpyDeviceToHost, str);
    cudaStreamSynchronize(str);
    cudaFree(d_cnt);
    return (float)h_cnt / (float)g->node_count;
}

float assoc_graph_saturation(AssociativeGraphGPU* g) {
    if (!g || g->node_count == 0) return 0.0f;
    cudaStream_t str = get_stream(g->stream);
    float* d_sat;
    cudaMalloc(&d_sat, sizeof(float));
    int threads = 256;
    int blocks = (g->capacity + threads - 1) / threads;
    saturation_kernel<<<blocks, threads, 0, str>>>(g->assoc_count, g->node_count, g->slots_per_node, d_sat);
    float h_sat;
    cudaMemcpyAsync(&h_sat, d_sat, sizeof(float), cudaMemcpyDeviceToHost, str);
    cudaStreamSynchronize(str);
    cudaFree(d_sat);
    return h_sat;
}

int assoc_graph_hebbian_learn(AssociativeGraphGPU* g, const ConceptId* active_set, int active_len) {
    if (!g || active_len < 2) return 0;
    cudaStream_t str = get_stream(g->stream);

    ConceptId* d_active;
    cudaMalloc(&d_active, active_len * sizeof(ConceptId));
    cudaMemcpyAsync(d_active, active_set, active_len * sizeof(ConceptId), cudaMemcpyHostToDevice, str);

    int total_pairs = active_len * (active_len - 1) / 2;
    int threads = 256;
    int blocks = (total_pairs + threads - 1) / threads;

    hebbian_kernel<<<blocks, threads, 0, str>>>(
        g->assoc_slots, g->assoc_count, g->slots_per_node,
        g->node_ids, g->node_count, d_active, active_len);

    cudaStreamSynchronize(str);
    cudaFree(d_active);
    return total_pairs;
}

int assoc_graph_evict_oldest(AssociativeGraphGPU* g, float ratio) {
    if (!g || g->node_count == 0) return 0;
    cudaStream_t str = get_stream(g->stream);
    int to_evict = (int)(g->node_count * ratio);
    if (to_evict < 1) to_evict = 1;
    if (to_evict > g->node_count) to_evict = g->node_count;

    // Copy node IDs to host, evict first to_evict
    ConceptId* h_ids = (ConceptId*)malloc(g->node_count * sizeof(ConceptId));
    cudaMemcpyAsync(h_ids, g->node_ids, g->node_count * sizeof(ConceptId), cudaMemcpyDeviceToHost, str);
    cudaStreamSynchronize(str);

    // On CPU: slide remaining nodes down
    int remaining = g->node_count - to_evict;
    if (remaining > 0) {
        cudaMemcpyAsync(g->node_ids, g->node_ids + to_evict, remaining * sizeof(ConceptId), cudaMemcpyDeviceToDevice, str);
        cudaMemcpyAsync(g->node_labels, g->node_labels + to_evict * MAX_LABEL_LEN, remaining * MAX_LABEL_LEN, cudaMemcpyDeviceToDevice, str);
        cudaMemcpyAsync(g->node_modality, g->node_modality + to_evict, remaining * sizeof(int), cudaMemcpyDeviceToDevice, str);
        cudaMemcpyAsync(g->assoc_slots, g->assoc_slots + to_evict * g->slots_per_node, remaining * g->slots_per_node * sizeof(ConceptId), cudaMemcpyDeviceToDevice, str);
        cudaMemcpyAsync(g->assoc_count, g->assoc_count + to_evict, remaining * sizeof(int), cudaMemcpyDeviceToDevice, str);
        cudaMemcpyAsync(g->activation, g->activation + to_evict, remaining * sizeof(float), cudaMemcpyDeviceToDevice, str);

        // Clear stale references to evicted nodes
        int* d_removed_ids;
        cudaMalloc(&d_removed_ids, to_evict * sizeof(int));

        // For each remaining node, remove associations pointing to evicted nodes
        ConceptId* h_remaining = (ConceptId*)malloc(remaining * sizeof(ConceptId));
        cudaMemcpyAsync(h_remaining, g->node_ids, remaining * sizeof(ConceptId), cudaMemcpyDeviceToHost, str);
        cudaStreamSynchronize(str);

        for (int i = 0; i < remaining; i++) {
            ConceptId* h_row = (ConceptId*)malloc(g->slots_per_node * sizeof(ConceptId));
            int* h_cnt = (int*)malloc(sizeof(int));
            cudaMemcpyAsync(h_cnt, g->assoc_count + i, sizeof(int), cudaMemcpyDeviceToHost, str);
            cudaMemcpyAsync(h_row, g->assoc_slots + i * g->slots_per_node, g->slots_per_node * sizeof(ConceptId), cudaMemcpyDeviceToHost, str);
            cudaStreamSynchronize(str);

            // Check if any evicted IDs remain
            int cnt = *h_cnt;
            if (cnt > g->slots_per_node) cnt = g->slots_per_node;
            int new_cnt = 0;
            for (int s = 0; s < cnt; s++) {
                int is_evicted = 0;
                for (int e = 0; e < to_evict; e++) {
                    if (h_row[s] == h_ids[h_ids[e] == h_row[s] ? e : -1]) {
                        // check properly
                    }
                    if (h_row[s] == h_ids[e]) {
                        is_evicted = 1;
                        break;
                    }
                }
                if (!is_evicted) {
                    h_row[new_cnt++] = h_row[s];
                }
            }
            cudaMemcpyAsync(g->assoc_slots + i * g->slots_per_node, h_row, g->slots_per_node * sizeof(ConceptId), cudaMemcpyHostToDevice, str);
            *h_cnt = new_cnt;
            cudaMemcpyAsync(g->assoc_count + i, h_cnt, sizeof(int), cudaMemcpyHostToDevice, str);
            free(h_row);
            free(h_cnt);
        }
        free(h_remaining);
        cudaFree(d_removed_ids);
    }

    g->node_count = remaining;
    free(h_ids);
    return to_evict;
}

void assoc_graph_reset_activations(AssociativeGraphGPU* g) {
    if (!g) return;
    cudaStream_t str = get_stream(g->stream);
    int threads = 256;
    int blocks = (g->capacity + threads - 1) / threads;
    cudaMemsetAsync(g->activation, 0, g->capacity * sizeof(float), str);
    cudaMemsetAsync(g->frontier, 0, g->capacity * sizeof(int), str);
    cudaMemsetAsync(g->visited, 0, g->capacity * sizeof(int), str);
}

int assoc_graph_active_count(AssociativeGraphGPU* g) {
    if (!g) return 0;
    cudaStream_t str = get_stream(g->stream);
    int* d_cnt;
    cudaMalloc(&d_cnt, sizeof(int));
    cudaMemsetAsync(d_cnt, 0, sizeof(int), str);
    int threads = 256;
    int blocks = (g->capacity + threads - 1) / threads;
    active_count_kernel<<<blocks, threads, 0, str>>>(g->activation, g->node_count, d_cnt);
    int h_cnt;
    cudaMemcpyAsync(&h_cnt, d_cnt, sizeof(int), cudaMemcpyDeviceToHost, str);
    cudaStreamSynchronize(str);
    cudaFree(d_cnt);
    return h_cnt;
}

int assoc_graph_node_count(AssociativeGraphGPU* g) {
    return g ? g->node_count : 0;
}

int assoc_graph_capacity(AssociativeGraphGPU* g) {
    return g ? g->capacity : 0;
}

int assoc_graph_slots(AssociativeGraphGPU* g) {
    return g ? g->slots_per_node : 0;
}

ConceptId assoc_graph_get_node_id(AssociativeGraphGPU* g, int idx) {
    if (!g || idx < 0 || idx >= g->node_count) return 0;
    cudaStream_t str = get_stream(g->stream);
    ConceptId h_id;
    cudaMemcpyAsync(&h_id, g->node_ids + idx, sizeof(ConceptId), cudaMemcpyDeviceToHost, str);
    cudaStreamSynchronize(str);
    return h_id;
}

int assoc_graph_get_node_ids(AssociativeGraphGPU* g, ConceptId* dst, int max_len) {
    if (!g) return 0;
    int n = g->node_count;
    if (n > max_len) n = max_len;
    cudaStream_t str = get_stream(g->stream);
    cudaMemcpyAsync(dst, g->node_ids, n * sizeof(ConceptId), cudaMemcpyDeviceToHost, str);
    cudaStreamSynchronize(str);
    return n;
}

int assoc_graph_get_activations(AssociativeGraphGPU* g, float* dst, int max_len) {
    if (!g) return 0;
    int n = g->node_count;
    if (n > max_len) n = max_len;
    cudaStream_t str = get_stream(g->stream);
    cudaMemcpyAsync(dst, g->activation, n * sizeof(float), cudaMemcpyDeviceToHost, str);
    cudaStreamSynchronize(str);
    return n;
}
