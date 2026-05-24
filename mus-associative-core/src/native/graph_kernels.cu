#include "../include/mus_associative.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define BLOCK_SIZE 256
#define ENERGY_DECAY 0.85f

// ─── node_id → index lookup hash ─────────────────────────────
// Simple hash table in shared memory per block
__device__ int node_to_idx(ConceptId id, ConceptId* ids, int count) {
    for (int i = 0; i < count; i++) {
        if (ids[i] == id) return i;
    }
    return -1;
}

// ─── BFS expansion (CSR version) ─────────────────────────────
__global__ void bfs_expand_kernel_csr(
    ConceptId* node_ids,
    int* row_ptr,
    int* col_indices,
    float* values,
    float* short_values,
    float* activation,
    ConceptId* frontier,
    int* visited,
    int capacity,
    int frontier_size,
    int max_depth,
    int use_chemistry,
    float dopamin,
    float adrenaline,
    float energy,
    ConceptId* next_frontier,
    int* next_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frontier_size) return;

    ConceptId current = frontier[idx];
    int current_idx = node_to_idx(current, node_ids, capacity);
    if (current_idx < 0) return;

    float current_energy = activation[current_idx] > 0.0f ? activation[current_idx] : energy;
    uint32_t rng = (uint32_t)(current * 2654435761U) ^ (uint32_t)(blockIdx.x * 104729) ^ (uint32_t)(threadIdx.x * 7919);

    int start = row_ptr[current_idx];
    int end = row_ptr[current_idx + 1];

    for (int i = start; i < end; i++) {
        int nidx = col_indices[i];
        float weight = values[i];
        if (short_values) weight += short_values[i];
        if (weight > 1.0f) weight = 1.0f;
        if (nidx < 0 || nidx >= capacity || weight < 0.01f) continue;
        ConceptId neighbor = node_ids[nidx];
        if (neighbor == 0) continue;
        if (visited[nidx]) continue;

        if (use_chemistry) {
            float threshold = 0.1f - dopamin * 0.15f;
            if (threshold < 0.01f) threshold = 0.01f;
            if (weight < threshold) continue;
        }

        visited[nidx] = 1;
        float energy_factor = 0.8f + weight * 0.2f;
        activation[nidx] = current_energy * (ENERGY_DECAY * energy_factor);

        if (next_frontier && next_count) {
            int pos = atomicAdd(next_count, 1);
            if (pos < capacity) {
                next_frontier[pos] = neighbor;
            }
        }
        
        // STDP: Boost short-term weight for this edge
        if (short_values) {
            atomicAdd(&short_values[i], 0.1f * weight);
        }
    }
}

// ─── BFS init ──────────────────────────────────────────────────
__global__ void bfs_init_kernel_csr(
    ConceptId* node_ids,
    float* activation,
    int* visited,
    ConceptId* frontier,
    int capacity,
    ConceptId seed
) {
    for (int i = threadIdx.x; i < capacity; i += blockDim.x) {
        visited[i] = 0;
        if (node_ids[i] == seed) {
            visited[i] = 1;
            activation[i] = 1.0f;
        } else {
            activation[i] = 0.0f;
        }
    }
    if (threadIdx.x == 0) frontier[0] = seed;
}

// ─── Coherence ──────────────────────────────────────────────────
__global__ void coherence_kernel_csr(
    float* activation,
    int capacity,
    float* result
) {
    __shared__ float s[256];
    if (blockIdx.x > 0) return;

    int count = 0;
    for (int i = threadIdx.x; i < capacity; i += blockDim.x) {
        if (activation[i] > 0.0f) count++;
    }
    s[threadIdx.x] = (float)count;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) *result = s[0] / (float)(capacity > 0 ? capacity : 1);
}

// ─── Saturation (CSR version: fraction of slots filled) ───────
__global__ void saturation_kernel_csr(
    int* row_ptr,
    int capacity,
    int max_edges,
    float* result
) {
    __shared__ float s[256];
    if (blockIdx.x > 0) return;

    int total = 0;
    for (int i = threadIdx.x; i < capacity; i += blockDim.x) {
        total += row_ptr[i + 1] - row_ptr[i];
    }
    s[threadIdx.x] = (float)total;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) *result = s[0] / (float)(max_edges > 0 ? max_edges : 1);
}

// ─── Batch link (CSR rebuild kernel) ──────────────────────────
// Builds row_ptr, col_indices, values from sorted edge list
// edges: interleaved [from, to, from, to, ...]  (node IDs)
// work_deg: [capacity] temporary for degree counting
__global__ void batch_link_csr_kernel(
    ConceptId* node_ids,
    int* row_ptr,
    int* col_indices,
    float* values,
    const ConceptId* edges,
    int num_edges,
    int* degree,
    int capacity,
    float dopamin,
    float adrenaline,
    int use_chemistry
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = blockDim.x * gridDim.x;

    for (int i = tid; i < capacity; i += total) {
        degree[i] = 0;
    }
    __syncthreads();

    // Count degrees — both directions (undirected graph)
    for (int i = tid; i < num_edges; i += total) {
        int base = i * 2;
        ConceptId from = edges[base];
        ConceptId to = edges[base + 1];
        for (int j = 0; j < capacity; j++) {
            if (node_ids[j] == from) { atomicAdd(&degree[j], 1); break; }
        }
        for (int j = 0; j < capacity; j++) {
            if (node_ids[j] == to) { atomicAdd(&degree[j], 1); break; }
        }
    }
    __syncthreads();

    // Build row_ptr (exclusive prefix sum)
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < capacity; i++) {
            row_ptr[i] = sum;
            sum += degree[i];
        }
        row_ptr[capacity] = sum;
        for (int i = 0; i < capacity; i++) {
            degree[i] = 0;
        }
    }
    __syncthreads();

}

// ─── CSR fill kernel (single block) ────────────────────────────
__global__ void fill_csr_kernel(
    ConceptId* node_ids,
    int* row_ptr,
    int* col_indices,
    float* values,
    const ConceptId* edges,
    int num_edges,
    int* degree,
    int capacity,
    float dopamin,
    float adrenaline,
    int use_chemistry
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = blockDim.x * gridDim.x;

    for (int p = tid; p < num_edges; p += total) {
        int base = p * 2;
        ConceptId from = edges[base];
        ConceptId to = edges[base + 1];
        int from_idx = -1, to_idx = -1;
        for (int j = 0; j < capacity; j++) {
            if (node_ids[j] == from) { from_idx = j; break; }
        }
        for (int j = 0; j < capacity; j++) {
            if (node_ids[j] == to) { to_idx = j; break; }
        }
        if (from_idx >= 0 && to_idx >= 0) {
            float w = ASSOC_WEIGHT_INIT;
            if (use_chemistry) {
                w = ASSOC_WEIGHT_INIT + dopamin * 0.5f;
                if (adrenaline > 0.0f) w -= adrenaline * 0.03f * w;
                if (w < 0.01f) w = 0.01f;
                if (w > ASSOC_WEIGHT_MAX) w = ASSOC_WEIGHT_MAX;
            }
            int pos_f = atomicAdd(&degree[from_idx], 1);
            int insert_f = row_ptr[from_idx] + pos_f;
            col_indices[insert_f] = to_idx;
            values[insert_f] = w;
            int pos_t = atomicAdd(&degree[to_idx], 1);
            int insert_t = row_ptr[to_idx] + pos_t;
            col_indices[insert_t] = from_idx;
            values[insert_t] = w;
        }
    }
}

// ─── STDP Hebbian learn (CSR version) ─────────────────────────
__global__ void hebbian_learn_csr_kernel(
    ConceptId* node_ids,
    int* row_ptr,
    int* col_indices,
    float* values,
    const ConceptId* active_set,
    int active_len,
    int capacity,
    float dopamin,
    float adrenaline
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= active_len * active_len) return;

    int i = tid / active_len;
    int j = tid % active_len;
    if (i == j) return;

    ConceptId a = active_set[i];
    ConceptId b = active_set[j];
    int ai = node_to_idx(a, node_ids, capacity);
    int bi = node_to_idx(b, node_ids, capacity);
    if (ai < 0 || bi < 0) return;

    int start = row_ptr[ai];
    int end = row_ptr[ai + 1];
    for (int k = start; k < end; k++) {
        if (col_indices[k] == bi) {
            // LTP: boost * (1.0 + dopamin * 0.5)
            float boost = ASSOC_WEIGHT_BOOST;
            float ltp = boost * (1.0f + dopamin * 0.5f);
            values[k] += ltp;
            if (values[k] > ASSOC_WEIGHT_MAX) values[k] = ASSOC_WEIGHT_MAX;

            // LTD: adrenaline * 0.05 * current weight
            float ltd = adrenaline * 0.05f * values[k];
            values[k] -= ltd;
            if (values[k] < 0.01f) values[k] = 0.01f;
            break;
        }
    }
}

// ─── Decay weights ────────────────────────────────────────────
__global__ void decay_weights_csr_kernel(
    float* values,
    int num_edges,
    float decay_rate
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_edges) {
        values[i] *= decay_rate;
        if (values[i] < 0.01f) values[i] = 0.0f;
    }
}

// ─── Short-term plasticity: fast decay ─────────────────────────
__global__ void short_decay_kernel(
    float* short_values,
    int num_edges,
    float decay_rate
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_edges) {
        short_values[i] *= decay_rate;
        if (short_values[i] < 0.001f) short_values[i] = 0.0f;
    }
}

// ─── Short-term plasticity: boost edges traversed by BFS ───────
__global__ void short_boost_edges_kernel(
    int* row_ptr,
    int* col_indices,
    float* short_values,
    ConceptId* frontier,
    int frontier_size,
    ConceptId* node_ids,
    int capacity,
    float boost
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frontier_size) return;

    ConceptId current = frontier[idx];
    int current_idx = -1;
    for (int i = 0; i < capacity; i++) {
        if (node_ids[i] == current) { current_idx = i; break; }
    }
    if (current_idx < 0) return;

    int start = row_ptr[current_idx];
    int end = row_ptr[current_idx + 1];

    for (int i = start; i < end; i++) {
        float existing = short_values[i];
        short_values[i] += boost;
        if (short_values[i] > 1.0f) short_values[i] = 1.0f;
    }
}

// ─── Prune weights ────────────────────────────────────────────
__global__ void prune_weights_csr_kernel(
    int* row_ptr,
    int* col_indices,
    float* values,
    int* degree,
    int capacity,
    float threshold
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity) return;

    int start = row_ptr[i];
    int end = row_ptr[i + 1];
    int write = start;

    // Mark edges below threshold for removal (set weight to 0)
    for (int k = start; k < end; k++) {
        if (values[k] >= threshold) {
            if (write != k) {
                col_indices[write] = col_indices[k];
                values[write] = values[k];
            }
            write++;
        }
    }
    degree[i] = write - start;
}

// ─── Rebuild row_ptr from degree after prune ─────────────────
__global__ void rebuild_row_ptr_kernel(
    int* row_ptr,
    int* degree,
    int capacity
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        row_ptr[0] = 0;
        for (int i = 0; i < capacity; i++) {
            row_ptr[i + 1] = row_ptr[i] + degree[i];
        }
    }
}

// ─── Evict oldest (reset node) ────────────────────────────────
__global__ void evict_oldest_csr_kernel(
    ConceptId* node_ids,
    int* row_ptr,
    int capacity,
    int* num_edges,
    char* node_labels
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity) return;
    if (node_ids[i] == 0) return;

    // Simple LRU: clear nodes with degree 0
    int deg = row_ptr[i + 1] - row_ptr[i];
    if (deg == 0) {
        node_ids[i] = 0;
        if (node_labels) node_labels[i * MAX_LABEL_LEN] = 0;
    }
}

// ─── Panic clear ──────────────────────────────────────────────
__global__ void panic_clear_csr_kernel(
    int* row_ptr,
    float* values,
    int* col_indices,
    int capacity,
    ConceptId* node_ids,
    float adrenaline,
    float threshold
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity) return;

    int start = row_ptr[i];
    int end = row_ptr[i + 1];
    for (int k = start; k < end; k++) {
        if (values[k] < threshold) {
            values[k] *= (1.0f - adrenaline * 0.3f);
            if (values[k] < 0.01f) values[k] = 0.0f;
        }
    }
}

// ─── K-WTA filter: keep only top-k activations ──────────────
__global__ void top_k_filter_kernel(float* activation, int capacity, int k) {
    __shared__ float s_vals[256];
    __shared__ int s_idx[256];
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;
    int total = blockDim.x * gridDim.x;

    // Each thread finds local max among its elements
    float best_val = -1e10f;
    int best_idx = -1;
    for (int i = global_tid; i < capacity; i += total) {
        if (activation[i] > best_val) {
            best_val = activation[i];
            best_idx = i;
        }
    }
    s_vals[tid] = best_val;
    s_idx[tid] = best_idx;
    __syncthreads();

    // Grid-wide top-k via repeated block-level max
    for (int round = 0; round < k && round < capacity; round++) {
        // Find max in shared
        int max_lane = 0;
        for (int i = 1; i < blockDim.x; i++) {
            if (s_vals[i] > s_vals[max_lane]) max_lane = i;
        }
        float max_val = s_vals[max_lane];
        int max_i = s_idx[max_lane];
        __syncthreads();

        // Suppress the max so next round finds the next-best
        if (max_i >= 0) {
            // Zero out all activations below the k-th threshold
            if (round == k - 1) {
                for (int i = tid; i < capacity; i += total) {
                    if (activation[i] < max_val) activation[i] = 0.0f;
                }
            }
            // Block-level: remove this winner from shared
            if (max_lane == tid) { s_vals[tid] = -1e10f; s_idx[tid] = -1; }
        }
        __syncthreads();
    }
}

// ─── C API ─────────────────────────────────────────────────────
extern "C" {

AssociativeGraphGPU* assoc_graph_create(int capacity, int max_edges_per_node) {
    AssociativeGraphGPU* g = (AssociativeGraphGPU*)calloc(1, sizeof(AssociativeGraphGPU));
    if (!g) return NULL;

    int max_edges = capacity * max_edges_per_node;

    cudaMalloc(&g->node_ids, capacity * sizeof(ConceptId));
    cudaMalloc(&g->node_labels, capacity * MAX_LABEL_LEN);
    cudaMalloc(&g->node_modality, capacity * sizeof(int));
    cudaMalloc(&g->row_ptr, (capacity + 1) * sizeof(int));
    cudaMalloc(&g->col_indices, max_edges * sizeof(int));
    cudaMalloc(&g->values, max_edges * sizeof(float));
    cudaMalloc(&g->short_values, max_edges * sizeof(float));
    cudaMalloc(&g->activation, capacity * sizeof(float));
    cudaMalloc(&g->frontier, capacity * sizeof(ConceptId));
    cudaMalloc(&g->next_frontier, capacity * sizeof(ConceptId));
    cudaMalloc(&g->frontier_next_count, sizeof(int));
    cudaMalloc(&g->visited, capacity * sizeof(int));
    cudaMalloc(&g->degree, capacity * sizeof(int));

    g->capacity = capacity;
    g->node_count = 0;
    g->num_edges = 0;
    g->max_edges = max_edges;
    g->use_chemistry = 0;
    g->stream = NULL;

    // Init row_ptr to zero
    cudaMemset(g->row_ptr, 0, (capacity + 1) * sizeof(int));
    cudaMemset(g->node_ids, 0, capacity * sizeof(ConceptId));
    cudaMemset(g->short_values, 0, max_edges * sizeof(float));

    return g;
}

void assoc_graph_destroy(AssociativeGraphGPU* g) {
    if (!g) return;
    cudaFree(g->node_ids);
    cudaFree(g->node_labels);
    cudaFree(g->node_modality);
    cudaFree(g->row_ptr);
    cudaFree(g->col_indices);
    cudaFree(g->values);
    cudaFree(g->activation);
    cudaFree(g->frontier);
    cudaFree(g->next_frontier);
    cudaFree(g->frontier_next_count);
    cudaFree(g->visited);
    cudaFree(g->degree);
    cudaFree(g->short_values);
    free(g);
}

int assoc_graph_add_node(AssociativeGraphGPU* g, ConceptId id, const char* label, int modality) {
    if (!g || g->node_count >= g->capacity) return -1;
    int idx = g->node_count;
    cudaMemcpy(&g->node_ids[idx], &id, sizeof(ConceptId), cudaMemcpyHostToDevice);
    if (label) {
        char buf[MAX_LABEL_LEN] = {0};
        strncpy(buf, label, MAX_LABEL_LEN - 1);
        cudaMemcpy(&g->node_labels[idx * MAX_LABEL_LEN], buf, MAX_LABEL_LEN, cudaMemcpyHostToDevice);
    }
    cudaMemcpy(&g->node_modality[idx], &modality, sizeof(int), cudaMemcpyHostToDevice);
    g->node_count++;
    return idx;
}

int assoc_graph_link(AssociativeGraphGPU* g, ConceptId a, ConceptId b) {
    // CSR is batch-rebuilt; single-edge link is a no-op.
    // Use assoc_graph_batch_link for additions.
    return 0;
}

int assoc_graph_has_edge(AssociativeGraphGPU* g, ConceptId a, ConceptId b) {
    // Find indices
    int ai = -1, bi = -1;
    for (int i = 0; i < g->node_count; i++) {
        ConceptId id;
        cudaMemcpy(&id, &g->node_ids[i], sizeof(ConceptId), cudaMemcpyDeviceToHost);
        if (id == a) ai = i;
        if (id == b) bi = i;
    }
    if (ai < 0 || bi < 0) return 0;

    // Check CSR
    int start, end;
    cudaMemcpy(&start, &g->row_ptr[ai], sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&end, &g->row_ptr[ai + 1], sizeof(int), cudaMemcpyDeviceToHost);

    for (int k = start; k < end; k++) {
        int nidx;
        cudaMemcpy(&nidx, &g->col_indices[k], sizeof(int), cudaMemcpyDeviceToHost);
        if (nidx == bi) return 1;
    }
    return 0;
}

int assoc_graph_activate(AssociativeGraphGPU* g, ConceptId seed, int depth) {
    CudaSystemState state = {0};
    state.energy = 1.0f;
    return assoc_graph_activate_chem(g, seed, depth, &state);
}

int assoc_graph_activate_chem(AssociativeGraphGPU* g, ConceptId seed, int max_depth, const CudaSystemState* state) {
    if (!g) return -1;

    int capacity = g->capacity;
    int* next_count = g->frontier_next_count;

    // Init BFS
    bfs_init_kernel_csr<<<1, BLOCK_SIZE>>>(
        g->node_ids, g->activation, g->visited, g->frontier,
        capacity, seed
    );

    int frontier_size = 1;
    ConceptId* cur_frontier = g->frontier;
    ConceptId* nxt_frontier = g->next_frontier;

    for (int d = 0; d < max_depth && frontier_size > 0; d++) {
        cudaMemset(next_count, 0, sizeof(int));

        int blocks = (frontier_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (blocks > 256) blocks = 256;

        bfs_expand_kernel_csr<<<blocks, BLOCK_SIZE>>>(
            g->node_ids, g->row_ptr, g->col_indices, g->values, g->short_values,
            g->activation,
            cur_frontier, g->visited, capacity,
            frontier_size, max_depth,
            state ? state->use_chemistry : 0,
            state ? state->dopamin : 0,
            state ? state->adrenaline : 0,
            state ? state->energy : 1.0f,
            nxt_frontier, next_count
        );

        cudaMemcpy(&frontier_size, next_count, sizeof(int), cudaMemcpyDeviceToHost);
        if (frontier_size > capacity) frontier_size = capacity;

        // Swap frontiers
        ConceptId* tmp = cur_frontier;
        cur_frontier = nxt_frontier;
        nxt_frontier = tmp;
    }
    return 0;
}

float assoc_graph_coherence(AssociativeGraphGPU* g) {
    if (!g) return 0.0f;
    float* d_result;
    cudaMalloc(&d_result, sizeof(float));
    coherence_kernel_csr<<<1, BLOCK_SIZE>>>(g->activation, g->capacity, d_result);
    float result;
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    return result;
}

float assoc_graph_saturation(AssociativeGraphGPU* g) {
    if (!g) return 0.0f;
    float* d_result;
    cudaMalloc(&d_result, sizeof(float));
    saturation_kernel_csr<<<1, BLOCK_SIZE>>>(g->row_ptr, g->capacity, g->max_edges, d_result);
    float result;
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    return result;
}

int assoc_graph_hebbian_learn(AssociativeGraphGPU* g, const ConceptId* active_set, int active_len, float dopamin, float adrenaline) {
    if (!g || !active_set || active_len < 2) return -1;

    ConceptId* d_active;
    cudaMalloc(&d_active, active_len * sizeof(ConceptId));
    cudaMemcpy(d_active, active_set, active_len * sizeof(ConceptId), cudaMemcpyHostToDevice);

    int total_pairs = active_len * active_len;
    int blocks = (total_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks > 256) blocks = 256;

    hebbian_learn_csr_kernel<<<blocks, BLOCK_SIZE>>>(
        g->node_ids, g->row_ptr, g->col_indices, g->values,
        d_active, active_len, g->capacity,
        dopamin, adrenaline
    );

    cudaFree(d_active);
    return 0;
}

int assoc_graph_batch_link(AssociativeGraphGPU* g, const ConceptId* pairs, int num_pairs, float dopamin, float adrenaline) {
    if (!g || !pairs || num_pairs < 1) return -1;

    // Debug: verify pairs on host
    printf("[BLINK] num_pairs=%d capacity=%d node_count=%d\n", num_pairs, g->capacity, g->node_count);
    ConceptId* host_ids = (ConceptId*)malloc(g->capacity * sizeof(ConceptId));
    cudaMemcpy(host_ids, g->node_ids, g->capacity * sizeof(ConceptId), cudaMemcpyDeviceToHost);
    int found = 0, miss = 0;
    for (int i = 0; i < num_pairs * 2; i++) {
        int ok = 0;
        for (int j = 0; j < g->capacity; j++) {
            if (host_ids[j] == pairs[i]) { ok = 1; break; }
        }
        if (ok) found++; else miss++;
    }
    free(host_ids);

    ConceptId* d_pairs;
    cudaMalloc(&d_pairs, num_pairs * 2 * sizeof(ConceptId));
    cudaMemcpy(d_pairs, pairs, num_pairs * 2 * sizeof(ConceptId), cudaMemcpyHostToDevice);

    // Build CSR from pairs
    int blocks = (g->capacity + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks > 256) blocks = 256;

    batch_link_csr_kernel<<<blocks, BLOCK_SIZE>>>(
        g->node_ids, g->row_ptr, g->col_indices, g->values,
        d_pairs, num_pairs, g->degree, g->capacity,
        dopamin, adrenaline, g->use_chemistry
    );
    cudaError_t cerr = cudaDeviceSynchronize();
    if (cerr != cudaSuccess) printf("[BLINK] COUNT ERROR: %s\n", cudaGetErrorString(cerr));

    // Fill phase — separate kernel ensures global-memory coherence of row_ptr
    fill_csr_kernel<<<1, BLOCK_SIZE>>>(
        g->node_ids, g->row_ptr, g->col_indices, g->values,
        d_pairs, num_pairs, g->degree, g->capacity,
        dopamin, adrenaline, g->use_chemistry
    );
    cerr = cudaDeviceSynchronize();
    if (cerr != cudaSuccess) printf("[BLINK] FILL ERROR: %s\n", cudaGetErrorString(cerr));

    // Rebuild row_ptr from degree counters
    rebuild_row_ptr_kernel<<<1, BLOCK_SIZE>>>(
        g->row_ptr, g->degree, g->capacity
    );
    cerr = cudaDeviceSynchronize();
    if (cerr != cudaSuccess) printf("[BLINK] REBUILD ERROR: %s\n", cudaGetErrorString(cerr));
    cudaMemcpy(&g->num_edges, &g->row_ptr[g->capacity], sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_pairs);
    return 0;
}

int assoc_graph_decay_weights(AssociativeGraphGPU* g, float decay_rate) {
    if (!g) return -1;
    cudaMemcpy(&g->num_edges, &g->row_ptr[g->capacity], sizeof(int), cudaMemcpyDeviceToHost);
    int n = g->num_edges;
    if (n > g->max_edges) n = g->max_edges;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks > 256) blocks = 256;
    decay_weights_csr_kernel<<<blocks, BLOCK_SIZE>>>(g->values, n, decay_rate);
    return 0;
}

int assoc_graph_prune_weights(AssociativeGraphGPU* g, float threshold) {
    if (!g) return -1;
    int blocks = (g->capacity + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks > 256) blocks = 256;
    prune_weights_csr_kernel<<<blocks, BLOCK_SIZE>>>(
        g->row_ptr, g->col_indices, g->values, g->degree,
        g->capacity, threshold
    );
    rebuild_row_ptr_kernel<<<1, BLOCK_SIZE>>>(
        g->row_ptr, g->degree, g->capacity
    );
    cudaMemcpy(&g->num_edges, &g->row_ptr[g->capacity], sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return 0;
}

int assoc_graph_evict_oldest(AssociativeGraphGPU* g, float ratio) {
    int blocks = (g->capacity + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks > 256) blocks = 256;
    evict_oldest_csr_kernel<<<blocks, BLOCK_SIZE>>>(
        g->node_ids, g->row_ptr, g->capacity, &g->num_edges, g->node_labels
    );
    return 0;
}

int assoc_graph_panic_clear(AssociativeGraphGPU* g, float adrenaline, float panic_threshold) {
    if (!g) return -1;
    int blocks = (g->capacity + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks > 256) blocks = 256;
    panic_clear_csr_kernel<<<blocks, BLOCK_SIZE>>>(
        g->row_ptr, g->values, g->col_indices,
        g->capacity, g->node_ids, adrenaline, panic_threshold
    );
    return 0;
}

void assoc_graph_reset_activations(AssociativeGraphGPU* g) {
    if (g) cudaMemset(g->activation, 0, g->capacity * sizeof(float));
}

int assoc_graph_active_count(AssociativeGraphGPU* g) {
    if (!g) return 0;
    float coh = assoc_graph_coherence(g);
    return (int)(coh * g->capacity);
}

int assoc_graph_node_count(AssociativeGraphGPU* g) {
    return g ? g->node_count : 0;
}

int assoc_graph_capacity(AssociativeGraphGPU* g) {
    return g ? g->capacity : 0;
}

int assoc_graph_slots(AssociativeGraphGPU* g) {
    return g ? (g->capacity > 0 ? g->max_edges / g->capacity : 0) : 0;
}

ConceptId assoc_graph_get_node_id(AssociativeGraphGPU* g, int idx) {
    if (!g || idx < 0 || idx >= g->capacity) return 0;
    ConceptId id;
    cudaMemcpy(&id, &g->node_ids[idx], sizeof(ConceptId), cudaMemcpyDeviceToHost);
    return id;
}

int assoc_graph_get_node_ids(AssociativeGraphGPU* g, ConceptId* dst, int max_len) {
    if (!g || !dst) return 0;
    int n = g->node_count < max_len ? g->node_count : max_len;
    cudaMemcpy(dst, g->node_ids, n * sizeof(ConceptId), cudaMemcpyDeviceToHost);
    return n;
}

int assoc_graph_get_activations(AssociativeGraphGPU* g, float* dst, int max_len) {
    if (!g || !dst) return 0;
    int n = g->capacity < max_len ? g->capacity : max_len;
    cudaMemcpy(dst, g->activation, n * sizeof(float), cudaMemcpyDeviceToHost);
    return n;
}

int assoc_graph_get_weights(AssociativeGraphGPU* g, int node_idx, float* dst, int max_len) {
    if (!g || !dst || node_idx < 0 || node_idx >= g->capacity) return 0;
    int start, end;
    cudaMemcpy(&start, &g->row_ptr[node_idx], sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&end, &g->row_ptr[node_idx + 1], sizeof(int), cudaMemcpyDeviceToHost);
    int n = end - start;
    if (n > max_len) n = max_len;
    cudaMemcpy(dst, &g->values[start], n * sizeof(float), cudaMemcpyDeviceToHost);
    return n;
}

int assoc_graph_top_k(AssociativeGraphGPU* g, int k) {
    if (!g || k < 1) return -1;
    int blocks = (g->capacity + 255) / 256;
    if (blocks > 256) blocks = 256;
    top_k_filter_kernel<<<blocks, 256>>>(g->activation, g->capacity, k);
    cudaDeviceSynchronize();
    return 0;
}

// ─── STDP Boost Kernel ────────────────────────────────────────
// Boost short-term weights for active edges during BFS traversal
__global__ void stdp_boost_kernel(
    float* short_values,
    float* activation,
    int* row_ptr,
    int capacity,
    float boost_amount
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= capacity) return;

    if (activation[node] > 0.0f) {
        int start = row_ptr[node];
        int end = row_ptr[node + 1];
        for (int i = start; i < end; i++) {
            atomicAdd(&short_values[i], boost_amount * activation[node]);
        }
    }
}

// ─── STDP Decay Kernel ───────────────────────────────────────
// Decay short-term weights towards zero
__global__ void stdp_decay_kernel(
    float* short_values,
    int max_edges,
    float decay_rate
) {
    int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge >= max_edges) return;

    short_values[edge] *= decay_rate;
}

// ─── STDP API Functions ─────────────────────────────────────
int assoc_graph_stdp_boost(AssociativeGraphGPU* g, float boost) {
    if (!g) return -1;
    int blocks = (g->capacity + 255) / 256;
    if (blocks > 256) blocks = 256;
    stdp_boost_kernel<<<blocks, 256>>>(g->short_values, g->activation, g->row_ptr, g->capacity, boost);
    cudaDeviceSynchronize();
    return 0;
}

int assoc_graph_stdp_decay(AssociativeGraphGPU* g, float decay_rate) {
    if (!g) return -1;
    int blocks = (g->max_edges + 255) / 256;
    if (blocks > 256) blocks = 256;
    stdp_decay_kernel<<<blocks, 256>>>(g->short_values, g->max_edges, decay_rate);
    cudaDeviceSynchronize();
    return 0;
}

int assoc_graph_stdp_get_weights(AssociativeGraphGPU* g, int node_idx, float* dst, int max_len) {
    if (!g || !dst || node_idx < 0 || node_idx >= g->capacity) return 0;
    int start, end;
    cudaMemcpy(&start, &g->row_ptr[node_idx], sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&end, &g->row_ptr[node_idx + 1], sizeof(int), cudaMemcpyDeviceToHost);
    int n = end - start;
    if (n > max_len) n = max_len;
    cudaMemcpy(dst, &g->short_values[start], n * sizeof(float), cudaMemcpyDeviceToHost);
    return n;
}

// ─── Predictive Coding Kernel ──────────────────────────────────
// After BFS, each active node checks if its neighbors were
// correctly predicted. Correct predictions → strengthen,
// prediction errors → depotentiate (surprise signal).
#define PREDICTION_THRESHOLD 0.15f
#define SURPRISE_THRESHOLD   0.40f

__global__ void predictive_coding_kernel(
    float* activation,
    int* row_ptr,
    int* col_indices,
    float* values,
    float* short_values,
    int capacity,
    float learning_rate
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity) return;
    if (activation[i] <= 0.0f) return;

    int start = row_ptr[i];
    int end = row_ptr[i + 1];
    float act_i = activation[i];

    for (int k = start; k < end; k++) {
        int j = col_indices[k];
        if (j < 0 || j >= capacity) continue;

        float weight = values[k] + short_values[k];
        if (weight > 1.0f) weight = 1.0f;
        if (weight < 0.01f) continue;

        // Prediction: what energy should neighbor j have?
        float predicted = act_i * weight * ENERGY_DECAY;
        // Actual activation of neighbor
        float actual = activation[j];
        // Prediction error
        float error = fabsf(predicted - actual);

        if (error < PREDICTION_THRESHOLD) {
            // Prediction correct → strengthen
            float boost = learning_rate * (1.0f - error / PREDICTION_THRESHOLD);
            atomicAdd(&short_values[k], boost * weight);
            if (short_values[k] > 0.3f) short_values[k] = 0.3f;
        } else if (error > SURPRISE_THRESHOLD) {
            // Surprise → depotentiate
            float penalty = learning_rate * (error / SURPRISE_THRESHOLD);
            atomicAdd(&short_values[k], -penalty * weight);
            if (short_values[k] < 0.0f) short_values[k] = 0.0f;
        }
    }
}

// ─── GSOM Growth: grow CSR matrix capacity ──────────────────────
// Allocates new larger buffers, copies old data, frees old buffers.
int assoc_graph_grow(AssociativeGraphGPU* g, int new_capacity) {
    if (!g || new_capacity <= g->capacity) return -1;

    int old_capacity = g->capacity;
    int new_max_edges = new_capacity * (g->max_edges / g->capacity);

    // Allocate new buffers
    ConceptId* new_node_ids;
    char* new_node_labels;
    int* new_node_modality;
    int* new_row_ptr;
    int* new_col_indices;
    float* new_values;
    float* new_short_values;
    float* new_activation;
    ConceptId* new_frontier;
    ConceptId* new_next_frontier;
    int* new_visited;
    int* new_degree;

    cudaMalloc(&new_node_ids, new_capacity * sizeof(ConceptId));
    cudaMalloc(&new_node_labels, new_capacity * MAX_LABEL_LEN);
    cudaMalloc(&new_node_modality, new_capacity * sizeof(int));
    cudaMalloc(&new_row_ptr, (new_capacity + 1) * sizeof(int));
    cudaMalloc(&new_col_indices, new_max_edges * sizeof(int));
    cudaMalloc(&new_values, new_max_edges * sizeof(float));
    cudaMalloc(&new_short_values, new_max_edges * sizeof(float));
    cudaMalloc(&new_activation, new_capacity * sizeof(float));
    cudaMalloc(&new_frontier, new_capacity * sizeof(ConceptId));
    cudaMalloc(&new_next_frontier, new_capacity * sizeof(ConceptId));
    cudaMalloc(&new_visited, new_capacity * sizeof(int));
    cudaMalloc(&new_degree, new_capacity * sizeof(int));

    // Copy old data to new buffers
    cudaMemcpy(new_node_ids, g->node_ids, old_capacity * sizeof(ConceptId), cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_node_labels, g->node_labels, old_capacity * MAX_LABEL_LEN, cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_node_modality, g->node_modality, old_capacity * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_row_ptr, g->row_ptr, (old_capacity + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
    // Preserve row_ptr[capacity] which holds num_edges
    int old_num_edges;
    cudaMemcpy(&old_num_edges, &g->row_ptr[old_capacity], sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(new_col_indices, g->col_indices, old_num_edges * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_values, g->values, old_num_edges * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_short_values, g->short_values, old_num_edges * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_activation, g->activation, old_capacity * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_frontier, g->frontier, old_capacity * sizeof(ConceptId), cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_next_frontier, g->next_frontier, old_capacity * sizeof(ConceptId), cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_visited, g->visited, old_capacity * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_degree, g->degree, old_capacity * sizeof(int), cudaMemcpyDeviceToDevice);

    // Zero out new rows/columns (capacity beyond old_capacity)
    cudaMemset(&new_node_ids[old_capacity], 0, (new_capacity - old_capacity) * sizeof(ConceptId));
    cudaMemset(&new_node_labels[old_capacity * MAX_LABEL_LEN], 0, (new_capacity - old_capacity) * MAX_LABEL_LEN);
    cudaMemset(&new_node_modality[old_capacity], 0, (new_capacity - old_capacity) * sizeof(int));
    cudaMemset(&new_row_ptr[old_capacity + 1], 0, (new_capacity - old_capacity) * sizeof(int));
    cudaMemset(&new_activation[old_capacity], 0, (new_capacity - old_capacity) * sizeof(float));
    cudaMemset(&new_frontier[old_capacity], 0, (new_capacity - old_capacity) * sizeof(ConceptId));
    cudaMemset(&new_next_frontier[old_capacity], 0, (new_capacity - old_capacity) * sizeof(ConceptId));
    cudaMemset(&new_visited[old_capacity], 0, (new_capacity - old_capacity) * sizeof(int));
    cudaMemset(&new_degree[old_capacity], 0, (new_capacity - old_capacity) * sizeof(int));

    // Copy row_ptr[capacity] (num_edges) to new row_ptr[new_capacity]
    cudaMemcpy(&new_row_ptr[new_capacity], &new_row_ptr[old_capacity], sizeof(int), cudaMemcpyDeviceToDevice);

    // Free old buffers
    cudaFree(g->node_ids);
    cudaFree(g->node_labels);
    cudaFree(g->node_modality);
    cudaFree(g->row_ptr);
    cudaFree(g->col_indices);
    cudaFree(g->values);
    cudaFree(g->short_values);
    cudaFree(g->activation);
    cudaFree(g->frontier);
    cudaFree(g->next_frontier);
    cudaFree(g->frontier_next_count);
    cudaFree(g->visited);
    cudaFree(g->degree);

    // Update pointers
    g->node_ids = new_node_ids;
    g->node_labels = new_node_labels;
    g->node_modality = new_node_modality;
    g->row_ptr = new_row_ptr;
    g->col_indices = new_col_indices;
    g->values = new_values;
    g->short_values = new_short_values;
    g->activation = new_activation;
    g->frontier = new_frontier;
    g->next_frontier = new_next_frontier;
    g->visited = new_visited;
    g->degree = new_degree;
    g->capacity = new_capacity;
    g->max_edges = new_max_edges;

    return 0;
}

// ─── Predictive Coding API ─────────────────────────────────────
int assoc_graph_predictive_step(AssociativeGraphGPU* g, float learning_rate) {
    if (!g) return -1;
    int blocks = (g->capacity + 255) / 256;
    if (blocks > 256) blocks = 256;
    predictive_coding_kernel<<<blocks, 256>>>(
        g->activation, g->row_ptr, g->col_indices,
        g->values, g->short_values,
        g->capacity, learning_rate
    );
    cudaDeviceSynchronize();
    return 0;
}

} // extern "C"
