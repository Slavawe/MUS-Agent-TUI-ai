#ifndef MUS_ASSOCIATIVE_H
#define MUS_ASSOCIATIVE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t ConceptId;

#define MAX_LABEL_LEN 32
#define ASSOC_WEIGHT_INIT 1.0f
#define ASSOC_WEIGHT_DECAY 0.995f
#define ASSOC_WEIGHT_PRUNE 0.05f
#define ASSOC_WEIGHT_BOOST 0.15f
#define ASSOC_WEIGHT_MAX 1.0f

// CSR-format graph: edges stored as compressed sparse rows
// row_ptr[i] .. row_ptr[i+1]-1  = edges for node i
// col_indices[row_ptr[i] + k]   = neighbor node INDEX
// values[row_ptr[i] + k]        = edge weight
typedef struct __align__(128) {
    ConceptId* node_ids;
    char* node_labels;
    int* node_modality;
    int* row_ptr;           // [capacity + 1] cumulative edge count
    int* col_indices;       // [max_edges] neighbor node indices
    float* values;          // [max_edges] long-term edge weights
    float* short_values;    // [max_edges] short-term plasticity boost
    int num_edges;
    int max_edges;
    float* activation;
    ConceptId* frontier;
    ConceptId* next_frontier;
    int* frontier_next_count;
    int* visited;
    int* degree;            // [capacity] edge count per node
    int capacity;
    int node_count;
    int use_chemistry;
    void* stream;
} AssociativeGraphGPU;

typedef struct {
    float dopamin;
    float adrenaline;
    float energy;
    float energy_decay;
    float panic_threshold;
    float weight_decay;
    float weight_prune;
    float weight_boost;
    int use_chemistry;
} CudaSystemState;

AssociativeGraphGPU* assoc_graph_create(int capacity, int max_edges_per_node);
void assoc_graph_destroy(AssociativeGraphGPU* g);

int assoc_graph_add_node(AssociativeGraphGPU* g, ConceptId id, const char* label, int modality);
int assoc_graph_link(AssociativeGraphGPU* g, ConceptId a, ConceptId b);
int assoc_graph_has_edge(AssociativeGraphGPU* g, ConceptId a, ConceptId b);

int assoc_graph_activate(AssociativeGraphGPU* g, ConceptId seed, int depth);
int assoc_graph_activate_chem(AssociativeGraphGPU* g, ConceptId seed, int max_depth, const CudaSystemState* state);

float assoc_graph_coherence(AssociativeGraphGPU* g);
float assoc_graph_saturation(AssociativeGraphGPU* g);

int assoc_graph_hebbian_learn(AssociativeGraphGPU* g, const ConceptId* active_set, int active_len, float dopamin, float adrenaline);
int assoc_graph_batch_link(AssociativeGraphGPU* g, const ConceptId* pairs, int num_pairs, float dopamin, float adrenaline);
int assoc_graph_decay_weights(AssociativeGraphGPU* g, float decay_rate);
int assoc_graph_prune_weights(AssociativeGraphGPU* g, float threshold);
int assoc_graph_evict_oldest(AssociativeGraphGPU* g, float ratio);
int assoc_graph_panic_clear(AssociativeGraphGPU* g, float adrenaline, float panic_threshold);

void assoc_graph_reset_activations(AssociativeGraphGPU* g);
int assoc_graph_active_count(AssociativeGraphGPU* g);
int assoc_graph_node_count(AssociativeGraphGPU* g);
int assoc_graph_capacity(AssociativeGraphGPU* g);
int assoc_graph_slots(AssociativeGraphGPU* g);  // returns max degree
ConceptId assoc_graph_get_node_id(AssociativeGraphGPU* g, int idx);

int assoc_graph_get_node_ids(AssociativeGraphGPU* g, ConceptId* dst, int max_len);
int assoc_graph_get_activations(AssociativeGraphGPU* g, float* dst, int max_len);
int assoc_graph_get_weights(AssociativeGraphGPU* g, int node_idx, float* dst, int max_len);
int assoc_graph_top_k(AssociativeGraphGPU* g, int k);
int assoc_graph_short_decay(AssociativeGraphGPU* g, float decay_rate);
int assoc_graph_short_boost_edges(AssociativeGraphGPU* g, const ConceptId* frontier, int frontier_size, float boost);
int assoc_graph_short_get(AssociativeGraphGPU* g, int node_idx, float* dst, int max_len);

// Predictive Coding
int assoc_graph_predictive_step(AssociativeGraphGPU* g, float learning_rate);

// GSOM: dynamic capacity growth
int assoc_graph_grow(AssociativeGraphGPU* g, int new_capacity);

#ifdef __cplusplus
}
#endif

#endif
