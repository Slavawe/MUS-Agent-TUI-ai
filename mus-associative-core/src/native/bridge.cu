#include "mus_associative.h"

#ifdef __cplusplus
extern "C" {
#endif

AssociativeGraphGPU* assoc_graph_create_ffi(int capacity, int slots_per_node) {
    return assoc_graph_create(capacity, slots_per_node);
}

void assoc_graph_destroy_ffi(AssociativeGraphGPU* g) {
    assoc_graph_destroy(g);
}

int assoc_graph_add_node_ffi(AssociativeGraphGPU* g, ConceptId id, const char* label, int modality) {
    return assoc_graph_add_node(g, id, label, modality);
}

int assoc_graph_link_ffi(AssociativeGraphGPU* g, ConceptId a, ConceptId b) {
    return assoc_graph_link(g, a, b);
}

int assoc_graph_activate_ffi(AssociativeGraphGPU* g, ConceptId seed, int depth) {
    return assoc_graph_activate(g, seed, depth);
}

float assoc_graph_coherence_ffi(AssociativeGraphGPU* g) {
    return assoc_graph_coherence(g);
}

float assoc_graph_saturation_ffi(AssociativeGraphGPU* g) {
    return assoc_graph_saturation(g);
}

int assoc_graph_hebbian_learn_ffi(AssociativeGraphGPU* g, const ConceptId* active_set, int active_len) {
    return assoc_graph_hebbian_learn(g, active_set, active_len);
}

int assoc_graph_batch_link_ffi(AssociativeGraphGPU* g, const ConceptId* pairs, int num_pairs) {
    return assoc_graph_batch_link(g, pairs, num_pairs);
}

int assoc_graph_evict_oldest_ffi(AssociativeGraphGPU* g, float ratio) {
    return assoc_graph_evict_oldest(g, ratio);
}

void assoc_graph_reset_activations_ffi(AssociativeGraphGPU* g) {
    assoc_graph_reset_activations(g);
}

int assoc_graph_active_count_ffi(AssociativeGraphGPU* g) {
    return assoc_graph_active_count(g);
}

int assoc_graph_node_count_ffi(AssociativeGraphGPU* g) {
    return assoc_graph_node_count(g);
}

int assoc_graph_capacity_ffi(AssociativeGraphGPU* g) {
    return assoc_graph_capacity(g);
}

int assoc_graph_slots_ffi(AssociativeGraphGPU* g) {
    return assoc_graph_slots(g);
}

ConceptId assoc_graph_get_node_id_ffi(AssociativeGraphGPU* g, int idx) {
    return assoc_graph_get_node_id(g, idx);
}

int assoc_graph_get_node_ids_ffi(AssociativeGraphGPU* g, ConceptId* dst, int max_len) {
    return assoc_graph_get_node_ids(g, dst, max_len);
}

int assoc_graph_get_activations_ffi(AssociativeGraphGPU* g, float* dst, int max_len) {
    return assoc_graph_get_activations(g, dst, max_len);
}

#ifdef __cplusplus
}
#endif
