#include "../include/mus_associative.h"

extern "C" {

AssociativeGraphGPU* bridge_graph_create(int capacity, int slots_per_node) {
    return assoc_graph_create(capacity, slots_per_node);
}

void bridge_graph_destroy(AssociativeGraphGPU* g) {
    assoc_graph_destroy(g);
}

int bridge_add_node(AssociativeGraphGPU* g, ConceptId id, const char* label, int modality) {
    return assoc_graph_add_node(g, id, label, modality);
}

int bridge_link(AssociativeGraphGPU* g, ConceptId a, ConceptId b) {
    return assoc_graph_link(g, a, b);
}

int bridge_activate(AssociativeGraphGPU* g, ConceptId seed, int depth) {
    return assoc_graph_activate(g, seed, depth);
}

int bridge_activate_chem(AssociativeGraphGPU* g, ConceptId seed, int max_depth, const CudaSystemState* state) {
    return assoc_graph_activate_chem(g, seed, max_depth, state);
}

float bridge_coherence(AssociativeGraphGPU* g) {
    return assoc_graph_coherence(g);
}

float bridge_saturation(AssociativeGraphGPU* g) {
    return assoc_graph_saturation(g);
}

int bridge_hebbian_learn(AssociativeGraphGPU* g, const ConceptId* active_set, int active_len, float dopamin, float adrenaline) {
    return assoc_graph_hebbian_learn(g, active_set, active_len, dopamin, adrenaline);
}

int bridge_batch_link(AssociativeGraphGPU* g, const ConceptId* pairs, int num_pairs, float dopamin, float adrenaline) {
    return assoc_graph_batch_link(g, pairs, num_pairs, dopamin, adrenaline);
}

int bridge_decay_weights(AssociativeGraphGPU* g, float decay_rate) {
    return assoc_graph_decay_weights(g, decay_rate);
}

int bridge_prune_weights(AssociativeGraphGPU* g, float threshold) {
    return assoc_graph_prune_weights(g, threshold);
}

int bridge_evict_oldest(AssociativeGraphGPU* g, float ratio) {
    return assoc_graph_evict_oldest(g, ratio);
}

int bridge_panic_clear(AssociativeGraphGPU* g, float adrenaline, float panic_threshold) {
    return assoc_graph_panic_clear(g, adrenaline, panic_threshold);
}

void bridge_reset_activations(AssociativeGraphGPU* g) {
    assoc_graph_reset_activations(g);
}

int bridge_active_count(AssociativeGraphGPU* g) {
    return assoc_graph_active_count(g);
}

int bridge_node_count(AssociativeGraphGPU* g) {
    return assoc_graph_node_count(g);
}

int bridge_capacity(AssociativeGraphGPU* g) {
    return assoc_graph_capacity(g);
}

int bridge_slots(AssociativeGraphGPU* g) {
    return assoc_graph_slots(g);
}

ConceptId bridge_get_node_id(AssociativeGraphGPU* g, int idx) {
    return assoc_graph_get_node_id(g, idx);
}

int bridge_get_node_ids(AssociativeGraphGPU* g, ConceptId* dst, int max_len) {
    return assoc_graph_get_node_ids(g, dst, max_len);
}

int bridge_get_activations(AssociativeGraphGPU* g, float* dst, int max_len) {
    return assoc_graph_get_activations(g, dst, max_len);
}

int bridge_get_weights(AssociativeGraphGPU* g, int node_idx, float* dst, int max_len) {
    return assoc_graph_get_weights(g, node_idx, dst, max_len);
}

int bridge_top_k(AssociativeGraphGPU* g, int k) {
    return assoc_graph_top_k(g, k);
}

} // extern "C"
