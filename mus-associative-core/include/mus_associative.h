#ifndef MUS_ASSOCIATIVE_H
#define MUS_ASSOCIATIVE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t ConceptId;

typedef enum {
    MODALITY_TEXT = 0,
    MODALITY_VISION = 1,
    MODALITY_AUDIO = 2,
    MODALITY_COMPOSITE = 3,
} Modality;

#define MAX_LABEL_LEN 32

typedef struct {
    ConceptId* node_ids;
    char* node_labels;
    int* node_modality;
    ConceptId* assoc_slots;
    int* assoc_count;
    float* activation;
    int* frontier;
    int* visited;
    int capacity;
    int slots_per_node;
    int node_count;
    void* stream;
} AssociativeGraphGPU;

AssociativeGraphGPU* assoc_graph_create(int capacity, int slots_per_node);
void assoc_graph_destroy(AssociativeGraphGPU* g);

int assoc_graph_add_node(AssociativeGraphGPU* g, ConceptId id, const char* label, int modality);
int assoc_graph_link(AssociativeGraphGPU* g, ConceptId a, ConceptId b);
int assoc_graph_activate(AssociativeGraphGPU* g, ConceptId seed, int depth);
float assoc_graph_coherence(AssociativeGraphGPU* g);
float assoc_graph_saturation(AssociativeGraphGPU* g);
int assoc_graph_hebbian_learn(AssociativeGraphGPU* g, const ConceptId* active_set, int active_len);
int assoc_graph_evict_oldest(AssociativeGraphGPU* g, float ratio);
void assoc_graph_reset_activations(AssociativeGraphGPU* g);
int assoc_graph_active_count(AssociativeGraphGPU* g);
int assoc_graph_node_count(AssociativeGraphGPU* g);
int assoc_graph_capacity(AssociativeGraphGPU* g);
int assoc_graph_slots(AssociativeGraphGPU* g);
ConceptId assoc_graph_get_node_id(AssociativeGraphGPU* g, int idx);

int assoc_graph_get_node_ids(AssociativeGraphGPU* g, ConceptId* dst, int max_len);
int assoc_graph_get_activations(AssociativeGraphGPU* g, float* dst, int max_len);

#ifdef __cplusplus
}
#endif

#endif
