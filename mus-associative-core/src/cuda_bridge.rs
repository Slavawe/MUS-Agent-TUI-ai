use std::ffi::{CStr, CString};
use std::os::raw::c_char;

type ConceptId = u64;

#[repr(C)]
struct AssociativeGraphGPU {
    _private: [u8; 0],
}

extern "C" {
    fn assoc_graph_create_ffi(capacity: i32, slots_per_node: i32) -> *mut AssociativeGraphGPU;
    fn assoc_graph_destroy_ffi(g: *mut AssociativeGraphGPU);
    fn assoc_graph_add_node_ffi(g: *mut AssociativeGraphGPU, id: ConceptId, label: *const c_char, modality: i32) -> i32;
    fn assoc_graph_link_ffi(g: *mut AssociativeGraphGPU, a: ConceptId, b: ConceptId) -> i32;
    fn assoc_graph_activate_ffi(g: *mut AssociativeGraphGPU, seed: ConceptId, depth: i32) -> i32;
    fn assoc_graph_coherence_ffi(g: *mut AssociativeGraphGPU) -> f32;
    fn assoc_graph_saturation_ffi(g: *mut AssociativeGraphGPU) -> f32;
    fn assoc_graph_hebbian_learn_ffi(g: *mut AssociativeGraphGPU, active_set: *const ConceptId, active_len: i32) -> i32;
    fn assoc_graph_batch_link_ffi(g: *mut AssociativeGraphGPU, pairs: *const ConceptId, num_pairs: i32) -> i32;
    fn assoc_graph_evict_oldest_ffi(g: *mut AssociativeGraphGPU, ratio: f32) -> i32;
    fn assoc_graph_reset_activations_ffi(g: *mut AssociativeGraphGPU);
    fn assoc_graph_active_count_ffi(g: *mut AssociativeGraphGPU) -> i32;
    fn assoc_graph_node_count_ffi(g: *mut AssociativeGraphGPU) -> i32;
    fn assoc_graph_capacity_ffi(g: *mut AssociativeGraphGPU) -> i32;
    fn assoc_graph_slots_ffi(g: *mut AssociativeGraphGPU) -> i32;
    fn assoc_graph_get_node_id_ffi(g: *mut AssociativeGraphGPU, idx: i32) -> ConceptId;
    fn assoc_graph_get_node_ids_ffi(g: *mut AssociativeGraphGPU, dst: *mut ConceptId, max_len: i32) -> i32;
    fn assoc_graph_get_activations_ffi(g: *mut AssociativeGraphGPU, dst: *mut f32, max_len: i32) -> i32;
}

pub struct CudaGraph {
    inner: *mut AssociativeGraphGPU,
    capacity: i32,
    slots_per_node: i32,
}

impl CudaGraph {
    pub fn new(capacity: i32, slots_per_node: i32) -> Self {
        let inner = unsafe { assoc_graph_create_ffi(capacity, slots_per_node) };
        assert!(!inner.is_null(), "assoc_graph_create_ffi failed");
        CudaGraph { inner, capacity, slots_per_node }
    }

    pub fn add_node(&mut self, id: ConceptId, label: &str, modality: i32) {
        let c_label = CString::new(label).unwrap();
        unsafe { assoc_graph_add_node_ffi(self.inner, id, c_label.as_ptr(), modality); }
    }

    pub fn link(&mut self, a: ConceptId, b: ConceptId) {
        unsafe { assoc_graph_link_ffi(self.inner, a, b); }
    }

    pub fn activate(&mut self, seed: ConceptId, depth: i32) -> i32 {
        unsafe { assoc_graph_activate_ffi(self.inner, seed, depth) }
    }

    pub fn coherence(&self) -> f32 {
        unsafe { assoc_graph_coherence_ffi(self.inner) }
    }

    pub fn saturation(&self) -> f32 {
        unsafe { assoc_graph_saturation_ffi(self.inner) }
    }

    pub fn batch_link(&mut self, pairs: &[(u64, u64)]) -> i32 {
        let flat: Vec<u64> = pairs.iter().flat_map(|&(a, b)| vec![a, b]).collect();
        unsafe { assoc_graph_batch_link_ffi(self.inner, flat.as_ptr(), pairs.len() as i32) }
    }

    pub fn hebbian_learn(&mut self, active_set: &[ConceptId]) -> i32 {
        unsafe { assoc_graph_hebbian_learn_ffi(self.inner, active_set.as_ptr(), active_set.len() as i32) }
    }

    pub fn evict_oldest(&mut self, ratio: f32) -> i32 {
        unsafe { assoc_graph_evict_oldest_ffi(self.inner, ratio) }
    }

    pub fn reset_activations(&mut self) {
        unsafe { assoc_graph_reset_activations_ffi(self.inner); }
    }

    pub fn active_count(&self) -> i32 {
        unsafe { assoc_graph_active_count_ffi(self.inner) }
    }

    pub fn node_count(&self) -> i32 {
        unsafe { assoc_graph_node_count_ffi(self.inner) }
    }

    pub fn get_node_ids(&self) -> Vec<ConceptId> {
        let n = self.node_count() as usize;
        let mut ids = vec![0u64; n];
        unsafe { assoc_graph_get_node_ids_ffi(self.inner, ids.as_mut_ptr(), n as i32); }
        ids
    }

    pub fn get_activations(&self) -> Vec<f32> {
        let n = self.node_count() as usize;
        let mut acts = vec![0.0f32; n];
        unsafe { assoc_graph_get_activations_ffi(self.inner, acts.as_mut_ptr(), n as i32); }
        acts
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        unsafe { assoc_graph_destroy_ffi(self.inner); }
    }
}
