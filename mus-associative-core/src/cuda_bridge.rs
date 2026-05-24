#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ffi::CString;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CudaSystemState {
    pub dopamin: f32,
    pub adrenaline: f32,
    pub energy: f32,
    pub energy_decay: f32,
    pub panic_threshold: f32,
    pub weight_decay: f32,
    pub weight_prune: f32,
    pub weight_boost: f32,
    pub use_chemistry: i32,
}

#[repr(C)]
struct AssociativeGraphGPU {
    node_ids: *mut u64,
    node_labels: *mut u8,
    node_modality: *mut i32,
    row_ptr: *mut i32,
    col_indices: *mut i32,
    values: *mut f32,
    short_values: *mut f32,        // STDP: short-term weights
    num_edges: i32,
    max_edges: i32,
    activation: *mut f32,
    frontier: *mut u64,
    next_frontier: *mut u64,
    frontier_next_count: *mut i32,
    visited: *mut i32,
    degree: *mut i32,              // [capacity] edge count per node
    capacity: i32,
    node_count: i32,
    use_chemistry: i32,
    stream: *mut std::ffi::c_void,
}

extern "C" {
    fn assoc_graph_create(capacity: i32, max_edges_per_node: i32) -> *mut AssociativeGraphGPU;
    fn assoc_graph_destroy(g: *mut AssociativeGraphGPU);
    fn assoc_graph_add_node(g: *mut AssociativeGraphGPU, id: u64, label: *const i8, modality: i32) -> i32;
    fn assoc_graph_link(g: *mut AssociativeGraphGPU, a: u64, b: u64) -> i32;
    fn assoc_graph_activate(g: *mut AssociativeGraphGPU, seed: u64, depth: i32) -> i32;
    fn assoc_graph_activate_chem(g: *mut AssociativeGraphGPU, seed: u64, max_depth: i32, state: *const CudaSystemState) -> i32;
    fn assoc_graph_coherence(g: *mut AssociativeGraphGPU) -> f32;
    fn assoc_graph_saturation(g: *mut AssociativeGraphGPU) -> f32;
    fn assoc_graph_hebbian_learn(g: *mut AssociativeGraphGPU, active_set: *const u64, active_len: i32, dopamin: f32, adrenaline: f32) -> i32;
    fn assoc_graph_batch_link(g: *mut AssociativeGraphGPU, pairs: *const u64, num_pairs: i32, dopamin: f32, adrenaline: f32) -> i32;
    fn assoc_graph_decay_weights(g: *mut AssociativeGraphGPU, decay_rate: f32) -> i32;
    fn assoc_graph_prune_weights(g: *mut AssociativeGraphGPU, threshold: f32) -> i32;
    fn assoc_graph_evict_oldest(g: *mut AssociativeGraphGPU, ratio: f32) -> i32;
    fn assoc_graph_panic_clear(g: *mut AssociativeGraphGPU, adrenaline: f32, panic_threshold: f32) -> i32;
    fn assoc_graph_reset_activations(g: *mut AssociativeGraphGPU);
    fn assoc_graph_active_count(g: *mut AssociativeGraphGPU) -> i32;
    fn assoc_graph_node_count(g: *mut AssociativeGraphGPU) -> i32;
    fn assoc_graph_capacity(g: *mut AssociativeGraphGPU) -> i32;
    fn assoc_graph_slots(g: *mut AssociativeGraphGPU) -> i32;
    fn assoc_graph_get_node_id(g: *mut AssociativeGraphGPU, idx: i32) -> u64;
    fn assoc_graph_get_node_ids(g: *mut AssociativeGraphGPU, dst: *mut u64, max_len: i32) -> i32;
    fn assoc_graph_get_activations(g: *mut AssociativeGraphGPU, dst: *mut f32, max_len: i32) -> i32;
    fn assoc_graph_get_weights(g: *mut AssociativeGraphGPU, node_idx: i32, dst: *mut f32, max_len: i32) -> i32;
    fn assoc_graph_top_k(g: *mut AssociativeGraphGPU, k: i32) -> i32;
    // STDP: Short-Term Plasticity
    fn assoc_graph_stdp_boost(g: *mut AssociativeGraphGPU, boost: f32) -> i32;
    fn assoc_graph_stdp_decay(g: *mut AssociativeGraphGPU, decay_rate: f32) -> i32;
    fn assoc_graph_stdp_get_weights(g: *mut AssociativeGraphGPU, node_idx: i32, dst: *mut f32, max_len: i32) -> i32;
    // Predictive Coding
    fn assoc_graph_predictive_step(g: *mut AssociativeGraphGPU, learning_rate: f32) -> i32;
    // GSOM: dynamic capacity growth
    fn assoc_graph_grow(g: *mut AssociativeGraphGPU, new_capacity: i32) -> i32;
}

pub struct CudaGraph {
    ptr: *mut AssociativeGraphGPU,
}

unsafe impl Send for CudaGraph {}

impl CudaGraph {
    pub fn new(capacity: i32, max_edges_per_node: i32) -> Self {
        let ptr = unsafe { assoc_graph_create(capacity, max_edges_per_node) };
        assert!(!ptr.is_null(), "CudaGraph creation failed");
        CudaGraph { ptr }
    }

    pub fn add_node(&self, id: u64, label: &str, modality: i32) -> i32 {
        let c_label = CString::new(label).unwrap();
        unsafe { assoc_graph_add_node(self.ptr, id, c_label.as_ptr(), modality) }
    }

    pub fn link(&self, a: u64, b: u64) {
        unsafe { assoc_graph_link(self.ptr, a, b); }
    }

    pub fn activate(&self, seed: u64, depth: i32) -> i32 {
        unsafe { assoc_graph_activate(self.ptr, seed, depth) }
    }

    pub fn activate_chem(&self, seed: u64, max_depth: i32, state: &CudaSystemState) -> i32 {
        unsafe { assoc_graph_activate_chem(self.ptr, seed, max_depth, state as *const CudaSystemState) }
    }

    pub fn coherence(&self) -> f32 {
        unsafe { assoc_graph_coherence(self.ptr) }
    }

    pub fn saturation(&self) -> f32 {
        unsafe { assoc_graph_saturation(self.ptr) }
    }

    pub fn hebbian_learn(&self, active_set: &[u64], dopamin: f32, adrenaline: f32) {
        if active_set.is_empty() { return; }
        unsafe { assoc_graph_hebbian_learn(self.ptr, active_set.as_ptr(), active_set.len() as i32, dopamin, adrenaline); }
    }

    pub fn batch_link(&self, pairs: &[u64], dopamin: f32, adrenaline: f32) {
        if pairs.is_empty() || pairs.len() % 2 != 0 { return; }
        unsafe { assoc_graph_batch_link(self.ptr, pairs.as_ptr(), (pairs.len() / 2) as i32, dopamin, adrenaline); }
    }

    pub fn decay_weights(&self, decay_rate: f32) {
        unsafe { assoc_graph_decay_weights(self.ptr, decay_rate); }
    }

    pub fn prune_weights(&self, threshold: f32) {
        unsafe { assoc_graph_prune_weights(self.ptr, threshold); }
    }

    pub fn evict_oldest(&self, ratio: f32) -> i32 {
        unsafe { assoc_graph_evict_oldest(self.ptr, ratio) }
    }

    pub fn panic_clear(&self, adrenaline: f32, panic_threshold: f32) -> i32 {
        unsafe { assoc_graph_panic_clear(self.ptr, adrenaline, panic_threshold) }
    }

    pub fn reset_activations(&self) {
        unsafe { assoc_graph_reset_activations(self.ptr); }
    }

    pub fn active_count(&self) -> i32 {
        unsafe { assoc_graph_active_count(self.ptr) }
    }

    pub fn node_count(&self) -> i32 {
        unsafe { assoc_graph_node_count(self.ptr) }
    }

    pub fn capacity(&self) -> i32 {
        unsafe { assoc_graph_capacity(self.ptr) }
    }

    pub fn max_edges_per_node(&self) -> i32 {
        unsafe { assoc_graph_slots(self.ptr) }
    }

    pub fn get_node_id(&self, idx: i32) -> u64 {
        unsafe { assoc_graph_get_node_id(self.ptr, idx) }
    }

    pub fn get_node_ids(&self) -> Vec<u64> {
        let count = self.node_count() as usize;
        let mut dst = vec![0u64; count];
        unsafe { assoc_graph_get_node_ids(self.ptr, dst.as_mut_ptr(), count as i32); }
        dst
    }

    pub fn get_activations(&self) -> Vec<f32> {
        let count = self.capacity() as usize;
        let mut dst = vec![0.0f32; count];
        unsafe { assoc_graph_get_activations(self.ptr, dst.as_mut_ptr(), count as i32); }
        dst
    }

    pub fn get_weights(&self, node_idx: i32) -> Vec<f32> {
        let max = self.max_edges_per_node() as usize;
        let mut dst = vec![0.0f32; max];
        let n = unsafe { assoc_graph_get_weights(self.ptr, node_idx, dst.as_mut_ptr(), max as i32) };
        dst.truncate(n as usize);
        dst
    }

    pub fn top_k(&self, k: i32) {
        unsafe { assoc_graph_top_k(self.ptr, k); }
    }

    // STDP: Short-Term Plasticity
    pub fn stdp_boost(&self, boost: f32) {
        unsafe { assoc_graph_stdp_boost(self.ptr, boost); }
    }

    pub fn stdp_decay(&self, decay_rate: f32) {
        unsafe { assoc_graph_stdp_decay(self.ptr, decay_rate); }
    }

    pub fn stdp_get_weights(&self, node_idx: i32) -> Vec<f32> {
        let max = self.max_edges_per_node() as usize;
        let mut dst = vec![0.0f32; max];
        let n = unsafe { assoc_graph_stdp_get_weights(self.ptr, node_idx, dst.as_mut_ptr(), max as i32) };
        dst.truncate(n as usize);
        dst
    }

    // Predictive Coding
    pub fn predictive_step(&self, learning_rate: f32) {
        unsafe { assoc_graph_predictive_step(self.ptr, learning_rate); }
    }

    // GSOM: grow graph capacity
    pub fn grow(&self, new_capacity: i32) -> i32 {
        unsafe { assoc_graph_grow(self.ptr, new_capacity) }
    }

    pub fn get_top_activations(&self, n: usize) -> Vec<(u64, f32)> {
        let ids = self.get_node_ids();
        let acts = self.get_activations();
        let mut pairs: Vec<(u64, f32)> = ids.into_iter()
            .zip(acts.into_iter())
            .filter(|(id, act)| *id != 0 && *act > 0.0)
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.truncate(n);
        pairs
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        unsafe { assoc_graph_destroy(self.ptr); }
    }
}
