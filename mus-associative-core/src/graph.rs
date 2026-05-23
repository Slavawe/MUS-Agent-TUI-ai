use std::collections::{HashMap, HashSet, VecDeque};

pub type ConceptId = u64;

const MAX_ASSOC_PER_NODE: usize = 64;

pub struct Node {
    pub id: ConceptId,
    pub label: String,
    pub modality: Modality,
    pub associations: Vec<ConceptId>,
    pub activation: f32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Modality {
    Text,
    Vision,
    Audio,
    Composite,
}

pub struct Graph {
    pub nodes: HashMap<ConceptId, Node>,
    pub slots_per_node: usize,
    pub coherence_history: VecDeque<f32>,
}

impl Graph {
    pub fn new(slots_per_node: usize) -> Self {
        Graph {
            nodes: HashMap::new(),
            slots_per_node: slots_per_node.min(MAX_ASSOC_PER_NODE),
            coherence_history: VecDeque::with_capacity(100),
        }
    }

    pub fn get_or_create(&mut self, id: ConceptId, label: &str, modality: Modality) -> &mut Node {
        let slots = self.slots_per_node;
        self.nodes.entry(id).or_insert_with(|| Node {
            id,
            label: label.to_string(),
            modality,
            associations: Vec::with_capacity(slots),
            activation: 0.0,
        })
    }

    pub fn link(&mut self, a: ConceptId, b: ConceptId) {
        if a == b { return; }
        self.add_assoc(a, b);
        self.add_assoc(b, a);
    }

    fn add_assoc(&mut self, src: ConceptId, dst: ConceptId) {
        if let Some(node) = self.nodes.get_mut(&src) {
            if node.associations.len() >= self.slots_per_node {
                return;
            }
            if !node.associations.contains(&dst) {
                node.associations.push(dst);
            }
        }
    }

    pub fn activate(&mut self, seed: ConceptId, depth: usize) -> HashSet<ConceptId> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back((seed, 0));
        visited.insert(seed);

        while let Some((id, d)) = queue.pop_front() {
            if d >= depth { continue; }
            if let Some(node) = self.nodes.get(&id) {
                for &assoc in &node.associations {
                    if visited.insert(assoc) {
                        queue.push_back((assoc, d + 1));
                    }
                }
            }
        }

        for &id in &visited {
            if let Some(node) = self.nodes.get_mut(&id) {
                node.activation = 1.0;
            }
        }

        visited
    }

    pub fn slot_saturation(&self) -> f32 {
        if self.nodes.is_empty() { return 0.0; }
        let total: usize = self.nodes.values().map(|n| n.associations.len()).sum();
        let max = self.nodes.len() * self.slots_per_node;
        total as f32 / max as f32
    }

    pub fn coherence(&self) -> f32 {
        if self.nodes.len() < 2 { return 1.0; }
        let connected = self.nodes.values()
            .filter(|n| !n.associations.is_empty())
            .count();
        connected as f32 / self.nodes.len() as f32
    }

    pub fn prune_rare(&mut self, _threshold: usize) -> usize {
        let mut removed = 0;
        for node in self.nodes.values_mut() {
            let before = node.associations.len();
            node.associations.retain(|_| {
                true
            });
            removed += before - node.associations.len();
        }
        removed
    }

    pub fn evict_oldest(&mut self, ratio: f32) -> usize {
        if self.nodes.is_empty() { return 0; }
        let to_evict = (self.nodes.len() as f32 * ratio).max(1.0) as usize;
        let ids: Vec<ConceptId> = self.nodes.keys().copied().take(to_evict).collect();
        for id in &ids {
            self.nodes.remove(id);
            for node in self.nodes.values_mut() {
                node.associations.retain(|a| a != id);
            }
        }
        ids.len()
    }

    pub fn active_concepts(&self) -> usize {
        self.nodes.values().filter(|n| n.activation > 0.5).count()
    }

    pub fn reset_activations(&mut self) {
        for node in self.nodes.values_mut() {
            node.activation = 0.0;
        }
    }
}
