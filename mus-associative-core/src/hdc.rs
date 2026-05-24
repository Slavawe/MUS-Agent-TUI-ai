pub type ConceptId = u64;

pub const HDC_DIM: usize = 8192;

fn xorshift64(seed: &mut u64) -> u64 {
    let mut x = *seed;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *seed = x;
    x
}

pub fn bipolar_from_hash(seed: u64, dim: usize) -> Vec<f32> {
    let mut rng = seed;
    let mut hv = Vec::with_capacity(dim);
    for _ in 0..dim {
        hv.push(if (xorshift64(&mut rng) & 1) == 0 { 1.0 } else { -1.0 });
    }
    hv
}

pub fn role_hv(role_name: &str, dim: usize) -> Vec<f32> {
    let seed: u64 = role_name.bytes().fold(14695981039346656037, |h, b| {
        h ^ b as u64
    }).wrapping_mul(1099511628211);
    bipolar_from_hash(seed, dim)
}

fn position_hv(pos: usize, dim: usize) -> Vec<f32> {
    let mut rng = (pos as u64).wrapping_mul(0x9E3779B97F4A7C15);
    let mut hv = Vec::with_capacity(dim);
    for _ in 0..dim {
        hv.push(if (xorshift64(&mut rng) & 1) == 0 { 1.0 } else { -1.0 });
    }
    hv
}

pub fn bind(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// bind(role, concept) — then unbind with same role to recover concept
pub fn unbind(thought: &[f32], role: &[f32]) -> Vec<f32> {
    bind(thought, role)
}

pub fn bundle_into(acc: &mut [f32], other: &[f32]) {
    for (a, b) in acc.iter_mut().zip(other.iter()) {
        *a += b;
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

// ─── VSA Role Binding ────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThoughtRole {
    Subject,
    Action,
    Object,
    Modifier,
    Attribute,
    Modal,
}

impl ThoughtRole {
    pub fn name(&self) -> &'static str {
        match self {
            ThoughtRole::Subject => "subject",
            ThoughtRole::Action => "action",
            ThoughtRole::Object => "object",
            ThoughtRole::Modifier => "modifier",
            ThoughtRole::Attribute => "attribute",
            ThoughtRole::Modal => "modal",
        }
    }

    pub fn hv(&self, dim: usize) -> Vec<f32> {
        role_hv(self.name(), dim)
    }
}

/// Encode a concept into a role-bound hypervector: role ⊗ concept
pub fn bind_role(concept_id: ConceptId, role: ThoughtRole, dim: usize) -> Vec<f32> {
    let rhv = role.hv(dim);
    let chv = bipolar_from_hash(concept_id, dim);
    bind(&rhv, &chv)
}

/// Decode a concept from a role-bound thought vector by unbinding with role
pub fn unbind_role(thought: &[f32], role: ThoughtRole, dim: usize, candidates: &[ConceptId]) -> Option<ConceptId> {
    let rhv = role.hv(dim);
    let decoded = unbind(thought, &rhv);
    let mut best = None;
    let mut best_sim = -1.0f32;
    for &cid in candidates {
        let chv = bipolar_from_hash(cid, dim);
        let sim = cosine_similarity(&decoded, &chv);
        if sim > best_sim {
            best_sim = sim;
            best = Some(cid);
        }
    }
    if best_sim > 0.15 { best } else { None }
}

/// Encode a triple (subject, action, object) into one thought vector:
/// V_thought = (V_subj ⊗ V_subj_role) ⊕ (V_act ⊗ V_act_role) ⊕ (V_obj ⊗ V_obj_role)
pub fn encode_triple(subj: ConceptId, act: ConceptId, obj: ConceptId, dim: usize) -> Vec<f32> {
    let mut acc = vec![0.0; dim];
    bundle_into(&mut acc, &bind_role(subj, ThoughtRole::Subject, dim));
    bundle_into(&mut acc, &bind_role(act, ThoughtRole::Action, dim));
    bundle_into(&mut acc, &bind_role(obj, ThoughtRole::Object, dim));
    acc
}

/// Decode a triple thought vector into its three components
pub fn decode_triple(thought: &[f32], dim: usize, candidates: &[ConceptId])
    -> (Option<ConceptId>, Option<ConceptId>, Option<ConceptId>)
{
    let subj = unbind_role(thought, ThoughtRole::Subject, dim, candidates);
    let act  = unbind_role(thought, ThoughtRole::Action, dim, candidates);
    let obj  = unbind_role(thought, ThoughtRole::Object, dim, candidates);
    (subj, act, obj)
}

/// Bind a whole chain of concepts with roles (alternating subj/act/obj)
pub fn bind_chain(chain: &[(ConceptId, ThoughtRole)], dim: usize) -> Vec<f32> {
    if chain.is_empty() { return vec![0.0; dim]; }
    let mut acc = vec![0.0; dim];
    for &(cid, role) in chain {
        bundle_into(&mut acc, &bind_role(cid, role, dim));
    }
    acc
}

pub fn unbind_chain(thought: &[f32], chain_roles: &[ThoughtRole], dim: usize, candidates: &[ConceptId]) -> Vec<Option<ConceptId>> {
    chain_roles.iter().map(|&role| unbind_role(thought, role, dim, candidates)).collect()
}

// ─── Legacy HDCPatternMemory ─────────────────────────────────

pub struct HDCPatternMemory {
    pub dim: usize,
    pub max_patterns: usize,
    pub raw_patterns: Vec<Vec<ConceptId>>,
    pub hdv_patterns: Vec<Vec<f32>>,
    pub relations: Vec<(ConceptId, String, ConceptId)>, // (subject, relation, object)
}

impl HDCPatternMemory {
    pub fn new(max_patterns: usize) -> Self {
        HDCPatternMemory {
            dim: HDC_DIM,
            raw_patterns: Vec::with_capacity(max_patterns),
            hdv_patterns: Vec::with_capacity(max_patterns),
            max_patterns,
            relations: Vec::new(),
        }
    }

    pub fn store(&mut self, chain: &[ConceptId]) {
        if chain.len() < 2 { return; }
        let pattern_hdv = self.encode_chain(chain);
        self.raw_patterns.push(chain.to_vec());
        self.hdv_patterns.push(pattern_hdv);
        if self.raw_patterns.len() > self.max_patterns {
            self.raw_patterns.remove(0);
            self.hdv_patterns.remove(0);
        }
    }

    fn encode_chain(&self, chain: &[ConceptId]) -> Vec<f32> {
        let mut acc = vec![0.0; self.dim];
        for (pos, &id) in chain.iter().enumerate() {
            let phv = position_hv(pos, self.dim);
            let chv = bipolar_from_hash(id, self.dim);
            let bound = bind(&phv, &chv);
            bundle_into(&mut acc, &bound);
        }
        acc
    }

    pub fn complete(&self, query: &[ConceptId], max_len: usize) -> Option<Vec<ConceptId>> {
        if self.raw_patterns.is_empty() || query.is_empty() {
            return None;
        }

        let query_hdv = self.encode_chain(query);

        let mut best_idx = None;
        let mut best_sim = -1.0f32;

        for (i, pattern_hdv) in self.hdv_patterns.iter().enumerate() {
            if self.raw_patterns[i].len() <= query.len() { continue; }
            let sim = cosine_similarity(&query_hdv, pattern_hdv);
            if sim > best_sim {
                best_sim = sim;
                best_idx = Some(i);
            }
        }

        let idx = best_idx?;
        if best_sim < 0.1 { return None; }

        let pattern = &self.raw_patterns[idx];
        let completion: Vec<ConceptId> = pattern.iter()
            .skip(query.len())
            .take(max_len)
            .copied()
            .collect();

        if completion.is_empty() { None } else { Some(completion) }
    }

    pub fn len(&self) -> usize {
        self.raw_patterns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.raw_patterns.is_empty()
    }

    pub fn store_relation(&mut self, subj: ConceptId, rel_name: &str, obj: ConceptId) {
        self.relations.push((subj, rel_name.to_string(), obj));
    }

    pub fn query_relations(&self, subj: ConceptId, rel_name: &str) -> Vec<ConceptId> {
        self.relations.iter()
            .filter(|&(s, r, _)| *s == subj && r == rel_name)
            .map(|&(_, _, o)| o)
            .collect()
    }

    pub fn get_all_relations(&self) -> Vec<(ConceptId, String, ConceptId)> {
        self.relations.clone()
    }
}

// ─── Relation Vectors ─────────────────────────────────────────
// Semantic Pointer Architecture: Encode/decode relations between concepts

/// Predefined relation types with unique hypervectors
pub struct RelationVector {
    pub name: &'static str,
    pub hv: Vec<f32>,
}

pub const RELATIONS: &[RelationVector] = &[
    RelationVector {
        name: "is_part_of",
        hv: Vec::new(), // Will be initialized at runtime
    },
    RelationVector {
        name: "causes",
        hv: Vec::new(),
    },
    RelationVector {
        name: "is_a",
        hv: Vec::new(),
    },
    RelationVector {
        name: "has_property",
        hv: Vec::new(),
    },
    RelationVector {
        name: "leads_to",
        hv: Vec::new(),
    },
];

/// Initialize all relation hypervectors
pub fn init_relations(dim: usize) -> Vec<RelationVector> {
    RELATIONS.iter().map(|r| RelationVector {
        name: r.name,
        hv: role_hv(r.name, dim),
    }).collect()
}

/// Store a relation fact in HDC memory
pub fn encode_relation(subj: ConceptId, rel_name: &str, obj: ConceptId, dim: usize) -> Vec<f32> {
    let relations = init_relations(dim);
    let relation = relations.iter().find(|r| r.name == rel_name).unwrap_or(&relations[0]);
    
    let subj_hv = bipolar_from_hash(subj, dim);
    let obj_hv = bipolar_from_hash(obj, dim);
    
    // Bind subject with relation, then bind with object
    let bound1 = bind(&relation.hv, &subj_hv);
    let result = bind(&bound1, &obj_hv);
    result
}

/// Query relations from HDC memory
pub fn query_relations_from_memory(memory: &HDCPatternMemory, subj: ConceptId, rel_name: &str) -> Vec<ConceptId> {
    memory.query_relations(subj, rel_name)
}

/// Decode a relation vector to find the third element
/// Returns (subject, relation, object) where any can be None
pub fn decode_relation(
    thought: &[f32], 
    rel_name: &str, 
    dim: usize, 
    candidates: &[ConceptId]
) -> (Option<ConceptId>, Option<ConceptId>, Option<ConceptId>) {
    let relations = init_relations(dim);
    let relation = relations.iter().find(|r| r.name == rel_name).unwrap_or(&relations[0]);
    
    // Unbind with relation to get (subj ⊗ obj)
    let subj_obj = unbind(thought, &relation.hv);
    
    // Try to decode subject and object from candidates
    let mut best_subj = None;
    let mut best_obj = None;
    let mut best_subj_sim = -1.0;
    let mut best_obj_sim = -1.0;
    
    for &cid in candidates {
        let chv = bipolar_from_hash(cid, dim);
        
        // Try as subject: unbind thought with relation, then unbind with candidate to get object
        let decoded_obj = unbind(&subj_obj, &chv);
        let obj_sim = cosine_similarity(&decoded_obj, &relation.hv);
        
        // Try as object: bind candidate with relation, then see if it matches subj_obj
        let bound = bind(&chv, &relation.hv);
        let obj_sim2 = cosine_similarity(&bound, &subj_obj);
        
        if obj_sim > best_obj_sim {
            best_obj_sim = obj_sim;
            best_obj = Some(cid);
        }
        
        if obj_sim2 > best_subj_sim {
            best_subj_sim = obj_sim2;
            best_subj = Some(cid);
        }
    }
    
    (best_subj, Some(relation.name.parse().unwrap()), best_obj)
}
