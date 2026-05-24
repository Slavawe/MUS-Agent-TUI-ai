pub use crate::graph::Modality;
use crate::cuda_bridge::{CudaGraph, CudaSystemState};
use crate::hdc::{self, HDCPatternMemory, ThoughtRole};
use rand::Rng;
use rand::SeedableRng;
use std::collections::HashMap;

pub type ConceptId = u64;

pub fn concept_hash(s: &str) -> ConceptId {
    let mut h: u64 = 14695981039346656037;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

#[derive(Debug, Clone)]
pub struct SystemState {
    pub dopamin: f32,
    pub adrenaline: f32,
    pub energy: f32,
    pub energy_decay: f32,
    pub panic_threshold: f32,
    pub panic_active: bool,
    pub weight_decay: f32,
    pub weight_prune: f32,
    pub weight_boost: f32,
}

impl SystemState {
    pub fn new() -> Self {
        SystemState {
            dopamin: 0.3,
            adrenaline: 0.15,
            energy: 1.0,
            energy_decay: 0.85,
            panic_threshold: 0.5,
            panic_active: false,
            weight_decay: 0.995,
            weight_prune: 0.05,
            weight_boost: 0.15,
        }
    }

    pub fn update_by_metrics(&mut self, coherence: f32, saturation: f32) {
        self.dopamin = 0.1 + (coherence * 0.4).min(2.0);
        self.adrenaline = (saturation * 0.4).min(0.8);
        self.energy = (1.0 - saturation * 0.2).max(0.2);

        if saturation > self.panic_threshold && !self.panic_active {
            self.panic_active = true;
        } else if saturation < self.panic_threshold * 0.4 {
            self.panic_active = false;
        }

        // STDP: higher saturation = faster decay to prune aggressively
        self.weight_decay = 0.995 - saturation * 0.01;
        if self.weight_decay < 0.98 { self.weight_decay = 0.98; }
        self.weight_prune = 0.05 + saturation * 0.05;
    }

    pub fn to_cuda(&self) -> CudaSystemState {
        CudaSystemState {
            dopamin: self.dopamin,
            adrenaline: self.adrenaline,
            energy: self.energy,
            energy_decay: self.energy_decay,
            panic_threshold: self.panic_threshold,
            weight_decay: self.weight_decay,
            weight_prune: self.weight_prune,
            weight_boost: self.weight_boost,
            use_chemistry: 1,
        }
    }
}

impl Default for SystemState {
    fn default() -> Self {
        Self::new()
    }
}

static CONCEPT_NAMES: &[&str] = &[
    "ракета", "двигатель", "топливо", "орбита", "спутник",
    "космос", "полёт", "сопло", "тяга", "шаблон",
    "контейнер", "алгоритм", "итератор", "указатель", "ссылка",
    "нейросеть", "трансформер", "внимание", "слой", "градиент",
    "токен", "эмбеддинг", "память", "символ", "концепт",
    "аналогия", "абдукция", "индукция", "дедукция", "метафора",
    "int", "float", "double", "char", "void",
    "string", "vector", "map", "set", "option",
    "результат", "ошибка", "поток", "данные", "процесс",
    "функция", "модуль", "пакет", "зависимость", "интерфейс",
    "реализация", "абстракция", "композиция", "агрегация", "ассоциация",
    "когнитивная_карта", "сознание", "категория", "ценность", "цель",
];

pub struct Thinker {
    pub state: SystemState,
    pub names: Vec<String>,
    pub labels: HashMap<ConceptId, String>,
    pub associations: HashMap<ConceptId, Vec<ConceptId>>,
    pub hdc_memory: HDCPatternMemory,
}

impl Thinker {
    pub fn new() -> Self {
        let names: Vec<String> = CONCEPT_NAMES.iter().map(|s| s.to_string()).collect();
        Thinker {
            state: SystemState::new(),
            names,
            labels: HashMap::new(),
            associations: HashMap::new(),
            hdc_memory: HDCPatternMemory::new(100),
        }
    }

    pub fn add(&mut self, id: ConceptId, label: &str, _modality: Modality) {
        self.labels.entry(id).or_insert_with(|| label.to_string());
        self.associations.entry(id).or_default();
    }

    pub fn label(&self, id: ConceptId) -> &str {
        self.labels.get(&id).map(|s| s.as_str()).unwrap_or("?")
    }

    pub fn concept_names(&self) -> Vec<String> {
        self.labels.values().cloned().collect()
    }

    pub fn set_assoc(&mut self, a: ConceptId, b: ConceptId) {
        self.associations.entry(a).or_default().push(b);
        self.associations.entry(b).or_default().push(a);
    }

    pub fn bootstrap_graph(&self, g: &CudaGraph) {
        let base: u64 = 1000;
        for (i, name) in self.names.iter().enumerate() {
            g.add_node(base + i as u64, name, 0);
        }
        for i in 0..self.names.len().saturating_sub(1) {
            g.link(base + i as u64, base + i as u64 + 1);
        }
    }

    pub fn think(&self, g: &CudaGraph, seed_id: ConceptId, max_depth: i32) -> Vec<String> {
        let cuda_state = self.state.to_cuda();
        let count = g.activate_chem(seed_id, max_depth, &cuda_state);
        if count <= 0 {
            return vec!["(nothing activated)".to_string()];
        }
        let top = g.get_top_activations(16);
        let mut result = Vec::new();
        for (id, _act) in &top {
            if let Some(label) = self.labels.get(id) {
                result.push(label.clone());
            } else if *id >= 1000 {
                let local_idx = (*id - 1000) as usize;
                if local_idx < self.names.len() {
                    result.push(self.names[local_idx].clone());
                }
            }
        }
        result
    }

    pub fn concept_id(&self, idx: usize) -> ConceptId {
        1000 + idx as u64
    }

    // HDC pattern storage
    pub fn store_pattern(&mut self, chain: &[ConceptId]) {
        self.hdc_memory.store(chain);
    }

    // HDC attractor: find the closest stored pattern by hypervector similarity
    fn hdc_complete(&self, query: &[ConceptId], max_len: usize) -> Option<Vec<ConceptId>> {
        self.hdc_memory.complete(query, max_len)
    }

    pub fn generate_response(&mut self, input: &str, g: &CudaGraph) -> String {
        let words: Vec<&str> = input.split_whitespace()
            .filter(|w| !w.is_empty())
            .collect();
        if words.is_empty() {
            return "Скажи что-нибудь.".to_string();
        }

        let cuda_state = self.state.to_cuda();
        let mut rng = rand::rngs::StdRng::seed_from_u64(input.len() as u64);

        // ── Detect intent ──────────────────────────────────────
        let lower = input.to_lowercase();
        let is_greeting = lower.contains("привет") || lower.contains("здравств") || lower.contains("hello") || lower.contains("hi");
        let is_what_is = lower.contains("что такое") || lower.contains("что значит") || lower.contains("кто такой") || lower.contains("what is");
        let is_tell = lower.contains("расскажи") || lower.contains("объясни") || lower.contains("опиши") || lower.contains("tell") || lower.contains("explain");
        let is_compare = lower.contains("сравни") || lower.contains("чем отлич") || lower.contains("что общего") || lower.contains("compare") || lower.contains("vs");
        let is_why = lower.contains("почему") || lower.contains("зачем") || lower.contains("why") || lower.contains("как");
        let is_complex = words.len() > 4 || (is_tell && words.len() > 2) || lower.contains("всё") || lower.contains("everything");

        if is_greeting && words.len() <= 3 {
            let greetings = [
                "Привет! Я MUS — ассоциативная нейросеть. Спрашивай что угодно.",
                "Здравствуй. Я связана с понятиями — спроси меня о чём-нибудь.",
                "Приветствую. Моя память полна ассоциаций. Что тебя интересует?",
                "Hi! Я помню связи между концептами. Попробуй спросить 'что такое память'.",
            ];
            return greetings[rng.gen_range(0..greetings.len())].to_string();
        }

        // ── Find seed concepts ──────────────────────────────────
        let seed_ids: Vec<ConceptId> = words.iter()
            .filter_map(|w| {
                let clean = w.trim_matches(|c: char| c.is_ascii_punctuation());
                let h = concept_hash(clean);
                if self.labels.contains_key(&h) || g.node_count() > 0 {
                    Some(h)
                } else { None }
            })
            .collect();

        let (seed_id, seed_label) = if seed_ids.is_empty() {
            let ids = g.get_node_ids();
            if ids.is_empty() {
                return "В графе пока нет концептов. Добавь несколько (например, 'память код ассоциация').".to_string();
            }
            let id = ids[rng.gen_range(0..ids.len())];
            (id, self.label(id).to_string())
        } else {
            (seed_ids[0], {
                let lbl = self.label(seed_ids[0]);
                if lbl == "?" { words[0].to_string() } else { lbl.to_string() }
            })
        };

        // ── Activate graph with chemistry + K-WTA ────────────────
        g.activate_chem(seed_id, if is_complex { 10 } else { 6 }, &cuda_state);
        let top = g.get_top_activations(if is_complex { 24 } else { 12 });

        let chain: Vec<(u64, String)> = top.iter()
            .filter(|(id, _)| *id != seed_id && *id != 0)
            .map(|(id, _)| (*id, self.label(*id).to_string()))
            .collect();

        if chain.is_empty() {
            let fallbacks = [
                format!("{}. И больше ничего не вспоминается.", seed_label),
                format!("{}. Пока ассоциаций нет.", seed_label),
            ];
            return fallbacks[rng.gen_range(0..fallbacks.len())].to_string();
        }

        // ── Store activation chain as HDC pattern ──────────────
        let pattern_ids: Vec<ConceptId> = chain.iter().map(|(id, _)| *id).collect();
        self.store_pattern(&pattern_ids);
        let query_ids: Vec<ConceptId> = chain.iter().take(3).map(|(id, _)| *id).collect();
        let completion = self.hdc_complete(&query_ids, 5);

        // ── VSA role analysis on top concepts ──────────────────
        let top_ids: Vec<ConceptId> = top.iter().take(6).map(|(id, _)| *id).collect();
        let vsa_roles: Vec<&str> = if top_ids.len() >= 3 {
            // Encode a triple and try to decode roles
            let thought = hdc::encode_triple(top_ids[0], top_ids.get(1).copied().unwrap_or(top_ids[0]), top_ids.get(2).copied().unwrap_or(top_ids[0]), hdc::HDC_DIM);
            let (subj, act, obj) = hdc::decode_triple(&thought, hdc::HDC_DIM, &top_ids);
            vec![
                subj.map(|id| self.label(id)).unwrap_or("?"),
                act.map(|id| self.label(id)).unwrap_or("?"),
                obj.map(|id| self.label(id)).unwrap_or("?"),
            ]
        } else { Vec::new() };

        // ── Build response ────────────────────────────────────
        let display_chain: Vec<&str> = chain.iter().take(if is_complex { 8 } else { 4 }).map(|(_, name)| name.as_str()).collect();

        // ── Relation-aware lookup ──────────────────────────────
        let rel_facts = self.hdc_memory.get_all_relations();
        let rel_response = if !rel_facts.is_empty() {
            let mut rel_parts: Vec<String> = Vec::new();
            for &(sid, ref rel, oid) in &rel_facts {
                let s_label = self.label(sid);
                let o_label = self.label(oid);
                if s_label != "?" && o_label != "?" {
                    if seed_ids.iter().any(|&id| id == sid || id == oid) {
                        rel_parts.push(format!("{} {} {}", s_label, rel, o_label));
                    }
                }
            }
            if rel_parts.is_empty() { None } else { Some(rel_parts.join("; ")) }
        } else { None };

        let response = if let Some(ref rel_text) = rel_response {
            // Relation-grounded response
            if is_what_is || is_tell {
                let mut desc = String::new();
                desc.push_str(&seed_label);
                // Filter relations involving seed as subject
                let facts: Vec<String> = rel_facts.iter().filter_map(|&(sid, ref rel, oid)| {
                    if sid == seed_id {
                        let o_label = self.label(oid);
                        if o_label != "?" {
                            Some(format!("{} {}", rel, o_label))
                        } else { None }
                    } else { None }
                }).collect();
                if facts.is_empty() {
                    desc.push_str(&format!(": {}", rel_text));
                } else {
                    desc.push_str(&format!(" — это {}", facts.join(", ")));
                }
                desc.push('.');
                desc
            } else if is_why {
                // Check for causal relations
                let causes: Vec<String> = rel_facts.iter().filter_map(|&(sid, ref rel, oid)| {
                    if sid == seed_id && rel == "causes" {
                        Some(self.label(oid).to_string())
                    } else if oid == seed_id && rel == "causes" {
                        Some(self.label(sid).to_string())
                    } else { None }
                }).collect();
                if causes.is_empty() {
                    format!("{}: {}", seed_label, rel_text)
                } else {
                    format!("{} потому что {}", seed_label, causes.join(", "))
                }
            } else if is_compare && seed_ids.len() >= 2 {
                let id2 = seed_ids.get(1).copied().unwrap_or(seed_ids[0]);
                let label2 = self.label(id2);
                // Find common relations
                format!("{} и {} связаны: {}", seed_label, if label2 == "?" { "другой" } else { label2 }, rel_text)
            } else {
                format!("{}: {}", seed_label, rel_text)
            }
        } else if is_what_is || is_tell {
            // Description-style answer
            let mut desc = String::new();
            if !vsa_roles.is_empty() && vsa_roles[0] != "?" && vsa_roles[1] != "?" {
                desc.push_str(&format!("{} — это {} которая {}", seed_label, vsa_roles[0], vsa_roles[1]));
                if vsa_roles[2] != "?" {
                    desc.push_str(&format!(" {}", vsa_roles[2]));
                }
            } else {
                desc.push_str(&seed_label);
                let connectors = [
                    " связано с ", " ассоциируется с ", " напоминает о ",
                    " относится к ", " связано через ",
                ];
                desc.push_str(connectors[rng.gen_range(0..connectors.len())]);
                for (i, concept) in display_chain.iter().enumerate() {
                    if i == display_chain.len().saturating_sub(1) && display_chain.len() > 1 {
                        desc.push_str("и ");
                    } else if i > 0 {
                        desc.push_str(", ");
                    }
                    desc.push_str(concept);
                }
            }
            // Append HDC completion if found
            if let Some(comp) = completion {
                if !comp.is_empty() {
                    let comp_names: Vec<String> = comp.iter().take(3).map(|id| {
                        let lbl = self.label(*id);
                        if lbl == "?" { format!("#{}", id) } else { lbl.to_string() }
                    }).collect();
                    desc.push_str(&format!(". Похожий паттерн: {}", comp_names.join(", ")));
                }
            }
            desc.push('.');
            desc
        } else if is_why {
            // Causal-style answer
            let mut expl = String::new();
            expl.push_str(&seed_label);
            let causes = [
                " возникает из-за ", " происходит от ", " порождает ",
                " ведёт к ", " является причиной ",
            ];
            expl.push_str(causes[rng.gen_range(0..causes.len())]);
            if !display_chain.is_empty() {
                expl.push_str(display_chain[0]);
                if display_chain.len() > 1 {
                    expl.push_str(&format!(" через {}", display_chain[1]));
                }
            }
            expl.push('.');
            expl
        } else if is_compare && seed_ids.len() >= 2 {
            // Comparison-style answer
            let id2 = seed_ids.get(1).copied().unwrap_or(seed_ids[0]);
            let label2 = self.label(id2);
            let mut comp = String::new();
            comp.push_str(&seed_label);
            comp.push_str(" и ");
            comp.push_str(if label2 == "?" { "другой" } else { label2 });
            let comp_verbs = [
                " связаны через ", " объединяет ", " различаются в ",
                " имеют общее — ", " пересекаются в ",
            ];
            comp.push_str(comp_verbs[rng.gen_range(0..comp_verbs.len())]);
            if display_chain.len() >= 2 {
                comp.push_str(display_chain[0]);
                comp.push_str(" и ");
                comp.push_str(display_chain[1]);
            } else if !display_chain.is_empty() {
                comp.push_str(display_chain[0]);
            } else {
                comp.push_str("ассоциативную память");
            }
            comp.push('.');
            comp
        } else if is_complex {
            // Complex/deep answer
            let mut deep = String::new();
            let prefixes = [
                format!("{} образует сеть ассоциаций: ", seed_label),
                format!("В контексте {} активируются: ", seed_label),
                format!("{} встраивается в систему: ", seed_label),
            ];
            deep.push_str(&prefixes[rng.gen_range(0..prefixes.len())]);
            for (i, concept) in display_chain.iter().enumerate() {
                if i > 0 { deep.push_str(", "); }
                deep.push_str(concept);
            }
            if let Some(comp) = completion {
                if !comp.is_empty() {
                    let comp_names: Vec<String> = comp.iter().take(3).map(|id| {
                        let lbl = self.label(*id);
                        if lbl == "?" { format!("#{}", id) } else { lbl.to_string() }
                    }).collect();
                    deep.push_str(&format!(". HDC предсказывает: {}", comp_names.join(" → ")));
                }
            }
            deep.push('.');
            deep
        } else {
            // Default association answer
            let mut resp = String::new();
            resp.push_str(&seed_label);
            let connectors = [
                " — это ", " напоминает ", " связано с ",
                " вызывает ", " отсылает к ",
            ];
            resp.push_str(connectors[rng.gen_range(0..connectors.len())]);
            for (i, concept) in display_chain.iter().enumerate() {
                if i == display_chain.len().saturating_sub(1) && display_chain.len() > 1 {
                    resp.push_str("и ");
                } else if i > 0 {
                    resp.push_str(", ");
                }
                resp.push_str(concept);
            }
            resp.push('.');
            resp
        };

        // Update neurochemistry based on response complexity
        if is_complex || is_why {
            self.state.dopamin *= 0.98;
            self.state.adrenaline += 0.01;
        } else {
            self.state.dopamin += 0.005;
            self.state.adrenaline *= 0.995;
        }

        response
    }
}

// ─── FSM Grammar Templates ──────────────────────────────────

#[derive(Debug, Clone)]
pub struct SentenceSlot {
    pub role: ThoughtRole,
    pub domain_hint: Option<String>,
    pub optional: bool,
}

#[derive(Debug, Clone)]
pub struct GrammarTemplate {
    pub name: &'static str,
    pub slots: Vec<SentenceSlot>,
    pub word_order: Vec<usize>,
    pub connectors: Vec<&'static str>,
}

pub fn templates() -> &'static [GrammarTemplate] {
    use std::sync::LazyLock;
    static T: LazyLock<Vec<GrammarTemplate>> = LazyLock::new(|| {
        vec![
            GrammarTemplate {
                name: "statement",
                slots: vec![
                    SentenceSlot { role: ThoughtRole::Subject, domain_hint: Some("entity".into()), optional: false },
                    SentenceSlot { role: ThoughtRole::Action, domain_hint: Some("action".into()), optional: false },
                    SentenceSlot { role: ThoughtRole::Object, domain_hint: Some("entity".into()), optional: true },
                ],
                word_order: vec![0, 1, 2],
                connectors: vec![" ", " ", ""],
            },
            GrammarTemplate {
                name: "question",
                slots: vec![
                    SentenceSlot { role: ThoughtRole::Modal, domain_hint: Some("modal".into()), optional: false },
                    SentenceSlot { role: ThoughtRole::Subject, domain_hint: Some("entity".into()), optional: false },
                    SentenceSlot { role: ThoughtRole::Action, domain_hint: Some("action".into()), optional: false },
                    SentenceSlot { role: ThoughtRole::Object, domain_hint: Some("entity".into()), optional: true },
                ],
                word_order: vec![0, 1, 2, 3],
                connectors: vec![" ", " ", " ", "?"],
            },
            GrammarTemplate {
                name: "attribute",
                slots: vec![
                    SentenceSlot { role: ThoughtRole::Subject, domain_hint: Some("entity".into()), optional: false },
                    SentenceSlot { role: ThoughtRole::Attribute, domain_hint: None, optional: false },
                    SentenceSlot { role: ThoughtRole::Object, domain_hint: Some("entity".into()), optional: true },
                ],
                word_order: vec![0, 1, 2],
                connectors: vec![" ", " ", ""],
            },
        ]
    });
    &T
}

impl Thinker {
    pub fn generate_sentence(&self, template_idx: usize, seed: u64, graph: &CudaGraph) -> String {
        let tpls = templates();
        if template_idx >= tpls.len() {
            return format!("[unknown template {}]", template_idx);
        }
        let tmpl = &tpls[template_idx];

        let cuda_state = self.state.to_cuda();
        let count = graph.activate_chem(seed, 5, &cuda_state);
        if count <= 0 { return format!("[no activation for seed]"); }

        let top = graph.get_top_activations(16);

        // Map roles from BFS results using VSA
        let role_assignments: Vec<Option<ConceptId>> = tmpl.slots.iter()
            .map(|slot| {
                let candidates: Vec<ConceptId> = top.iter()
                    .filter(|(id, _)| {
                        if let Some(ref hint) = slot.domain_hint {
                            let label = self.label(*id);
                            label.contains(hint) || hint.contains(label)
                        } else { true }
                    })
                    .map(|(id, _)| *id)
                    .collect();
                if candidates.is_empty() {
                    top.iter().take(3).map(|(id, _)| *id).next()
                } else {
                    candidates.into_iter().next()
                }
            })
            .collect();

        let mut words: Vec<String> = Vec::new();
        for &slot_idx in &tmpl.word_order {
            if let Some(cid) = role_assignments[slot_idx] {
                let label = self.label(cid).to_string();
                let word = if slot_idx == 0 {
                    let mut c = label.chars();
                    c.next().map(|f| f.to_uppercase().to_string() + c.as_str()).unwrap_or(label)
                } else { label };
                words.push(word);
            }
        }

        if words.is_empty() { return format!("[empty sentence]"); }

        let mut sentence = String::new();
        for (i, word) in words.iter().enumerate() {
            if i > 0 {
                let conn = tmpl.connectors.get(i - 1).copied().unwrap_or(" ");
                sentence.push_str(conn);
            }
            sentence.push_str(word);
        }
        sentence.push('.');
        sentence
    }

    pub fn generate_thought(&self, seed: u64, graph: &CudaGraph) -> String {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let tmpl_idx = rng.gen_range(0..templates().len());
        self.generate_sentence(tmpl_idx, seed, graph)
    }

    // ─── Relation Vector Methods ───────────────────────────────
    pub fn store_relation(&mut self, subj: ConceptId, rel_name: &str, obj: ConceptId) {
        self.hdc_memory.store_relation(subj, rel_name, obj);
    }

    pub fn get_relations(&self, subj: ConceptId, rel_name: &str) -> Vec<ConceptId> {
        self.hdc_memory.query_relations(subj, rel_name)
    }

    pub fn infer_relations(&mut self, seed: ConceptId, graph: &CudaGraph) -> Vec<ConceptId> {
        let ids = graph.get_node_ids();
        let mut result = Vec::new();
        
        // Try to find relations from seed
        if let Some(rels) = self.get_relations(seed, "is_part_of").first() {
            result.push(*rels);
        }
        if let Some(rels) = self.get_relations(seed, "causes").first() {
            result.push(*rels);
        }
        if let Some(rels) = self.get_relations(seed, "is_a").first() {
            result.push(*rels);
        }
        
        result
    }
}

impl Default for Thinker {
    fn default() -> Self {
        Self::new()
    }
}
