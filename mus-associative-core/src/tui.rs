use crate::blackboard::{Blackboard, BlackboardEntry, EntryType, Source};
use crate::cuda_bridge::CudaGraph;
use crate::thinker::{self, ConceptId, Modality};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use rand::Rng;
use rand::SeedableRng;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, List, ListItem, Paragraph, Gauge},
    Frame,
};
use std::time::{Duration, Instant};

use thinker::concept_hash;

struct InputField {
    text: String,
    cursor: usize,
}

impl InputField {
    fn new() -> Self { InputField { text: String::new(), cursor: 0 } }
    fn insert(&mut self, c: char) {
        let byte_pos = self.text.char_indices().nth(self.cursor).map(|(i, _)| i).unwrap_or(self.text.len());
        self.text.insert(byte_pos, c);
        self.cursor += 1;
    }
    fn backspace(&mut self) {
        if self.cursor > 0 {
            let byte_pos = self.text.char_indices().nth(self.cursor - 1).map(|(i, _)| i).unwrap_or(0);
            self.text.remove(byte_pos);
            self.cursor -= 1;
        }
    }
    fn clear(&mut self) { self.text.clear(); self.cursor = 0; }
}

fn bar(filled: f32, total: usize) -> String {
    let n = (filled * total as f32).round() as usize;
    let n = n.min(total);
    "█".repeat(n) + &"░".repeat(total.saturating_sub(n))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    Text,
    Coder,
}

pub struct App {
    graph: CudaGraph,
    thinker: thinker::Thinker,
    coder: Option<crate::coder::Coder>,
    mode: Mode,
    thoughts: Vec<String>,
    step: usize,
    max_nodes: usize,
    slots_per_node: i32,
    input: InputField,
    status: String,
    blackboard: Blackboard,
    use_fsm: bool,
    auto_exec: bool,
    use_wasm: bool,
    vsa_memory: crate::hdc::HDCPatternMemory,
    last_vsa_thought: Vec<f32>,
    last_vsa_roles: Vec<(String, crate::hdc::ThoughtRole)>,
}

impl App {
    pub fn new(words: &[(&str, u32)], slots: i32, max_nodes: i32) -> Self {
        let graph = CudaGraph::new(max_nodes, slots);
        let mut thinker = thinker::Thinker::new();

        for &(label, mod_val) in words {
            let id = concept_hash(label);
            graph.add_node(id, label, mod_val as i32);
            let m = match mod_val {
                0 => thinker::Modality::Text,
                1 => thinker::Modality::Vision,
                2 => thinker::Modality::Audio,
                _ => thinker::Modality::Composite,
            };
            thinker.add(id, label, m);
        }

        let mut thoughts = Vec::new();
        thoughts.push("MUS Associative Core — TUI Test Suite".to_string());
        thoughts.push("Keys: s=step t=think Enter=inject ?=help q=quit".to_string());

        let coder = match crate::coder::Coder::new() {
            Ok(c) => Some(c),
            Err(e) => {
                let msg = format!("Coder init error: {}. Coder mode disabled.", e);
                thoughts.push(msg);
                None
            }
        };

        Self::from_components(graph, thinker, coder, thoughts, max_nodes as usize, slots, false, true)
    }

    pub fn from_components(
        graph: CudaGraph,
        thinker: thinker::Thinker,
        coder: Option<crate::coder::Coder>,
        initial_thoughts: Vec<String>,
        max_nodes: usize,
        slots_per_node: i32,
        auto_exec: bool,
        use_wasm: bool,
    ) -> Self {
        App {
            graph,
            thinker,
            coder,
            mode: Mode::Text,
            thoughts: initial_thoughts,
            step: 0,
            max_nodes,
            slots_per_node,
            input: InputField::new(),
            status: "Ready.".to_string(),
            blackboard: Blackboard::new(500),
            use_fsm: false,
            auto_exec,
            use_wasm,
            vsa_memory: crate::hdc::HDCPatternMemory::new(200),
            last_vsa_thought: vec![],
            last_vsa_roles: vec![],
        }
    }

    fn update_chem(&mut self) {
        let coh = self.graph.coherence();
        let sat = self.graph.saturation();
        self.thinker.state.update_by_metrics(coh, sat);
    }

    fn inject(&mut self, label: &str, mod_val: i32) {
        if self.graph.node_count() >= self.max_nodes as i32 {
            self.status = format!("Max nodes ({}) reached", self.max_nodes);
            return;
        }
        let id = concept_hash(label);
        self.graph.add_node(id, label, mod_val);
        let m = match mod_val {
            0 => thinker::Modality::Text,
            1 => thinker::Modality::Vision,
            2 => thinker::Modality::Audio,
            _ => thinker::Modality::Composite,
        };
        self.thinker.add(id, label, m);
        self.status = format!("Injected: {}", label);
    }

    fn inject_text(&mut self, text: &str) {
        let words: Vec<&str> = text.split_whitespace()
            .filter(|w| !w.is_empty())
            .map(|w| w.trim_matches(|c: char| c.is_ascii_punctuation()))
            .filter(|w| !w.is_empty())
            .collect();
        if words.is_empty() { return; }

        let mut ids: Vec<ConceptId> = Vec::new();
        for &w in &words {
            let id = concept_hash(w);
            if self.graph.node_count() < self.max_nodes as i32 {
                self.graph.add_node(id, w, 0);
                self.thinker.add(id, w, thinker::Modality::Text);
            }
            ids.push(id);
        }

        for i in 0..ids.len().saturating_sub(1) {
            self.graph.link(ids[i], ids[i + 1]);
            self.thinker.set_assoc(ids[i], ids[i + 1]);
        }

        self.thoughts.push(format!("── text: {} ──", text));
        let mut chain = String::new();
        for &id in &ids {
            chain.push_str(self.thinker.label(id));
            chain.push_str(" → ");
        }
        chain.push_str("✓");
        self.thoughts.push(chain);
        if self.thoughts.len() > 100 {
            self.thoughts.drain(0..self.thoughts.len() - 80);
        }

        self.status = format!("Text injected: {} words", words.len());
    }

    fn step_train(&mut self) {
        let nc = self.graph.node_count() as usize;
        if nc < 2 { return; }
        let ids = self.graph.get_node_ids();
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.step as u64);

        let seed = ids[rng.gen_range(0..ids.len())];
        self.graph.reset_activations();

        let cuda_state = self.thinker.state.to_cuda();
        self.graph.activate_chem(seed, 3, &cuda_state);
        self.graph.top_k(8);
        // Predictive Coding during step training (STDP boost is fused into BFS kernel)
        self.graph.predictive_step(0.2);

        let mut active: Vec<ConceptId> = vec![seed];
        let extra = rng.gen_range(1..4).min(nc - 1);
        for _ in 0..extra {
            let id = ids[rng.gen_range(0..ids.len())];
            if !active.contains(&id) { active.push(id); }
        }
        self.graph.hebbian_learn(&active, self.thinker.state.dopamin, self.thinker.state.adrenaline);

        for i in 0..active.len() {
            for j in (i + 1)..active.len() {
                self.thinker.set_assoc(active[i], active[j]);
            }
        }

        self.step += 1;
        let s = self.graph.saturation();
        self.status = format!("Step {}, Sat: {:.1}%", self.step, s * 100.0);

        // Panic clear if saturation exceeds threshold
        let state = &self.thinker.state;
        if state.panic_active {
            let cleared = self.graph.panic_clear(state.adrenaline, state.panic_threshold);
            if cleared > 0 {
                self.status.push_str(&format!(" | Panic cleared {} slots", cleared));
            }
        }

        // GSOM: auto-grow if saturated and dopamine high
        if s > 0.90 && state.dopamin > 0.5 {
            let old_cap = self.graph.capacity();
            let new_cap = (old_cap * 2).min(500000);
            if new_cap > old_cap {
                self.graph.grow(new_cap);
                self.thoughts.push(format!("GSOM: capacity {} → {}", old_cap, new_cap));
            }
        }
    }

    fn think(&mut self) {
        let ids = self.graph.get_node_ids();
        if ids.is_empty() { return; }
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.step as u64);
        let seed = ids[rng.gen_range(0..ids.len())];

        self.update_chem();
        let cuda_state = self.thinker.state.to_cuda();
        self.graph.activate_chem(seed, 8, &cuda_state);
        self.graph.top_k(12);
        // STDP boost is fused into BFS kernel; only Predictive Coding here
        self.graph.predictive_step(0.15);
        let lines = self.thinker.think(&self.graph, seed, 8);
        // STDP: Decay after thinking
        self.graph.stdp_decay(0.95);
        self.thoughts.push(format!("── think #{} (weighted chem) ──", self.thoughts.len()));
        for l in &lines {
            self.thoughts.push(l.clone());
        }
        if self.thoughts.len() > 100 {
            self.thoughts.drain(0..self.thoughts.len() - 80);
        }
        self.status = format!("Thought (weighted): {}", lines.first().unwrap_or(&"...".to_string()));
    }

    fn auto_grow(&mut self) {
        if self.graph.node_count() < self.max_nodes as i32 {
            let label = format!("auto_{}", self.graph.node_count());
            let mods = [0, 1, 3];
            self.inject(&label, mods[self.graph.node_count() as usize % 3]);
        }
    }

    fn show_node_list(&self) -> Vec<(ConceptId, String, f32)> {
        let ids = self.graph.get_node_ids();
        let acts = self.graph.get_activations();
        let mut nodes: Vec<(ConceptId, f32)> = ids.iter().copied()
            .zip(acts.iter().copied()).collect();
        nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        nodes.iter().map(|&(id, a)| {
            let label = self.thinker.label(id).to_string();
            (id, label, a)
        }).collect()
    }

    pub fn run(&mut self) -> std::io::Result<()> {
        use crossterm::{
            terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
            execute,
        };
        enable_raw_mode()?;
        let mut stdout = std::io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let mut terminal = ratatui::Terminal::new(ratatui::backend::CrosstermBackend::new(stdout))?;

        let mut last_tick = Instant::now();
        let tick_rate = Duration::from_millis(100);

        loop {
            terminal.draw(|f| self.draw(f))?;

            let timeout = tick_rate.saturating_sub(last_tick.elapsed());
            if event::poll(timeout)? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        match key.code {
                            KeyCode::Char('q') | KeyCode::Esc => break,
                            KeyCode::Char('c') => {
                                if self.coder.is_some() {
                                    self.mode = if self.mode == Mode::Text { Mode::Coder } else { Mode::Text };
                                    self.status = if self.mode == Mode::Coder { "Coder mode".into() } else { "Text mode".into() };
                                    self.thoughts.push(format!("Switched to {:?} mode", self.mode));
                                } else {
                                    self.status = "Coder not available".into();
                                }
                            }
                            KeyCode::Char('?') => {
                                self.thoughts.push("s=step t=think r=reset g=grow Enter=chat".to_string());
                                self.thoughts.push("p=Predictive Coding G=GSOM +=reward -=penalty".to_string());
                                self.thoughts.push("l=Clippy L=Python lint S=StyleLSH".to_string());
                                self.thoughts.push("c=toggle Coder mode :rel/:relq/:relex".to_string());
                                self.thoughts.push(":vsa <sentence> encode VSA :vsa-subj/:vsa-act/:vsa-obj".to_string());
                                self.thoughts.push(":vsa-pat :vsa-sim <word> V=last VSA stats".to_string());
                                self.thoughts.push(":vsa-code <sentence> — generate code from VSA roles".to_string());
                                self.thoughts.push(":rel subj rel obj — store relation".to_string());
                                self.thoughts.push(":relq subj rel — query relation".to_string());
                                self.thoughts.push(":relex — load example relations".to_string());
                                self.thoughts.push("Type something and press Enter — AI responds".to_string());
                                self.thoughts.push("1-4=inject concept with modality b=FSM toggle B=board ?=help q=quit".to_string());
                            }
                            KeyCode::Char('s') => { self.auto_grow(); self.step_train(); self.update_chem(); }
                            KeyCode::Char('t') => self.think(),
                            KeyCode::Char('r') => {
                                self.graph.reset_activations();
                                self.status = "Activations reset".to_string();
                            }
                            KeyCode::Char('g') => self.auto_grow(),
                            KeyCode::Char('p') => {
                                self.graph.predictive_step(0.2);
                                self.status = "Predictive Coding step".to_string();
                            }
                            KeyCode::Char('+') | KeyCode::Char('=') => {
                                let ids = self.graph.get_top_activations(8);
                                let pattern: Vec<u64> = ids.iter().map(|(id, _)| *id).collect();
                                let boost = self.thinker.reward.reward(&pattern, self.thinker.state.dopamin);
                                self.thinker.state.dopamin = (self.thinker.state.dopamin + boost).min(2.5);
                                self.status = format!("Reward +{:.2} (dopamin={:.2})", boost, self.thinker.state.dopamin);
                                self.thoughts.push(format!("+ Reward: streak={}", self.thinker.reward.streak));
                            }
                            KeyCode::Char('_') | KeyCode::Char('-') => {
                                let ids = self.graph.get_top_activations(8);
                                let pattern: Vec<u64> = ids.iter().map(|(id, _)| *id).collect();
                                let penalty = self.thinker.reward.penalize(&pattern, self.thinker.state.dopamin);
                                self.thinker.state.dopamin = (self.thinker.state.dopamin + penalty).max(0.1);
                                self.status = format!("Penalty {:.2} (dopamin={:.2})", penalty, self.thinker.state.dopamin);
                                self.thoughts.push(format!("- Penalty: streak reset"));
                            }
                            KeyCode::Char('l') => {
                                let result = crate::linter::Linter::clippy(".");
                                let score = result.score();
                                self.thoughts.push(format!("── Clippy: {} errors, {} warnings (score={:.2})", result.errors, result.warnings, score));
                                if score > 0.8 {
                                    self.thinker.state.dopamin = (self.thinker.state.dopamin + 0.1).min(2.5);
                                    self.status = format!("Clean code! +dopamin ({:.2})", self.thinker.state.dopamin);
                                } else if score < 0.3 {
                                    self.thinker.state.dopamin = (self.thinker.state.dopamin - 0.15).max(0.1);
                                    self.status = format!("Lint errors! -dopamin ({:.2})", self.thinker.state.dopamin);
                                } else {
                                    self.status = format!("Lint score: {:.2} ({})", score, result.duration_ms);
                                }
                            }
                            KeyCode::Char('L') => {
                                // Python syntax check mode
                                let result = crate::linter::Linter::py_compile("/tmp/mus_gen.py");
                                let score = result.score();
                                self.thoughts.push(format!("── Python lint: {} errors (score={:.2})", result.errors, score));
                                if score > 0.8 {
                                    self.thinker.state.dopamin = (self.thinker.state.dopamin + 0.1).min(2.5);
                                    self.status = format!("Valid Python! +dopamin ({:.2})", self.thinker.state.dopamin);
                                } else {
                                    self.thinker.state.dopamin = (self.thinker.state.dopamin - 0.1).max(0.1);
                                    self.status = format!("Python errors ({:.2})", self.thinker.state.dopamin);
                                }
                            }
                            KeyCode::Char('S') => {
                                // LSH style analysis of current code
                                use std::io::Read;
                                let mut code = String::new();
                                let path = if std::path::Path::new("src/main.rs").exists() { "src/main.rs" } else { "." };
                                if let Ok(mut f) = std::fs::File::open(path) {
                                    f.read_to_string(&mut code).ok();
                                }
                                let lsh = crate::style_lsh::StyleLSH::new(64);
                                let fp = lsh.analyze_code(&code);
                                self.thoughts.push(format!("── LSH Style Fingerprint ({})", path));
                                self.thoughts.push(format!("  Bits: {} words, {} features", fp.bits.len(), fp.feature_count));
                                self.thoughts.push(format!("  Raw: {:016x?}", fp.bits));
                                self.status = format!("Style LSH: {} features", fp.feature_count);
                            }
                            KeyCode::Char('V') => {
                                if self.last_vsa_thought.is_empty() {
                                    self.status = "No VSA encoded yet".to_string();
                                } else {
                                    let dim = self.last_vsa_thought.len();
                                    let nonzero = self.last_vsa_thought.iter().filter(|&&x| x != 0.0).count();
                                    let nrg: f32 = self.last_vsa_thought.iter().map(|x| x * x).sum();
                                    self.thoughts.push("── Last VSA ──".to_string());
                                    self.thoughts.push(format!("  Dim: {}, nonzero: {}/{}", dim, nonzero, dim));
                                    self.thoughts.push(format!("  Energy: {:.2}", nrg));
                                    let roles_str: String = self.last_vsa_roles.iter()
                                        .map(|(w, r)| format!("{}:{}", r.name(), w))
                                        .collect::<Vec<_>>()
                                        .join(" ");
                                    self.thoughts.push(format!("  Roles: {}", roles_str));
                                    // Compare to stored patterns
                                    let mut best_sim = -1.0f32;
                                    let mut best_idx = 0;
                                    for (i, pat_hdv) in self.vsa_memory.hdv_patterns.iter().enumerate() {
                                        let sim = crate::hdc::cosine_similarity(&self.last_vsa_thought, pat_hdv);
                                        if sim > best_sim {
                                            best_sim = sim;
                                            best_idx = i;
                                        }
                                    }
                                    if best_sim > 0.1 {
                                        let best_pat = &self.vsa_memory.raw_patterns[best_idx];
                                        let names: Vec<String> = best_pat.iter().map(|id| self.thinker.label(*id).to_string()).collect();
                                        self.thoughts.push(format!("  Closest pattern [{}] {} sim={:.3}", best_idx, names.join(" "), best_sim));
                                    }
                                    self.status = format!("VSA: {}d {} roles", dim, self.last_vsa_roles.len());
                                }
                            }
                            KeyCode::Char('G') => {
                                let sat = self.graph.saturation();
                                let old_cap = self.graph.capacity();
                                let new_cap = (old_cap * 2).min(500000);
                                if new_cap > old_cap {
                                    self.graph.grow(new_cap);
                                    self.thoughts.push(format!("GSOM: capacity {} → {} (sat={:.1}%)", old_cap, new_cap, sat * 100.0));
                                } else {
                                    self.status = "At max capacity".to_string();
                                }
                            }
                            KeyCode::Char('b') => {
                                self.use_fsm = !self.use_fsm;
                                self.status = if self.use_fsm { "FSM mode ON".into() } else { "FSM mode OFF".into() };
                            }
                            KeyCode::Char('B') => {
                                self.thoughts.push("── Blackboard Entries ──".to_string());
                                for e in self.blackboard.read_all().iter().rev().take(16) {
                                    self.thoughts.push(format!("  [{:?}/{:?}] {}", e.source, e.entry_type, e.text));
                                }
                            }
                            KeyCode::Char('1') => { let t = self.input.text.clone(); self.inject(&t, 0); }
                            KeyCode::Char('2') => { let t = self.input.text.clone(); self.inject(&t, 1); }
                            KeyCode::Char('3') => { let t = self.input.text.clone(); self.inject(&t, 2); }
                            KeyCode::Char('4') => { let t = self.input.text.clone(); self.inject(&t, 3); }
                            KeyCode::Char(c) if !c.is_control() => self.input.insert(c),
                            KeyCode::Backspace => self.input.backspace(),
                            KeyCode::Enter => {
                                let t = self.input.text.clone();
                                if !t.is_empty() {
                                    // ── Commands ──────────────────────────────
                                    if t.starts_with(":rel ") {
                                        let parts: Vec<&str> = t[5..].split_whitespace().collect();
                                        if parts.len() >= 3 {
                                            let subj = concept_hash(parts[0]);
                                            let obj = concept_hash(parts[2]);
                                            let rel = parts[1];
                                            self.thinker.store_relation(subj, rel, obj);
                                            self.thoughts.push(format!("✓ Relation: {} → {} → {}", parts[0], rel, parts[2]));
                                            // Ensure both nodes exist in graph
                                            for &word in &[parts[0], parts[2]] {
                                                let h = concept_hash(word);
                                                if self.thinker.label(h) == "?" {
                                                    self.graph.add_node(h, word, 0);
                                                    self.thinker.add(h, word, Modality::Text);
                                                }
                                            }
                                        } else {
                                            self.thoughts.push("Usage: :rel subj rel obj".to_string());
                                        }
                                        self.input.clear();
                                        last_tick = Instant::now();
                                        continue;
                                    }
                                    if t.starts_with(":relq ") {
                                        let parts: Vec<&str> = t[6..].split_whitespace().collect();
                                        if parts.len() >= 2 {
                                            let subj = concept_hash(parts[0]);
                                            let rel = parts[1];
                                            let results = self.thinker.get_relations(subj, rel);
                                            self.thoughts.push(format!("── Relations: {} {} ──", parts[0], rel));
                                            if results.is_empty() {
                                                self.thoughts.push("  (none found)".to_string());
                                            } else {
                                                for &obj in &results {
                                                    self.thoughts.push(format!("  → {}", self.thinker.label(obj)));
                                                }
                                            }
                                        } else {
                                            self.thoughts.push("Usage: :relq subj rel".to_string());
                                        }
                                        self.input.clear();
                                        last_tick = Instant::now();
                                        continue;
                                    }
                                    if t.starts_with(":relex") {
                                        Self::load_example_relations(&mut self.thinker);
                                        self.thoughts.push("✓ Example relations loaded".to_string());
                                        self.input.clear();
                                        last_tick = Instant::now();
                                        continue;
                                    }
                                    // ── VSA commands ────────────────────────────
                                    if t.starts_with(":vsa ") {
                                        let words: Vec<&str> = t[5..].split_whitespace().collect();
                                        if words.is_empty() { self.input.clear(); continue; }
                                        let dim = crate::hdc::HDC_DIM;
                                        let mut acc = vec![0.0; dim];
                                        let mut roles = Vec::new();
                                        let role_order = [
                                            crate::hdc::ThoughtRole::Subject,
                                            crate::hdc::ThoughtRole::Action,
                                            crate::hdc::ThoughtRole::Object,
                                            crate::hdc::ThoughtRole::Attribute,
                                            crate::hdc::ThoughtRole::Modifier,
                                        ];
                                        for (i, &word) in words.iter().enumerate() {
                                            let id = concept_hash(word);
                                            let role = role_order[i.min(role_order.len() - 1)];
                                            let bound = crate::hdc::bind_role(id, role, dim);
                                            crate::hdc::bundle_into(&mut acc, &bound);
                                            roles.push((word.to_string(), role));
                                            // Ensure node exists
                                            if self.thinker.label(id) == "?" {
                                                self.graph.add_node(id, word, 0);
                                                self.thinker.add(id, word, crate::thinker::Modality::Text);
                                            }
                                        }
                                        self.last_vsa_thought = acc.clone();
                                        self.last_vsa_roles = roles;
                                        let chain_str: String = words.iter()
                                            .enumerate()
                                            .map(|(i, w)| format!("{}:{}", role_order[i.min(role_order.len()-1)].name(), w))
                                            .collect::<Vec<_>>()
                                            .join(" ");
                                        self.thoughts.push(format!("── VSA encode: {} ──", chain_str));
                                        self.thoughts.push(format!("  Dim: {}, bound: {} roles", dim, words.len()));
                                        self.status = format!("VSA: {} → {}d vector", words.join(" "), dim);
                                        // Store in pattern memory
                                        let ids: Vec<u64> = words.iter().map(|w| concept_hash(w)).collect();
                                        self.vsa_memory.store(&ids);
                                        self.input.clear();
                                        last_tick = Instant::now();
                                        continue;
                                    }
                                    if t == ":vsa-subj" || t == ":vsa-act" || t == ":vsa-obj" {
                                        if self.last_vsa_thought.is_empty() {
                                            self.thoughts.push("  No VSA thought encoded yet. Use :vsa first.".to_string());
                                            self.input.clear();
                                            last_tick = Instant::now();
                                            continue;
                                        }
                                        let r = match t.as_str() {
                                            ":vsa-subj" => crate::hdc::ThoughtRole::Subject,
                                            ":vsa-act" => crate::hdc::ThoughtRole::Action,
                                            ":vsa-obj" => crate::hdc::ThoughtRole::Object,
                                            _ => crate::hdc::ThoughtRole::Subject,
                                        };
                                        let dim = crate::hdc::HDC_DIM;
                                        let candidates: Vec<u64> = self.graph.get_node_ids();
                                        let decoded = crate::hdc::unbind_role(&self.last_vsa_thought, r, dim, &candidates);
                                        match decoded {
                                            Some(id) => self.thoughts.push(format!("  {} → {} (id={})", r.name(), self.thinker.label(id), id)),
                                            None => self.thoughts.push(format!("  {} → ? (no match >0.15)", r.name())),
                                        }
                                        self.status = format!("VSA unbind {}", r.name());
                                        self.input.clear();
                                        last_tick = Instant::now();
                                        continue;
                                    }
                                    if t == ":vsa-pat" {
                                        self.thoughts.push(format!("── VSA Patterns ({}) ──", self.vsa_memory.len()));
                                        for (i, pat) in self.vsa_memory.raw_patterns.iter().enumerate().take(16) {
                                            let names: Vec<String> = pat.iter().map(|id| self.thinker.label(*id).to_string()).collect();
                                            self.thoughts.push(format!("  {}: [{}]", i, names.join(", ")));
                                        }
                                        self.input.clear();
                                        last_tick = Instant::now();
                                        continue;
                                    }
                                    if t == ":vsa-sim" || t.starts_with(":vsa-sim ") {
                                        let target = if t == ":vsa-sim" { "" } else { t[9..].trim() };
                                        if target.is_empty() || self.last_vsa_thought.is_empty() {
                                            self.thoughts.push("Usage: :vsa-sim <word>".to_string());
                                            self.input.clear();
                                            last_tick = Instant::now();
                                            continue;
                                        }
                                        let id = concept_hash(target);
                                        let chv = crate::hdc::bipolar_from_hash(id, crate::hdc::HDC_DIM);
                                        let sim = crate::hdc::cosine_similarity(&self.last_vsa_thought, &chv);
                                        self.thoughts.push(format!("  sim({}) = {:.4}", target, sim));
                                        self.status = format!("VSA sim: {:.4}", sim);
                                        self.input.clear();
                                        last_tick = Instant::now();
                                        continue;
                                    }
                                    // ── VSA + Coder: generate code from VSA roles ──
                                    if t == ":vsa-code" || t.starts_with(":vsa-code ") {
                                        let sentence = if t == ":vsa-code" { "" } else { t[9..].trim() };
                                        if !sentence.is_empty() {
                                            let words: Vec<&str> = sentence.split_whitespace().collect();
                                            if !words.is_empty() {
                                                let dim = crate::hdc::HDC_DIM;
                                                let mut acc = vec![0.0; dim];
                                                let role_order = [
                                                    crate::hdc::ThoughtRole::Subject,
                                                    crate::hdc::ThoughtRole::Action,
                                                    crate::hdc::ThoughtRole::Object,
                                                ];
                                                for (i, &word) in words.iter().enumerate() {
                                                    let id = concept_hash(word);
                                                    let role = role_order[i.min(role_order.len() - 1)];
                                                    let bound = crate::hdc::bind_role(id, role, dim);
                                                    crate::hdc::bundle_into(&mut acc, &bound);
                                                    if self.thinker.label(id) == "?" {
                                                        self.graph.add_node(id, word, 0);
                                                        self.thinker.add(id, word, crate::thinker::Modality::Text);
                                                    }
                                                }
                                                self.last_vsa_thought = acc.clone();
                                                let ids: Vec<u64> = words.iter().map(|w| concept_hash(w)).collect();
                                                self.vsa_memory.store(&ids);
                                            }
                                        }
                                        if self.last_vsa_thought.is_empty() {
                                            self.thoughts.push("  No VSA thought. Usage: :vsa-code кот ловит мышь".to_string());
                                            self.input.clear();
                                            last_tick = Instant::now();
                                            continue;
                                        }
                                        if let Some(ref coder) = self.coder {
                                            let dim = crate::hdc::HDC_DIM;
                                            let candidates: Vec<u64> = self.graph.get_node_ids();
                                            let subj = crate::hdc::unbind_role(&self.last_vsa_thought, crate::hdc::ThoughtRole::Subject, dim, &candidates);
                                            let act = crate::hdc::unbind_role(&self.last_vsa_thought, crate::hdc::ThoughtRole::Action, dim, &candidates);
                                            let obj = crate::hdc::unbind_role(&self.last_vsa_thought, crate::hdc::ThoughtRole::Object, dim, &candidates);
                                            let subj_name = subj.map(|id| self.thinker.label(id).to_string()).unwrap_or_default();
                                            let act_name = act.map(|id| self.thinker.label(id).to_string()).unwrap_or_default();
                                            let obj_name = obj.map(|id| self.thinker.label(id).to_string()).unwrap_or_default();
                                            self.thoughts.push("── VSA + Coder ──".to_string());
                                            self.thoughts.push(format!("  {} ({}) {} ({}) {} ({})",
                                                subj_name, subj.map(|_| "subject").unwrap_or("?"),
                                                act_name, act.map(|_| "action").unwrap_or("?"),
                                                obj_name, obj.map(|_| "object").unwrap_or("?")
                                            ));
                                            if act.is_some() {
                                                match coder.vsa_to_python(&subj_name, &act_name, &obj_name) {
                                                    Ok(code) => {
                                                        self.thoughts.push("  Python:".to_string());
                                                        for line in code.lines().take(8) {
                                                            self.thoughts.push(format!("    {}", line));
                                                        }
                                                        self.blackboard.post(&code, crate::blackboard::EntryType::Fact, crate::blackboard::Source::Coder, act, None);
                                                    }
                                                    Err(e) => self.thoughts.push(format!("  Template error: {}", e)),
                                                }
                                                match coder.vsa_to_rust(&subj_name, &act_name, &obj_name) {
                                                    Ok(code) => {
                                                        self.thoughts.push("  Rust:".to_string());
                                                        for line in code.lines().take(6) {
                                                            self.thoughts.push(format!("    {}", line));
                                                        }
                                                    }
                                                    Err(e) => self.thoughts.push(format!("  Rust error: {}", e)),
                                                }
                                            } else {
                                                self.thoughts.push("  No action role found — cannot generate function".to_string());
                                            }
                                            self.status = format!("VSA code: {}({}, {})", act_name, subj_name, obj_name);
                                        } else {
                                            self.thoughts.push("  Coder not available".to_string());
                                        }
                                        self.input.clear();
                                        last_tick = Instant::now();
                                        continue;
                                    }
                                    // ── Normal chat ────────────────────────────
                                    self.thoughts.push(format!("You: {}", t));
                                    // Post user intent to Blackboard
                                    let seed_id = concept_hash(&t);
                                    self.blackboard.post(&t, EntryType::Intent, Source::User, Some(seed_id), None);
                                    // Inject into graph so it learns
                                    if t.contains(' ') {
                                        self.inject_text(&t);
                                    } else {
                                        self.inject(&t, 0);
                                    }
                                    // Activate graph and apply K-WTA top-k
                                    let cuda_state = self.thinker.state.to_cuda();
                                    self.graph.activate_chem(seed_id, 5, &cuda_state);
                                    self.graph.top_k(12);
                                    // STDP boost is fused into BFS; Predictive Coding only
                                    self.graph.predictive_step(0.2);

                                    // GSOM: auto-grow if saturated and dopamine high
                                    let sat = self.graph.saturation();
                                    if sat > 0.90 && self.thinker.state.dopamin > 0.5 {
                                        let old_cap = self.graph.capacity();
                                        let new_cap = (old_cap * 2).min(500000);
                                        if new_cap > old_cap {
                                            self.graph.grow(new_cap);
                                            self.thoughts.push(format!("🌱 GSOM: capacity {} → {}", old_cap, new_cap));
                                        }
                                    }
                                    // Generate response based on mode
                                    self.update_chem();
                                    if self.mode == Mode::Coder {
                                        if let Some(ref coder) = self.coder {
                                            let top = self.graph.get_top_activations(16);
                                            let chain: Vec<(u64, &str, f32)> = top.iter()
                                                .map(|(id, sc)| (*id, self.thinker.label(*id), *sc))
                                                .collect();
                                            self.thoughts.push("── Coder ──".to_string());
                                            // Show chain
                                            let chain_names: Vec<&str> = chain.iter().map(|(_, n, _)| *n).collect();
                                            self.thoughts.push(format!("  Chain: {}", chain_names.join(" → ")));
                                            // AST
                                            let ast_text = coder.chain_to_ast_text(&chain);
                                            self.thoughts.push(format!("  AST: {}", ast_text));
                                            // Generate Python with style match
                                            match coder.render_with_style(&chain, "python") {
                                                Ok((code, style_match)) => {
                                                    self.thoughts.push("  Python:".to_string());
                                                    for line in code.lines().take(8) {
                                                        self.thoughts.push(format!("    {}", line));
                                                    }
                                                    if let Some((ref style_file, ref dist)) = style_match {
                                                        self.thoughts.push(format!("  Style match: {} (dist={:.2})", style_file, dist));
                                                    }
                                                    self.status = "Coder: code generated".to_string();
                                                    self.blackboard.post(&code, EntryType::Fact, Source::Coder, Some(seed_id), None);
                                                    // Auto-execute generated code
                                                    if self.auto_exec {
                                                        let exec = crate::executor::Executor::new();
                                                        self.thoughts.push("  ─── Exec ───".to_string());
                                                        let result = if self.use_wasm {
                                                            exec.run_wasm("compute.wasm", &code)
                                                        } else {
                                                            exec.run_python(&code)
                                                        };
                                                        self.thoughts.push(format!("  Exit: {}, {}ms", result.exit_code, result.duration_ms));
                                                        if !result.stdout.is_empty() {
                                                            self.thoughts.push(format!("  Out: {}", result.stdout.trim()));
                                                        }
                                                        if result.exit_code == 0 {
                                                            let top_acts = self.graph.get_top_activations(8);
                                                            let pattern: Vec<u64> = top_acts.iter().map(|(id, _)| *id).collect();
                                                            let boost = self.thinker.reward.reward(&pattern, self.thinker.state.dopamin);
                                                            self.thinker.state.dopamin = (self.thinker.state.dopamin + boost).min(2.5);
                                                            self.thoughts.push(format!("  Reward +{:.2}", boost));
                                                        }
                                                    }
                                                }
                                                Err(e) => {
                                                    self.thoughts.push(format!("  Template error: {}", e));
                                                }
                                            }
                                        }
                                    } else {
                                        let response = if self.use_fsm {
                                            self.thinker.generate_thought(seed_id, &self.graph)
                                        } else {
                                            self.thinker.generate_response(&t, &self.graph)
                                        };
                                        self.thoughts.push(format!("AI: {}", response));
                                        // Post Uran response to Blackboard
                                        self.blackboard.post(&response, EntryType::Fact, Source::Uran, Some(seed_id), None);
                                    }
                                    // STDP: Decay after response generation
                                    self.graph.stdp_decay(0.92);
                                    if self.thoughts.len() > 100 {
                                        self.thoughts.drain(0..self.thoughts.len() - 80);
                                    }
                                    let status_text = if self.mode == Mode::Coder {
                                        "Coder: code generated ✓".to_string()
                                    } else {
                                        let r = if self.use_fsm {
                                            self.thinker.generate_thought(seed_id, &self.graph)
                                        } else {
                                            self.thinker.generate_response(&t, &self.graph)
                                        };
                                        format!("AI: {}", &r[..r.char_indices().take(60).last().map(|(i, _)| i).unwrap_or(r.len())])
                                    };
                                    self.status = status_text;
                                    self.input.clear();
                                }
                            }
                            _ => {}
                        }
                        last_tick = Instant::now();
                    }
                }
            }
        }

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        Ok(())
    }

    fn draw(&self, frame: &mut Frame) {
        let area = frame.size();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(0), Constraint::Length(3)])
            .split(area);

        let mid = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
            ])
            .split(chunks[1]);

        self.draw_header(frame, chunks[0]);
        self.draw_stats(frame, mid[0]);
        self.draw_chem(frame, mid[1]);
        self.draw_nodes(frame, mid[2]);
        self.draw_thoughts(frame, mid[3]);
        self.draw_footer(frame, chunks[2]);
    }

    fn draw_header(&self, frame: &mut Frame, area: Rect) {
        let coh = self.graph.coherence();
        let sat = self.graph.saturation();
        let s = &self.thinker.state;
        let text = format!(
            " MUS v0.3 | coh:{:.0}% sat:{:.0}% act:{} nodes:{}  |  D:{:.2} A:{:.2} E:{:.2}{}",
            coh * 100.0, sat * 100.0,
            self.graph.active_count(), self.graph.node_count(),
            s.dopamin, s.adrenaline, s.energy,
            if s.panic_active { " ⚠PANIC" } else { "" }
        );
        let p = Paragraph::new(Line::from(Span::styled(text, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))))
            .block(Block::bordered().title("MUS Associative Core — Neurochemical"));
        frame.render_widget(p, area);
    }

    fn draw_stats(&self, frame: &mut Frame, area: Rect) {
        let coh = self.graph.coherence();
        let sat = self.graph.saturation();
        let active = self.graph.active_count();
        let total = self.graph.node_count();

        let gauge_style = |val: f32| -> Style {
            if val > 0.7 { Style::default().fg(Color::Green) }
            else if val > 0.3 { Style::default().fg(Color::Yellow) }
            else { Style::default().fg(Color::Red) }
        };

        let items = vec![
            ListItem::new(Line::from(vec![
                Span::raw("Nodes:     "),
                Span::styled(format!("{} / {}", total, self.max_nodes), Style::default().fg(Color::Cyan)),
            ])),
            ListItem::new(Line::from(vec![
                Span::raw("Active:    "),
                Span::styled(format!("{}", active), Style::default().fg(Color::Green)),
            ])),
            ListItem::new(Line::from(vec![
                Span::raw("Slots/Node: "),
                Span::styled(format!("{}", self.slots_per_node), Style::default().fg(Color::Yellow)),
            ])),
            ListItem::new(""),
            ListItem::new(Line::from(vec![
                Span::raw("Coherence:"),
                Span::styled(format!(" {:.0}%", coh * 100.0), gauge_style(coh)),
            ])),
            ListItem::new(Line::from(Span::raw(format!("  [{}]", bar(coh, 20))))),
            ListItem::new(""),
            ListItem::new(Line::from(vec![
                Span::raw("Saturation:"),
                Span::styled(format!(" {:.0}%", sat * 100.0), gauge_style(sat.min(1.0))),
            ])),
            ListItem::new(Line::from(Span::raw(format!("  [{}]", bar(sat.min(1.0), 20))))),
            ListItem::new(""),
            ListItem::new(Line::from(vec![
                Span::raw("Status: "),
                Span::styled(&self.status, Style::default().fg(Color::White)),
            ])),
        ];

        let list = List::new(items).block(Block::bordered().title("Graph Stats"));
        frame.render_widget(list, area);
    }

    fn draw_chem(&self, frame: &mut Frame, area: Rect) {
        let s = &self.thinker.state;
        let max_dop = 2.0f32;
        let max_adr = 1.0f32;
        let max_eng = 1.0f32;

        let dop_pct = (s.dopamin / max_dop).min(1.0);
        let adr_pct = (s.adrenaline / max_adr).min(1.0);
        let eng_pct = (s.energy / max_eng).min(1.0);

        let gauge_col = |val: f32, high: f32| -> Color {
            if val > high * 0.8 { Color::Red }
            else if val > high * 0.5 { Color::Yellow }
            else { Color::Green }
        };

        let inner = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Min(0),
            ])
            .margin(1)
            .split(area);

        let gauge_style = |_pct: f32, color: Color| -> Style {
            Style::default().fg(color).bg(Color::Reset)
        };

        let dop_gauge = Gauge::default()
            .block(Block::default().title("Dopamin"))
            .gauge_style(gauge_style(dop_pct, gauge_col(s.dopamin, max_dop)))
            .percent((dop_pct * 100.0) as u16)
            .label(format!("{:.2}", s.dopamin));
        frame.render_widget(dop_gauge, inner[0]);

        let adr_gauge = Gauge::default()
            .block(Block::default().title("Adrenaline"))
            .gauge_style(gauge_style(adr_pct, gauge_col(s.adrenaline, max_adr)))
            .percent((adr_pct * 100.0) as u16)
            .label(format!("{:.2}", s.adrenaline));
        frame.render_widget(adr_gauge, inner[1]);

        let eng_gauge = Gauge::default()
            .block(Block::default().title("Energy"))
            .gauge_style(gauge_style(eng_pct, gauge_col(s.energy, max_eng)))
            .percent((eng_pct * 100.0) as u16)
            .label(format!("{:.2}", s.energy));
        frame.render_widget(eng_gauge, inner[2]);

        let decay_text = format!("Decay: {:.2}", s.energy_decay);
        let threshold_text = format!("Panic: {:.2}{}", s.panic_threshold, if s.panic_active { " ACTIVE" } else { "" });

        let info_items = vec![
            ListItem::new(Line::from(Span::styled(decay_text, Style::default().fg(Color::DarkGray)))),
            ListItem::new(Line::from(Span::styled(threshold_text,
                if s.panic_active { Style::default().fg(Color::Red).add_modifier(Modifier::BOLD) }
                else { Style::default().fg(Color::DarkGray) }
            ))),
        ];
        let info_list = List::new(info_items);
        frame.render_widget(info_list, inner[3]);
    }

    fn draw_nodes(&self, frame: &mut Frame, area: Rect) {
        let nodes = self.show_node_list();
        let items: Vec<ListItem> = nodes.iter().take(area.height as usize - 2).map(|(_id, label, act)| {
            let color = if *act > 0.5 { Color::Green } else { Color::DarkGray };
            let prefix = if *act > 0.5 { "●" } else { "○" };
            let bar_str = bar(*act, 8);
            ListItem::new(Line::from(vec![
                Span::styled(format!("{} {} ", prefix, label), Style::default().fg(color)),
                Span::styled(bar_str, Style::default().fg(color)),
                Span::raw(format!(" {:.1}", act)),
            ]))
        }).collect();

        let list = List::new(items)
            .block(Block::bordered().title(format!("Nodes ({})", nodes.len())));
        frame.render_widget(list, area);
    }

    fn draw_thoughts(&self, frame: &mut Frame, area: Rect) {
        let items: Vec<ListItem> = self.thoughts.iter().rev()
            .take(area.height as usize - 2)
            .map(|l| {
                let color = if l.starts_with("──") { Color::Cyan } else { Color::White };
                ListItem::new(Line::from(Span::styled(l.clone(), Style::default().fg(color))))
            })
            .collect();

        let list = List::new(items)
            .block(Block::bordered().title(format!("Thoughts ({})", self.thoughts.len())));
        frame.render_widget(list, area);
    }

    fn draw_footer(&self, frame: &mut Frame, area: Rect) {
        let mode_str = match self.mode {
            Mode::Text => "TEXT",
            Mode::Coder => "CODER",
        };
        let text = format!(
            " [s]tep [t]hink [g]row [r]eset [?]help [q]uit | [{}] c=toggle |  ask AI: {}█",
            mode_str,
            self.input.text,
        );
        let p = Paragraph::new(Line::from(Span::styled(text, Style::default().fg(Color::Gray))))
            .block(Block::bordered());
        frame.render_widget(p, area);
    }

    pub fn load_example_relations(thinker: &mut crate::thinker::Thinker) {
        use crate::thinker::concept_hash;
        let examples: &[(&str, &str, &str)] = &[
            ("thread", "causes", "block"),
            ("block", "causes", "grid"),
            ("grid", "causes", "kernel"),
            ("thread", "is_part_of", "block"),
            ("block", "is_part_of", "grid"),
            ("grid", "is_part_of", "kernel_launch"),
            ("GPU", "has_property", "parallel"),
            ("CUDA", "is_a", "API"),
            ("memory", "is_part_of", "GPU"),
            ("shared_memory", "is_part_of", "block"),
        ];
        for &(subj, rel, obj) in examples {
            thinker.store_relation(concept_hash(subj), rel, concept_hash(obj));
        }
    }
}
