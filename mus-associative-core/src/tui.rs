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

pub struct App {
    graph: CudaGraph,
    thinker: thinker::Thinker,
    thoughts: Vec<String>,
    step: usize,
    max_nodes: usize,
    slots_per_node: i32,
    input: InputField,
    status: String,
    blackboard: Blackboard,
    use_fsm: bool,
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

        App {
            graph,
            thinker,
            thoughts,
            step: 0,
            max_nodes: max_nodes as usize,
            slots_per_node: slots,
            input: InputField::new(),
            status: "Ready.".to_string(),
            blackboard: Blackboard::new(500),
            use_fsm: false,
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
                            KeyCode::Char('?') => {
                                self.thoughts.push("s=step t=think r=reset g=grow Enter=chat".to_string());
                                self.thoughts.push("p=Predictive Coding G=GSOM grow :rel/:relq/:relex".to_string());
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
                                    // Generate AI response: FSM templates or classic
                                    self.update_chem();
                                    let response = if self.use_fsm {
                                        self.thinker.generate_thought(seed_id, &self.graph)
                                    } else {
                                        self.thinker.generate_response(&t, &self.graph)
                                    };
                                    self.thoughts.push(format!("AI: {}", response));
                                    // Post Uran response to Blackboard
                                    self.blackboard.post(&response, EntryType::Fact, Source::Uran, Some(seed_id), None);
                                    // STDP: Decay after response generation
                                    self.graph.stdp_decay(0.92);
                                    if self.thoughts.len() > 100 {
                                        self.thoughts.drain(0..self.thoughts.len() - 80);
                                    }
                                    self.status = format!("AI: {}", &response[..response.char_indices().take(60).last().map(|(i, _)| i).unwrap_or(response.len())]);
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
        let text = format!(
            " [s]tep [t]hink [g]row [r]eset [?]help [q]uit | :rel/:relq  |  ask AI: {}█",
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
