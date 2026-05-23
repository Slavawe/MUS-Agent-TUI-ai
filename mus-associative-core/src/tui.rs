use crate::cuda_bridge::CudaGraph;
use crate::graph::ConceptId;
use crate::thinker;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use rand::Rng;
use rand::SeedableRng;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};
use std::time::{Duration, Instant};

pub fn concept_hash(s: &str) -> ConceptId {
    let mut h: u64 = 14695981039346656037;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

struct InputField {
    text: String,
    cursor: usize,
}

impl InputField {
    fn new() -> Self { InputField { text: String::new(), cursor: 0 } }
    fn insert(&mut self, c: char) { self.text.insert(self.cursor, c); self.cursor += 1; }
    fn backspace(&mut self) { if self.cursor > 0 { self.text.remove(self.cursor - 1); self.cursor -= 1; } }
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
}

impl App {
    pub fn new(words: &[(&str, u32)], slots: i32, max_nodes: i32) -> Self {
        let mut graph = CudaGraph::new(max_nodes, slots);
        let mut thinker = thinker::Thinker::new();

        for &(label, mod_val) in words {
            let id = concept_hash(label);
            graph.add_node(id, label, mod_val as i32);
            let m = match mod_val {
                0 => crate::graph::Modality::Text,
                1 => crate::graph::Modality::Vision,
                2 => crate::graph::Modality::Audio,
                _ => crate::graph::Modality::Composite,
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
        }
    }

    fn inject(&mut self, label: &str, mod_val: i32) {
        if self.graph.node_count() >= self.max_nodes as i32 {
            self.status = format!("Max nodes ({}) reached", self.max_nodes);
            return;
        }
        let id = concept_hash(label);
        self.graph.add_node(id, label, mod_val);
        let m = match mod_val {
            0 => crate::graph::Modality::Text,
            1 => crate::graph::Modality::Vision,
            2 => crate::graph::Modality::Audio,
            _ => crate::graph::Modality::Composite,
        };
        self.thinker.add(id, label, m);
        self.status = format!("Injected: {}", label);
    }

    fn step_train(&mut self) {
        let nc = self.graph.node_count() as usize;
        if nc < 2 { return; }
        let ids = self.graph.get_node_ids();
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.step as u64);

        let seed = ids[rng.gen_range(0..ids.len())];
        self.graph.reset_activations();
        self.graph.activate(seed, 3);

        let mut active: Vec<ConceptId> = vec![seed];
        let extra = rng.gen_range(1..4).min(nc - 1);
        for _ in 0..extra {
            let id = ids[rng.gen_range(0..ids.len())];
            if !active.contains(&id) { active.push(id); }
        }
        self.graph.hebbian_learn(&active);

        for i in 0..active.len() {
            for j in (i + 1)..active.len() {
                self.thinker.set_assoc(active[i], active[j]);
            }
        }

        self.step += 1;
        let s = self.graph.saturation();
        self.status = format!("Step {}, Sat: {:.1}%", self.step, s * 100.0);
    }

    fn think(&mut self) {
        let ids = self.graph.get_node_ids();
        if ids.is_empty() { return; }
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.step as u64);
        let seed = ids[rng.gen_range(0..ids.len())];
        let lines = self.thinker.think(seed, 4);
        self.thoughts.push(format!("── think #{} ──", self.thoughts.len()));
        for l in &lines {
            self.thoughts.push(l.clone());
        }
        if self.thoughts.len() > 100 {
            self.thoughts.drain(0..self.thoughts.len() - 80);
        }
        self.status = format!("Thought: {}", lines.first().unwrap_or(&"...".to_string()));
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
            terminal::{self, disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
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
                                self.thoughts.push("s=step t=think r=reset g=grow".to_string());
                                self.thoughts.push("Enter=inject 1-4=inj+mod ?=help q=quit".to_string());
                            }
                            KeyCode::Char('s') => { self.auto_grow(); self.step_train(); }
                            KeyCode::Char('t') => self.think(),
                            KeyCode::Char('r') => {
                                self.graph.reset_activations();
                                self.status = "Activations reset".to_string();
                            }
                            KeyCode::Char('g') => self.auto_grow(),
                            KeyCode::Char('1') => { let t = self.input.text.clone(); self.inject(&t, 0); }
                            KeyCode::Char('2') => { let t = self.input.text.clone(); self.inject(&t, 1); }
                            KeyCode::Char('3') => { let t = self.input.text.clone(); self.inject(&t, 2); }
                            KeyCode::Char('4') => { let t = self.input.text.clone(); self.inject(&t, 3); }
                            KeyCode::Char(c) if !c.is_control() => self.input.insert(c),
                            KeyCode::Backspace => self.input.backspace(),
                            KeyCode::Enter => {
                                let t = self.input.text.clone();
                                if !t.is_empty() {
                                    self.inject(&t, 0);
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
            .constraints([Constraint::Percentage(30), Constraint::Percentage(35), Constraint::Percentage(35)])
            .split(chunks[1]);

        self.draw_header(frame, chunks[0]);
        self.draw_stats(frame, mid[0]);
        self.draw_nodes(frame, mid[1]);
        self.draw_thoughts(frame, mid[2]);
        self.draw_footer(frame, chunks[2]);
    }

    fn draw_header(&self, frame: &mut Frame, area: Rect) {
        let coh = self.graph.coherence();
        let sat = self.graph.saturation();
        let text = format!(
            " MUS v{} | coh:{:.0}% sat:{:.0}% act:{} nodes:{}",
            0.2, coh * 100.0, sat * 100.0, self.graph.active_count(), self.graph.node_count()
        );
        let p = Paragraph::new(Line::from(Span::styled(text, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))))
            .block(Block::bordered().title("MUS Associative Core"));
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
            " [s]tep [t]hink [g]row [r]eset [?]help [q]uit  |  input: {}█",
            self.input.text,
        );
        let p = Paragraph::new(Line::from(Span::styled(text, Style::default().fg(Color::Gray))))
            .block(Block::bordered());
        frame.render_widget(p, area);
    }
}
