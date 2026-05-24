use crate::cuda_bridge::CudaGraph;
use crate::thinker::{self, ConceptId, SystemState};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, List, ListItem, Paragraph},
    Frame,
};

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

pub struct AssocApp {
    graph: CudaGraph,
    thinker: thinker::Thinker,
    state: SystemState,
    history: Vec<(String, Vec<(ConceptId, f32)>)>,
    input: InputField,
    status: String,
}

impl AssocApp {
    pub fn new(graph: CudaGraph, thinker: thinker::Thinker) -> Self {
        let state = SystemState::new();
        let mut history = Vec::new();
        history.push(("Welcome! Type a concept name to see associations.".to_string(), vec![]));
        AssocApp {
            graph, thinker, state,
            history, input: InputField::new(),
            status: "Ready. Type a concept, press Enter.".to_string(),
        }
    }

    fn query(&mut self, text: &str) {
        let trimmed = text.trim();
        if trimmed.is_empty() { return; }

        let id = thinker::concept_hash(trimmed);
        let label = self.thinker.label(id);

        let search_id = if label != "?" {
            id
        } else {
            let matches: Vec<ConceptId> = self.thinker.concept_names()
                .iter()
                .filter(|w| w.contains(trimmed) || trimmed.contains(w.as_str()))
                .map(|w| thinker::concept_hash(w))
                .collect();
            if matches.is_empty() {
                self.history.push((format!("Unknown: '{}'", trimmed), vec![]));
                self.status = format!("No concept '{}' found", trimmed);
                return;
            }
            matches[0]
        };

        self.graph.reset_activations();
        let cuda_state = self.state.to_cuda();
        self.graph.activate_chem(search_id, 5, &cuda_state);
        let activated = self.graph.get_top_activations(10);

        let label = self.thinker.label(search_id).to_string();
        self.history.push((label, activated.clone()));
        self.status = format!("Query: {} — {} associations", self.thinker.label(search_id), activated.len());
        self.input.clear();
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

        use std::time::{Duration, Instant};
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
                            KeyCode::Enter => {
                                let t = self.input.text.clone();
                                if !t.is_empty() {
                                    self.query(&t);
                                }
                            }
                            KeyCode::Char(c) if !c.is_control() => self.input.insert(c),
                            KeyCode::Backspace => self.input.backspace(),
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

        self.draw_header(frame, chunks[0]);
        self.draw_main(frame, chunks[1]);
        self.draw_footer(frame, chunks[2]);
    }

    fn draw_header(&self, frame: &mut Frame, area: Rect) {
        let coh = self.graph.coherence();
        let sat = self.graph.saturation();
        let text = format!(
            " MUS Concept Graph  |  coh:{:.0}% sat:{:.0}% nodes:{}  |  D:{:.2} A:{:.2} E:{:.2}",
            coh * 100.0, sat * 100.0,
            self.graph.node_count(),
            self.state.dopamin, self.state.adrenaline, self.state.energy,
        );
        let p = Paragraph::new(Line::from(Span::styled(text, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))))
            .block(Block::bordered().title("MUS Associative Core — Query"));
        frame.render_widget(p, area);
    }

    fn draw_main(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
            .split(area);

        self.draw_associations(frame, chunks[0]);
        self.draw_sidebar(frame, chunks[1]);
    }

    fn draw_associations(&self, frame: &mut Frame, area: Rect) {
        let items: Vec<ListItem> = self.history.iter().rev()
            .take(area.height as usize - 2)
            .flat_map(|(query, assocs)| {
                let mut result = Vec::new();
                let style = if assocs.is_empty() {
                    Style::default().fg(Color::Yellow)
                } else {
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
                };
                result.push(ListItem::new(Line::from(
                    Span::styled(format!("▶ {}", query), style)
                )));
                if assocs.is_empty() {
                    result.push(ListItem::new(Line::from(
                        Span::styled("  (no associations)", Style::default().fg(Color::DarkGray))
                    )));
                    result.push(ListItem::new(Line::from(Span::raw(""))));
                } else {
                    for (i, (id, score)) in assocs.iter().enumerate() {
                        let label = self.thinker.label(*id).to_string();
                        let bar_str = bar(*score, 10);
                        let color = if *score > 0.7 { Color::Green }
                                    else if *score > 0.3 { Color::Yellow }
                                    else { Color::DarkGray };
                        let rank = format!("{}.", i + 1);
                        result.push(ListItem::new(Line::from(vec![
                            Span::raw(format!("  {} ", rank)),
                            Span::styled(format!("{:<20}", label), Style::default().fg(color)),
                            Span::styled(bar_str, Style::default().fg(color)),
                            Span::raw(format!(" {:.2}", score)),
                        ])));
                    }
                    result.push(ListItem::new(Line::from(Span::raw(""))));
                }
                result
            })
            .collect();

        let list = List::new(items)
            .block(Block::bordered().title(format!("Query History ({})", self.history.len())));
        frame.render_widget(list, area);
    }

    fn draw_sidebar(&self, frame: &mut Frame, area: Rect) {
        let s = &self.state;
        let coh = self.graph.coherence();
        let sat = self.graph.saturation();

        let gauge_style = |val: f32| -> Style {
            if val > 0.7 { Style::default().fg(Color::Green) }
            else if val > 0.3 { Style::default().fg(Color::Yellow) }
            else { Style::default().fg(Color::Red) }
        };

        let items = vec![
            ListItem::new(Line::from(vec![
                Span::raw("Nodes: "),
                Span::styled(format!("{}", self.graph.node_count()), Style::default().fg(Color::Cyan)),
            ])),
            ListItem::new(Line::from(vec![
                Span::raw("Actives: "),
                Span::styled(format!("{}", self.graph.active_count()), Style::default().fg(Color::Green)),
            ])),
            ListItem::new(""),
            ListItem::new(Line::from(vec![
                Span::raw("Coherence:"),
                Span::styled(format!(" {:.0}%", coh * 100.0), gauge_style(coh)),
            ])),
            ListItem::new(Line::from(Span::raw(format!("  {}", bar(coh, 15))))),
            ListItem::new(Line::from(vec![
                Span::raw("Saturation:"),
                Span::styled(format!(" {:.0}%", sat * 100.0), gauge_style(sat.min(1.0))),
            ])),
            ListItem::new(Line::from(Span::raw(format!("  {}", bar(sat.min(1.0), 15))))),
            ListItem::new(""),
            ListItem::new(Line::from(vec![
                Span::raw("Dopamin:    "),
                Span::styled(format!("{:.2}", s.dopamin),
                    if s.dopamin > 1.0 { Style::default().fg(Color::Red) } else { Style::default().fg(Color::Green) }),
            ])),
            ListItem::new(Line::from(vec![
                Span::raw("Adrenaline: "),
                Span::styled(format!("{:.2}", s.adrenaline), Style::default().fg(Color::Yellow)),
            ])),
            ListItem::new(Line::from(vec![
                Span::raw("Energy:     "),
                Span::styled(format!("{:.2}", s.energy), Style::default().fg(Color::Cyan)),
            ])),
            ListItem::new(""),
            ListItem::new(Line::from(vec![
                Span::raw("Status:"),
            ])),
            ListItem::new(Line::from(
                Span::styled(&self.status, Style::default().fg(Color::White))
            )),
        ];

        let list = List::new(items).block(Block::bordered().title("Graph State"));
        frame.render_widget(list, area);
    }

    fn draw_footer(&self, frame: &mut Frame, area: Rect) {
        let text = format!(
            " Type concept: {}█   [Enter] query  [q/Esc] quit",
            self.input.text,
        );
        let p = Paragraph::new(Line::from(Span::styled(text, Style::default().fg(Color::Gray))))
            .block(Block::bordered());
        frame.render_widget(p, area);
    }
}
