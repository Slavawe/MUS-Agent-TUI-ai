use std::io;
use std::time::Instant;

use anyhow::Result;
use chrono::Local;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;
use ratatui::Terminal;

fn main() -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    stdout.execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new();
    let res = app.run(&mut terminal);

    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    if let Err(e) = res {
        eprintln!("Error: {}", e);
    }
    Ok(())
}

struct ChatMessage {
    role: String,
    text: String,
    time: String,
}

struct InputBuffer {
    content: String,
    cursor: usize,
}

struct App {
    messages: Vec<ChatMessage>,
    input: InputBuffer,
    scroll: usize,
    status: String,
    start_time: Instant,
}

impl App {
    fn new() -> Self {
        let mut app = Self {
            messages: Vec::new(),
            input: InputBuffer {
                content: String::new(),
                cursor: 0,
            },
            scroll: 0,
            status: "Ready".to_string(),
            start_time: Instant::now(),
        };
        app.add_message("system", "Uragan 1.0 Agent TUI initialized");
        app.add_message(
            "agent",
            "Hello! I am Uragan AI. Ask me anything about code, \
             generation, or analysis.\n\
             Commands:\n  \
             /clear  — clear chat\n  \
             /info   — system info\n  \
             /exit   — quit",
        );
        app
    }

    fn add_message(&mut self, role: &str, text: &str) {
        self.messages.push(ChatMessage {
            role: role.to_string(),
            text: text.to_string(),
            time: Local::now().format("%H:%M:%S").to_string(),
        });
        self.scroll = 0;
    }

    fn send_message(&mut self) {
        let text = self.input.content.trim().to_string();
        if text.is_empty() {
            return;
        }

        self.add_message("user", &text);

        let response = match text.as_str() {
            "/clear" => {
                self.messages.clear();
                self.add_message("system", "Chat cleared");
                return;
            }
            "/info" => {
                format!(
                    "Uragan 1.0\n  \
                     Uptime: {}s\n  \
                     Messages: {}\n  \
                     Platform: {}",
                    self.start_time.elapsed().as_secs(),
                    self.messages.len(),
                    std::env::consts::OS,
                )
            }
            "/exit" => std::process::exit(0),
            _ => {
                format!(
                    "[echo] {}\n\n\
                     (Model inference not yet connected — \
                     this is the TUI shell)",
                    text
                )
            }
        };

        self.add_message("agent", &response);
        self.input.content.clear();
        self.input.cursor = 0;
    }

    fn run(&mut self, terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
        loop {
            terminal.draw(|f| self.render(f))?;

            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Enter => {
                            if key.modifiers == KeyModifiers::SHIFT {
                                self.insert_char('\n');
                            } else {
                                self.send_message();
                            }
                        }
                        KeyCode::Char('q') if key.modifiers == KeyModifiers::NONE => break,
                        KeyCode::Char(c) => self.insert_char(c),
                        KeyCode::Backspace => self.backspace(),
                        KeyCode::Delete => self.delete(),
                        KeyCode::Left => {
                            self.input.cursor = self.input.cursor.saturating_sub(1)
                        }
                        KeyCode::Right => {
                            self.input.cursor =
                                self.input.cursor.min(self.input.content.len())
                        }
                        KeyCode::Home => self.input.cursor = 0,
                        KeyCode::End => self.input.cursor = self.input.content.len(),
                        KeyCode::Up => {
                            if self.messages.len() > 10 {
                                self.scroll =
                                    (self.scroll + 1).min(self.messages.len().saturating_sub(1));
                            }
                        }
                        KeyCode::Down => {
                            self.scroll = self.scroll.saturating_sub(1);
                        }
                        KeyCode::Esc => break,
                        _ => {}
                    }
                }
            }
        }
        Ok(())
    }

    fn insert_char(&mut self, c: char) {
        self.input.content.insert(self.input.cursor, c);
        self.input.cursor += 1;
    }

    fn backspace(&mut self) {
        if self.input.cursor > 0 {
            self.input.content.remove(self.input.cursor - 1);
            self.input.cursor -= 1;
        }
    }

    fn delete(&mut self) {
        if self.input.cursor < self.input.content.len() {
            self.input.content.remove(self.input.cursor);
        }
    }

    fn render(&self, frame: &mut Frame) {
        let areas = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(1),
                Constraint::Length(6),
                Constraint::Length(1),
            ])
            .split(frame.area());

        self.render_header(frame, areas[0]);
        self.render_chat(frame, areas[1]);
        self.render_input(frame, areas[2]);
        self.render_footer(frame, areas[3]);
    }

    fn render_header(&self, frame: &mut Frame, area: Rect) {
        let title = Line::from(Span::styled(
            " Uragan 1.0 — Agent TUI ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ));
        let status = Line::from(Span::styled(
            format!(" {} ", self.status),
            Style::default().fg(Color::Green),
        ));
        let header = Paragraph::new(vec![title, status])
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Cyan)),
            )
            .alignment(ratatui::layout::Alignment::Center);
        frame.render_widget(header, area);
    }

    fn render_chat(&self, frame: &mut Frame, area: Rect) {
        let lines: Vec<Line> = self
            .messages
            .iter()
            .rev()
            .skip(self.scroll)
            .rev()
            .flat_map(|msg| self.format_message(msg))
            .collect();

        let chat = Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray))
                    .title(format!(" Chat ({} msgs) ", self.messages.len())),
            )
            .wrap(Wrap { trim: false });
        frame.render_widget(chat, area);
    }

    fn format_message(&self, msg: &ChatMessage) -> Vec<Line<'static>> {
        let role_color = match msg.role.as_str() {
            "user" => Color::Yellow,
            "agent" => Color::Cyan,
            "system" => Color::DarkGray,
            _ => Color::White,
        };
        let tag = msg.role.to_uppercase();

        let mut lines: Vec<Line<'static>> = Vec::new();

        let header = Line::from(vec![
            Span::styled(
                format!(" {} ", tag),
                Style::default()
                    .fg(Color::Black)
                    .bg(role_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(
                msg.time.clone(),
                Style::default().fg(Color::DarkGray),
            ),
        ]);
        lines.push(header);

        for line in msg.text.lines() {
            let wrapped = textwrap::fill(line, 72);
            for wl in wrapped.lines() {
                lines.push(Line::from(Span::raw(format!("  {}", wl))));
            }
        }
        lines.push(Line::from(""));
        lines
    }

    fn render_input(&self, frame: &mut Frame, area: Rect) {
        let input_style = if self.input.content.is_empty() {
            Style::default().fg(Color::DarkGray)
        } else {
            Style::default().fg(Color::White)
        };

        let prefix = Span::styled(" > ", Style::default().fg(Color::Green));
        let text = Span::styled(&self.input.content, input_style);

        let hint = if self.input.content.is_empty() {
            Some(Span::styled(
                " Type a message... (Shift+Enter = newline, Esc = quit)",
                Style::default().fg(Color::DarkGray),
            ))
        } else {
            None
        };

        let mut spans = vec![prefix, text];
        if let Some(h) = hint {
            spans.push(h);
        }

        let input = Paragraph::new(Line::from(spans))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Green))
                    .title(" Input "),
            );
        frame.render_widget(input, area);

        let x = area.x + 4 + self.input.cursor as u16;
        let y = area.y + 1;
        frame.set_cursor_position((x, y));
    }

    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let footer = Paragraph::new(Line::from(vec![
            Span::styled(" Ctrl+C ", Style::default().fg(Color::DarkGray)),
            Span::raw("quit  "),
            Span::styled(" ↑/↓ ", Style::default().fg(Color::DarkGray)),
            Span::raw("scroll  "),
            Span::styled(" /info ", Style::default().fg(Color::DarkGray)),
            Span::raw("system info"),
        ]))
        .style(Style::default().fg(Color::DarkGray).bg(Color::Black));
        frame.render_widget(footer, area);
    }
}
