//! Live HTTP server log overlay for the TUI.
//!
//! Opened via `/http-server logs`, shows a scrollable view of recent request /
//! response log entries read from the shared `Arc<Mutex<VecDeque<HttpLogEntry>>>`
//! buffer.  The buffer is re-read on every draw so the display stays live.
//!
//! Key bindings mirror those of `StaticOverlay`: Up / Down / Page / Home / End
//! to scroll, Esc or `q` to close.

use std::collections::VecDeque;
use std::io::Result;
use std::sync::Arc;

use crate::http_server::HttpLogEntry;
use crate::key_hint;
use crate::key_hint::KeyBinding;
use crate::tui;
use crate::tui::TuiEvent;
use crossterm::event::KeyCode;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Stylize;
use ratatui::text::Line;
use ratatui::text::Span;
use ratatui::text::Text;
use ratatui::widgets::Clear;
use ratatui::widgets::Paragraph;
use ratatui::widgets::Widget;
use ratatui::widgets::WidgetRef;
use ratatui::widgets::Wrap;
use tokio::sync::Mutex;

// Key bindings (reuse constants from pager_overlay style).
const KEY_UP: KeyBinding = key_hint::plain(KeyCode::Up);
const KEY_DOWN: KeyBinding = key_hint::plain(KeyCode::Down);
const KEY_K: KeyBinding = key_hint::plain(KeyCode::Char('k'));
const KEY_J: KeyBinding = key_hint::plain(KeyCode::Char('j'));
const KEY_PAGE_UP: KeyBinding = key_hint::plain(KeyCode::PageUp);
const KEY_PAGE_DOWN: KeyBinding = key_hint::plain(KeyCode::PageDown);
const KEY_HOME: KeyBinding = key_hint::plain(KeyCode::Home);
const KEY_END: KeyBinding = key_hint::plain(KeyCode::End);
const KEY_SPACE: KeyBinding = key_hint::plain(KeyCode::Char(' '));
const KEY_ESC: KeyBinding = key_hint::plain(KeyCode::Esc);
const KEY_Q: KeyBinding = key_hint::plain(KeyCode::Char('q'));

/// Scrollable overlay that live-reads from the shared HTTP log buffer.
pub(crate) struct LogsOverlay {
    log_buffer: Arc<Mutex<VecDeque<HttpLogEntry>>>,
    scroll_offset: usize,
    pub(crate) is_done: bool,
}

impl LogsOverlay {
    pub(crate) fn new(log_buffer: Arc<Mutex<VecDeque<HttpLogEntry>>>) -> Self {
        Self {
            log_buffer,
            scroll_offset: usize::MAX, // start pinned to bottom
            is_done: false,
        }
    }

    pub(crate) fn is_done(&self) -> bool {
        self.is_done
    }

    /// Read the current log buffer and format it as display lines.
    ///
    /// Each `HttpLogEntry` produces two lines: the `→` request line and the
    /// `←` response line (or an error line if the call failed).
    fn format_lines(entries: &VecDeque<HttpLogEntry>) -> Vec<Line<'static>> {
        let mut lines = Vec::new();
        for entry in entries {
            // Timestamp prefix.
            let ts = entry.timestamp.format("%H:%M:%S").to_string();
            // Request line.
            let req_line = format!("{} {}", ts, entry.request_line());
            lines.push(Line::from(Span::raw(req_line)));
            // Response line (only if status or error is known).
            if entry.status.is_some() || entry.error.is_some() {
                let resp_line = format!("         {}", entry.response_line());
                lines.push(Line::from(Span::raw(resp_line).dim()));
            }
        }
        lines
    }

    pub(crate) fn render(&mut self, area: Rect, buf: &mut Buffer) {
        // Read the log buffer (non-async: best-effort try_lock).
        let entries = self
            .log_buffer
            .try_lock()
            .map(|g| g.clone())
            .unwrap_or_default();

        let lines = Self::format_lines(&entries);
        let content_height = lines.len();

        Clear.render(area, buf);

        // Header.
        let header = "/ HTTP Server Logs (Esc to close)";
        Span::from("/ ".repeat(area.width as usize / 2))
            .dim()
            .render_ref(area, buf);
        Span::from(header.to_string())
            .dim()
            .render_ref(area, buf);

        // Content area (leave one row for header, two for footer).
        let content_area = Rect {
            x: area.x,
            y: area.y + 1,
            width: area.width,
            height: area.height.saturating_sub(3),
        };

        // Clamp scroll_offset.
        let max_scroll = content_height.saturating_sub(content_area.height as usize);
        if self.scroll_offset == usize::MAX {
            self.scroll_offset = max_scroll;
        } else {
            self.scroll_offset = self.scroll_offset.min(max_scroll);
        }

        // Render visible lines.
        let visible_start = self.scroll_offset.min(content_height);
        let visible_lines: Vec<Line<'static>> = lines
            .into_iter()
            .skip(visible_start)
            .take(content_area.height as usize)
            .collect();

        let para = Paragraph::new(Text::from(visible_lines)).wrap(Wrap { trim: false });
        para.render(content_area, buf);

        // Footer separator + percentage + hint.
        let sep_y = content_area.y + content_area.height;
        let sep_rect = Rect::new(area.x, sep_y, area.width, 1);
        Span::from("─".repeat(area.width as usize))
            .dim()
            .render_ref(sep_rect, buf);

        let pct = if content_height == 0 {
            100u8
        } else if max_scroll == 0 {
            100
        } else {
            ((self.scroll_offset.min(max_scroll) as f32 / max_scroll as f32) * 100.0).round()
                as u8
        };
        let pct_text = format!(" {}% ", pct);
        let pct_w = pct_text.chars().count() as u16;
        let pct_x = sep_rect.x + sep_rect.width.saturating_sub(pct_w + 1);
        Span::from(pct_text)
            .dim()
            .render_ref(Rect::new(pct_x, sep_y, pct_w, 1), buf);

        // Hint line.
        let hint_rect = Rect::new(area.x, sep_y + 1, area.width, 1);
        let hint = Line::from(vec![
            " ".into(),
            Span::from("↑↓").dim(),
            " to scroll   ".dim(),
            Span::from("q/Esc").dim(),
            " to close".dim(),
        ]);
        Paragraph::new(hint).render(hint_rect, buf);
    }

    pub(crate) fn handle_event(&mut self, tui: &mut tui::Tui, event: TuiEvent) -> Result<()> {
        match event {
            TuiEvent::Key(key_event) => {
                if KEY_ESC.is_press(key_event) || KEY_Q.is_press(key_event) {
                    self.is_done = true;
                } else if KEY_UP.is_press(key_event) || KEY_K.is_press(key_event) {
                    self.scroll_offset = self.scroll_offset.saturating_sub(1);
                } else if KEY_DOWN.is_press(key_event) || KEY_J.is_press(key_event) {
                    self.scroll_offset = self.scroll_offset.saturating_add(1);
                } else if KEY_PAGE_UP.is_press(key_event) {
                    let page = tui
                        .terminal
                        .viewport_area
                        .height
                        .saturating_sub(4) as usize;
                    self.scroll_offset = self.scroll_offset.saturating_sub(page);
                } else if KEY_PAGE_DOWN.is_press(key_event) || KEY_SPACE.is_press(key_event) {
                    let page = tui
                        .terminal
                        .viewport_area
                        .height
                        .saturating_sub(4) as usize;
                    self.scroll_offset = self.scroll_offset.saturating_add(page);
                } else if KEY_HOME.is_press(key_event) {
                    self.scroll_offset = 0;
                } else if KEY_END.is_press(key_event) {
                    self.scroll_offset = usize::MAX;
                }
                tui.frame_requester().schedule_frame();
            }
            TuiEvent::Draw => {
                tui.draw(u16::MAX, |frame| {
                    self.render(frame.area(), frame.buffer);
                })?;
            }
            _ => {}
        }
        Ok(())
    }
}
