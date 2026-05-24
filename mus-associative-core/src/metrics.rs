pub fn bar(value: f32, max: f32, width: usize) -> String {
    if max <= 0.0 { return " ".repeat(width); }
    let filled = ((value / max) * width as f32).round() as usize;
    let filled = filled.min(width);
    let empty = width.saturating_sub(filled);
    format!("{}{}", "█".repeat(filled), "░".repeat(empty))
}

pub fn emotional_bar(value: f32, max: f32, width: usize, label: &str) -> String {
    let b = bar(value, max, width);
    format!("{} {:.2}/{:.2} {}", label, value, max, b)
}

pub fn saturation_bar(sat: f32, width: usize) -> String {
    bar(sat, 1.0, width)
}
