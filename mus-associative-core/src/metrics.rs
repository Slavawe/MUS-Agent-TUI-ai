use crate::graph::Graph;

pub fn report(graph: &Graph, step: usize, epoch: usize, epochs: usize) {
    let coherence = graph.coherence();
    let saturation = graph.slot_saturation();
    let active = graph.active_concepts();
    let total_nodes = graph.nodes.len();

    let sat_bar = saturation_bar(saturation, 12);

    println!();
    println!("  [uran-core] Эпоха {}/{} | Шаг: {}", epoch, epochs, step);
    println!("  ==================================================");
    println!("  Всего концептов:    {}", total_nodes);
    println!("  Активировано:       {} active concepts", active);
    println!("  Связность (Coh):    {:.1}% ({})", coherence * 100.0, if coherence > 0.7 { "Стабильно" } else { "Разряжено" });
    println!("  Насыщение (Sat):    {:.0}% [{}]", saturation * 100.0, sat_bar);
    println!("  ==================================================");
    println!();
}

pub fn saturation_bar(sat: f32, width: usize) -> String {
    let filled = (sat * width as f32).round() as usize;
    let filled = filled.min(width);
    let empty = width - filled;
    format!("{}{}", "|".repeat(filled), "-".repeat(empty))
}
