mod graph;
mod hebbian;
mod metrics;

use graph::{ConceptId, Graph, Modality};
use hebbian::HebbianLearner;
use rand::{Rng, SeedableRng};

fn concept_hash(s: &str) -> ConceptId {
    let mut h: u64 = 14695981039346656037;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

fn simulate_step(graph: &mut Graph, learner: &mut HebbianLearner, rng: &mut impl Rng) {
    let num_concepts = graph.nodes.len();
    if num_concepts < 2 { return; }

    let ids: Vec<ConceptId> = graph.nodes.keys().copied().collect();

    graph.reset_activations();

    let seed = ids[rng.gen_range(0..ids.len())];
    graph.activate(seed, 3);

    let mut active_set: Vec<ConceptId> = vec![seed];

    let extra = rng.gen_range(1..4).min(num_concepts - 1);
    for _ in 0..extra {
        let id = ids[rng.gen_range(0..ids.len())];
        if !active_set.contains(&id) {
            active_set.push(id);
            if let Some(node) = graph.nodes.get_mut(&id) {
                node.activation = 1.0;
            }
        }
    }

    learner.observe(graph, &active_set);
}

fn main() {
    let epochs = 3;
    let steps_per_epoch = 100;
    let slots_per_node = 32;

    let mut graph = Graph::new(slots_per_node);
    let mut learner = HebbianLearner::new(1);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    println!("  ╔══════════════════════════════════════════════════╗");
    println!("  ║   MUS Associative Core — Hebbian Graph Engine   ║");
    println!("  ║   Слотов на узел: {}                         ║", slots_per_node);
    println!("  ╚══════════════════════════════════════════════════╝");
    println!();

    let words = [
        ("лицо", Modality::Text), ("face", Modality::Vision),
        ("радость", Modality::Text), ("smile", Modality::Vision),
        ("грусть", Modality::Text), ("frown", Modality::Vision),
        ("код", Modality::Text), ("rust", Modality::Text),
        ("нейросеть", Modality::Text), ("ai", Modality::Composite),
        ("ассоциация", Modality::Text), ("link", Modality::Composite),
        ("ascii_face", Modality::Vision), ("contour", Modality::Vision),
        ("процессор", Modality::Text), ("gpu", Modality::Text),
        ("память", Modality::Text), ("slot", Modality::Composite),
    ];

    for &(label, modality) in &words {
        let id = concept_hash(label);
        graph.get_or_create(id, label, modality);
    }

    let mut global_step = 0;

    for epoch in 1..=epochs {
        println!("  ─── Эпоха {}/{} ──────────────────────", epoch, epochs);

        for step in 0..steps_per_epoch {
            global_step += 1;

            if global_step % 5 == 0 && graph.nodes.len() < 64 {
                let new_label = format!("concept_{}", global_step);
                let new_id = concept_hash(&new_label);
                let mods = [Modality::Text, Modality::Vision, Modality::Composite];
                graph.get_or_create(new_id, &new_label, mods[global_step % 3]);
            }

            simulate_step(&mut graph, &mut learner, &mut rng);

            let saturation = graph.slot_saturation();
            if saturation > 0.90 {
                let evicted = graph.evict_oldest(0.05);
                println!("    Evict: {} старых связей освобождено", evicted);
            }
        }

        metrics::report(&graph, global_step, epoch, epochs);
    }

    println!("  Done. Всего концептов: {}", graph.nodes.len());
}
