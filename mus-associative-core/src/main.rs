mod cuda_bridge;
mod graph;
mod hebbian;
mod metrics;
mod thinker;

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

fn run_cpu(words: &[(&str, Modality)], epochs: usize, steps_per_epoch: usize, slots_per_node: usize, max_nodes: usize) -> thinker::Thinker {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut graph = Graph::new(slots_per_node);
    let mut learner = HebbianLearner::new(1);
    let mut thinker = thinker::Thinker::new();

    for &(label, modality) in words {
        let id = concept_hash(label);
        graph.get_or_create(id, label, modality);
        thinker.add(id, label, modality);
    }

    let mut global_step = 0;
    for epoch in 1..=epochs {
        for _step in 0..steps_per_epoch {
            global_step += 1;
            if global_step % 5 == 0 && graph.nodes.len() < max_nodes {
                let new_label = format!("concept_{}", global_step);
                let new_id = concept_hash(&new_label);
                let mods = [Modality::Text, Modality::Vision, Modality::Composite];
                let m = mods[global_step % 3];
                graph.get_or_create(new_id, &new_label, m);
                thinker.add(new_id, &new_label, m);
            }

            let num_concepts = graph.nodes.len();
            if num_concepts >= 2 {
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
                learner.observe(&mut graph, &active_set);
                // Track associations in thinker
                for i in 0..active_set.len() {
                    for j in (i + 1)..active_set.len() {
                        thinker.set_assoc(active_set[i], active_set[j]);
                    }
                }
            }

            let saturation = graph.slot_saturation();
            if saturation > 0.90 {
                graph.evict_oldest(0.05);
            }
        }
        metrics::report(&graph, global_step, epoch, epochs);
    }

    println!("  Done. Всего концептов: {}", graph.nodes.len());
    thinker
}

fn run_gpu(words: &[(&str, Modality)], epochs: usize, steps_per_epoch: usize, slots_per_node: usize, max_nodes: usize) -> thinker::Thinker {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut graph = cuda_bridge::CudaGraph::new(max_nodes as i32, slots_per_node as i32);
    let mut thinker = thinker::Thinker::new();

    for &(label, modality) in words {
        let id = concept_hash(label);
        let mod_val = match modality {
            Modality::Text => 0,
            Modality::Vision => 1,
            Modality::Audio => 2,
            Modality::Composite => 3,
        };
        graph.add_node(id, label, mod_val);
        thinker.add(id, label, modality);
    }

    let mut global_step = 0;
    for epoch in 1..=epochs {
        for _step in 0..steps_per_epoch {
            global_step += 1;
            if global_step % 5 == 0 && graph.node_count() < max_nodes as i32 {
                let new_label = format!("concept_{}", global_step);
                let new_id = concept_hash(&new_label);
                let mods = [0, 1, 3];
                let mod_vals = [Modality::Text, Modality::Vision, Modality::Composite];
                let mi = global_step % 3;
                graph.add_node(new_id, &new_label, mods[mi]);
                thinker.add(new_id, &new_label, mod_vals[mi]);
            }

            let num_concepts = graph.node_count() as usize;
            if num_concepts >= 2 {
                let ids = graph.get_node_ids();
                graph.reset_activations();
                let seed = ids[rng.gen_range(0..ids.len())];
                graph.activate(seed, 3);
                let mut active_set: Vec<ConceptId> = vec![seed];
                let extra = rng.gen_range(1..4).min(num_concepts - 1);
                for _ in 0..extra {
                    let id = ids[rng.gen_range(0..ids.len())];
                    if !active_set.contains(&id) {
                        active_set.push(id);
                    }
                }
                graph.hebbian_learn(&active_set);
                for i in 0..active_set.len() {
                    for j in (i + 1)..active_set.len() {
                        thinker.set_assoc(active_set[i], active_set[j]);
                    }
                }
            }

            let saturation = graph.saturation();
            if saturation > 0.90 {
                graph.evict_oldest(0.05);
            }
        }

        let coherence = graph.coherence();
        let saturation = graph.saturation();
        let active = graph.active_count();
        let total_nodes = graph.node_count();
        let sat_bar = metrics::saturation_bar(saturation, 12);

        println!();
        println!("  [uran-core GPU] Эпоха {}/{} | Шаг: {}", epoch, epochs, global_step);
        println!("  ==================================================");
        println!("  Всего концептов:    {}", total_nodes);
        println!("  Активировано:       {} active concepts", active);
        println!("  Связность (Coh):    {:.1}% ({})", coherence * 100.0, if coherence > 0.7 { "Стабильно" } else { "Разряжено" });
        println!("  Насыщение (Sat):    {:.0}% [{}]", saturation * 100.0, sat_bar);
        println!("  ==================================================");
        println!();
    }

    println!("  Done (GPU). Всего концептов: {}", graph.node_count());
    thinker
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let use_gpu = !args.iter().any(|a| a == "--cpu");

    let epochs = 3;
    let steps_per_epoch = 100;
    let slots_per_node = 32;
    let max_nodes = 64;

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

    if use_gpu {
        println!("  ╔══════════════════════════════════════════════════╗");
        println!("  ║   MUS Associative Core — CUDA Graph Engine     ║");
        println!("  ║   Режим: GPU (CUDA)                            ║");
        println!("  ║   Слотов на узел: {}                         ║", slots_per_node);
        println!("  ╚══════════════════════════════════════════════════╝");
        println!();

        let thinker = run_gpu(&words, epochs, steps_per_epoch, slots_per_node, max_nodes);

        println!();
        println!("  ─── Мысли ──────────────────────────────");
        let seeds: Vec<ConceptId> = words.iter().map(|(l, _)| concept_hash(l)).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for _ in 0..3 {
            let seed = seeds[rng.gen_range(0..seeds.len())];
            let lines = thinker.think(seed, 4);
            for l in &lines {
                println!("  {}", l);
            }
            println!();
        }
    } else {
        println!("  ╔══════════════════════════════════════════════════╗");
        println!("  ║   MUS Associative Core — Hebbian Graph Engine  ║");
        println!("  ║   Режим: CPU (Rust)                            ║");
        println!("  ║   Слотов на узел: {}                         ║", slots_per_node);
        println!("  ╚══════════════════════════════════════════════════╝");
        println!();

        let thinker = run_cpu(&words, epochs, steps_per_epoch, slots_per_node, max_nodes);

        println!();
        println!("  ─── Мысли ──────────────────────────────");
        let seeds: Vec<ConceptId> = words.iter().map(|(l, _)| concept_hash(l)).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for _ in 0..3 {
            let seed = seeds[rng.gen_range(0..seeds.len())];
            let lines = thinker.think(seed, 4);
            for l in &lines {
                println!("  {}", l);
            }
            println!();
        }
    }
}
