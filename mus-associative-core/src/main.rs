mod assoc_tui;
mod ascii_3d;
mod blackboard;
mod coder;
mod cuda_bridge;
mod data;
mod executor;
mod graph;
mod hdc;
mod hebbian;
mod metrics;
mod thinker;
mod tui;
mod wasm_sandbox;

use graph::Graph;
use graph::Modality;
use hdc::HDCPatternMemory;
use hebbian::HebbianLearner;
use rand::{Rng, SeedableRng};
use thinker::{SystemState, ConceptId};

fn flatten_pairs(pairs: &[(u64, u64)]) -> Vec<u64> {
    let mut flat = Vec::with_capacity(pairs.len() * 2);
    for &(a, b) in pairs {
        flat.push(a);
        flat.push(b);
    }
    flat
}

fn concept_hash(s: &str) -> ConceptId {
    let mut h: u64 = 14695981039346656037;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

fn run_cpu(words: &[(&str, Modality)], epochs: usize, steps_per_epoch: usize, slots_per_node: usize, max_nodes: usize) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut graph = Graph::new(slots_per_node);
    let mut learner = HebbianLearner::new(1);
    let mut thinker = thinker::Thinker::new();

    for &(label, modality) in words {
        let id = concept_hash(label);
        graph.get_or_create(id, label, modality);
        thinker.add(id, label, modality);
    }

    let mut state = SystemState::new();
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
                for i in 0..active_set.len() {
                    for j in (i + 1)..active_set.len() {
                        thinker.set_assoc(active_set[i], active_set[j]);
                    }
                }
            }

            let saturation = graph.slot_saturation();
            state.update_by_metrics(graph.coherence(), saturation);
            if state.panic_active {
                graph.evict_oldest(state.adrenaline);
            }
            if saturation > 0.90 {
                graph.evict_oldest(0.05);
            }
        }

        let coh = graph.coherence();
        let sat = graph.slot_saturation();
        let active = graph.active_concepts();
        let total = graph.nodes.len();
        let sat_bar = metrics::saturation_bar(sat, 12);

        println!();
        println!("  [CPU] Epoch {}/{} | Step: {}", epoch, epochs, global_step);
        println!("  ==================================================");
        println!("  Nodes:              {}", total);
        println!("  Active:             {}", active);
        println!("  Coherence:          {:.1}% ({})", coh * 100.0, if coh > 0.7 { "Stable" } else { "Sparse" });
        println!("  Saturation:         {:.0}% [{}]", sat * 100.0, sat_bar);
        println!("  Dopamin:            {:.2}", state.dopamin);
        println!("  Adrenaline:         {:.2}", state.adrenaline);
        println!("  Energy:             {:.2}", state.energy);
        println!("  ==================================================");
    }

    println!();
    println!("  Done. Total concepts: {}", graph.nodes.len());
}

fn run_gpu(words: &[(&str, Modality)], epochs: usize, steps_per_epoch: usize, slots_per_node: usize, max_nodes: usize) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let graph = cuda_bridge::CudaGraph::new(max_nodes as i32, 256);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
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

    let mut state = SystemState::new();
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
                let cuda_state = state.to_cuda();
                graph.activate_chem(seed, 3, &cuda_state);

                let mut active_set: Vec<ConceptId> = vec![seed];
                let extra = rng.gen_range(1..4).min(num_concepts - 1);
                for _ in 0..extra {
                    let id = ids[rng.gen_range(0..ids.len())];
                    if !active_set.contains(&id) {
                        active_set.push(id);
                    }
                }
                graph.hebbian_learn(&active_set, state.dopamin, state.adrenaline);
                for i in 0..active_set.len() {
                    for j in (i + 1)..active_set.len() {
                        thinker.set_assoc(active_set[i], active_set[j]);
                    }
                }
            }

            // STDP weight management every 5 steps
            if global_step % 5 == 0 {
                let cuda_state = state.to_cuda();
                graph.decay_weights(cuda_state.weight_decay);
                graph.prune_weights(cuda_state.weight_prune);
            }

            let coherence = graph.coherence();
            let saturation = graph.saturation();
            state.update_by_metrics(coherence, saturation);

            if state.panic_active {
                graph.panic_clear(state.adrenaline, state.panic_threshold);
            }
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
        println!("  [GPU] Epoch {}/{} | Step: {}", epoch, epochs, global_step);
        println!("  ==================================================");
        println!("  Nodes:              {}", total_nodes);
        println!("  Active:             {}", active);
        println!("  Coherence:          {:.1}% ({})", coherence * 100.0, if coherence > 0.7 { "Stable" } else { "Sparse" });
        println!("  Saturation:         {:.0}% [{}]", saturation * 100.0, sat_bar);
        println!("  Dopamin:            {:.2}", state.dopamin);
        println!("  Adrenaline:         {:.2}", state.adrenaline);
        println!("  Energy:             {:.2}", state.energy);
        println!("  Panic:              {}", if state.panic_active { "ACTIVE" } else { "inactive" });
        println!("  ==================================================");
    }

    println!();
    println!("  Done (GPU). Total concepts: {}", graph.node_count());

    // Generate thoughts with neurochemical activation
    println!();
    println!("  ─── Neurochemical Thoughts ───");
    let seeds: Vec<ConceptId> = words.iter().map(|(l, _)| concept_hash(l)).collect();
    for _ in 0..3 {
        let seed = seeds[rng.gen_range(0..seeds.len())];
        let lines = thinker.think(&graph, seed, 6);
        for l in &lines {
            println!("  {}", l);
        }
        println!();
    }
}

fn load_concept_tsv(path: &str, graph: &cuda_bridge::CudaGraph, thinker: &mut thinker::Thinker) -> Vec<(u64, u64)> {
    use std::io::{BufRead, BufReader};
    use std::fs::File;

    let file = File::open(path).expect("Cannot open concept_data.tsv");
    let reader = BufReader::new(file);
    let mut pairs: Vec<(u64, u64)> = Vec::new();
    let mut all_words: Vec<String> = Vec::new();
    let mut word_set: std::collections::HashSet<String> = std::collections::HashSet::new();

    for line in reader.lines().skip(1) {
        let line = line.expect("Read error");
        if line.trim().is_empty() { continue; }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 2 { continue; }
        let a = parts[0].trim().to_string();
        let b = parts[1].trim().to_string();
        if a == b { continue; }

        let id_a = concept_hash(&a);
        let id_b = concept_hash(&b);
        pairs.push((id_a, id_b));

        if word_set.insert(a.clone()) { all_words.push(a); }
        if word_set.insert(b.clone()) { all_words.push(b); }
    }

    for word in &all_words {
        let id = concept_hash(word);
        let mod_val = match word.len() {
            l if l > 6 => 0,
            l if l < 3 => 1,
            _ => 3,
        };
        graph.add_node(id, word, mod_val);
        thinker.add(id, word, Modality::Text);
    }
    println!("  Read {} pairs, {} unique concepts", pairs.len(), all_words.len());
    pairs
}

fn train_concept_graph(graph: &cuda_bridge::CudaGraph, pairs: &[(u64, u64)], num_epochs: i32) {
    let mut state = SystemState::new();
    let flat = flatten_pairs(pairs);
    for epoch in 1..=num_epochs {
        graph.batch_link(&flat, state.dopamin, state.adrenaline);
        let sat = graph.saturation();
        let coh = graph.coherence();
        state.update_by_metrics(coh, sat);
        let cuda_state = state.to_cuda();
        graph.decay_weights(cuda_state.weight_decay);
        graph.prune_weights(cuda_state.weight_prune);
        println!("  Epoch {}/{}: Sat={:.1}% Coh={:.1}% D={:.2} A={:.2} E={:.2}",
            epoch, num_epochs, sat * 100.0, coh * 100.0,
            state.dopamin, state.adrenaline, state.energy);
    }
}

fn run_concept_data(path: String, _max_edges: i32, num_epochs: i32) {
    let max_nodes = 10000;
    let graph = cuda_bridge::CudaGraph::new(max_nodes, 256);
    let mut thinker = thinker::Thinker::new();
    let mut state = SystemState::new();

    let pairs = load_concept_tsv(&path, &graph, &mut thinker);
    println!("  Nodes created: {}", graph.node_count());
    train_concept_graph(&graph, &pairs, num_epochs);

    // HDC-enhanced walk
    let all_words: Vec<String> = thinker.concept_names();
    let mut hdc_hits = 0u64;
    let mut hdc_total = 0u64;
    let seeds: Vec<u64> = all_words.iter().step_by(all_words.len().max(10) / 10).take(5)
        .map(|w| concept_hash(w)).collect();

    for &seed in &seeds {
        let mut cur = seed;
        let mut tokens: Vec<(u64, f32)> = vec![(cur, 1.0)];
        for _ in 0..20 {
            hdc_total += 1;
            if thinker.hdc_memory.len() > 2 && tokens.len() >= 2 {
                let query: Vec<u64> = tokens.iter().rev().take(3).map(|x| x.0).rev().collect();
                if let Some(comp) = thinker.hdc_memory.complete(&query, 1) {
                    if let Some(&next_tok) = comp.first() {
                        if next_tok != cur {
                            cur = next_tok;
                            tokens.push((cur, 0.0));
                            hdc_hits += 1;
                            continue;
                        }
                    }
                }
            }

            graph.reset_activations();
            let cuda_state = state.to_cuda();
            graph.activate_chem(cur, 4, &cuda_state);

            let ids = graph.get_top_activations(10);
            if ids.is_empty() { break; }

                for &(act_id, score) in &ids {
                    let already = tokens.iter().any(|(id, _)| *id == act_id);
                    if !already {
                        thinker.set_assoc(cur, act_id);
                        cur = act_id;
                        tokens.push((cur, score));
                        break;
                    }
                }
        }
        let token_ids: Vec<u64> = tokens.iter().map(|(id, _)| *id).collect();
        thinker.hdc_memory.store(&token_ids);

        let chain_str: Vec<String> = tokens.iter()
            .map(|(id, score)| format!("{}[{:.2}]", thinker.label(*id), score))
            .collect();
        println!("  Chain: {}", chain_str.join(" → "));
    }

    println!("  HDC hits: {}/{} ({:.0}%)", hdc_hits, hdc_total,
        if hdc_total > 0 { hdc_hits as f64 / hdc_total as f64 * 100.0 } else { 0.0 });

    let sat = graph.saturation();
    let coh = graph.coherence();
    state.update_by_metrics(coh, sat);
    let cuda_state = state.to_cuda();
    graph.decay_weights(cuda_state.weight_decay);
    graph.prune_weights(cuda_state.weight_prune);

    // BFS traversal tests
    println!();
    println!("  ─── BFS Traversal Tests ───");
    let test_nodes: Vec<&str> = vec!["Rust", "GPU", "DNA", "quantum", "neural_network",
        "star", "algebra", "noun", "energy", "function"];
    for name in &test_nodes {
        let id = concept_hash(name);
        graph.reset_activations();
        let cuda_state = state.to_cuda();
        graph.activate_chem(id, 5, &cuda_state);
        let activated = graph.get_top_activations(8);
        let names: Vec<String> = activated.iter()
            .map(|(t, s)| format!("{}[{:.2}]", thinker.label(*t), s))
            .collect();
        println!("  {} → {}", name, names.join(", "));
    }

    // Interactive REPL
    println!();
    println!("  ─── Interactive Query Mode ───");
    println!("  Type a concept name to see associations. Type 'quit' to exit.");
    use std::io::{self, Write};
    let stdin = io::stdin();
    let mut input = String::new();
    loop {
        print!("  > ");
        io::stdout().flush().ok();
        input.clear();
        if stdin.read_line(&mut input).ok().is_none() { break; }
        let trimmed = input.trim();
        if trimmed.is_empty() || trimmed == "quit" || trimmed == "exit" { break; }

        let id = concept_hash(trimmed);
        let label = thinker.label(id);
        let search_id = if label == "?" {
            let matches: Vec<ConceptId> = all_words.iter()
                .filter(|w| w.contains(trimmed) || trimmed.contains(w.as_str()))
                .map(|w| concept_hash(w))
                .collect();
            if matches.is_empty() {
                println!("  Unknown concept '{}'", trimmed);
                continue;
            }
            println!("  (did you mean '{}'?)", thinker.label(matches[0]));
            matches[0]
        } else {
            id
        };

        graph.reset_activations();
        let cuda_state = state.to_cuda();
        graph.activate_chem(search_id, 5, &cuda_state);
        let activated = graph.get_top_activations(8);
        if activated.is_empty() {
            println!("  No associations for '{}'", thinker.label(search_id));
        } else {
            let assocs: Vec<String> = activated.iter()
                .map(|(t, s)| format!("{}[{:.2}]", thinker.label(*t), s))
                .collect();
            println!("  {} → {}", thinker.label(search_id), assocs.join(", "));
        }
    }
    println!("  Bye!");
}

fn run_real_gpu(data_path: &str, slots_per_node: i32, vocab_size: i32, use_chem: bool, num_epochs: i32) {
    let graph = cuda_bridge::CudaGraph::new(vocab_size + 1, 256);
    let mut state = SystemState::new();
    let batch_size = 500_000;

    println!("  Pre-creating {} token nodes on GPU...", vocab_size);
    for id in 0..vocab_size {
        let label = format!("tok_{}", id);
        graph.add_node(id as u64, &label, 0);
    }
    println!("  Nodes created. node_count={}", graph.node_count());

    let mut hdc_mem = HDCPatternMemory::new(200);
    let dataset = data::TokenDataset::open(data_path, 256)
        .expect("Failed to open dataset");
    let total_pairs: usize = dataset.num_samples * (dataset.seq_len - 1);
    println!("  Dataset: {} samples, {} bigram pairs", dataset.num_samples, total_pairs);

    for epoch in 1..=num_epochs {
        println!();
        println!("  ─── Epoch {}/{} ───", epoch, num_epochs);

        // Re-iterate dataset
        let dataset = data::TokenDataset::open(data_path, 256).expect("Failed to open dataset");
        let mut pairs: Vec<(u64, u64)> = Vec::with_capacity(batch_size);
        let mut processed = 0usize;
        let mut hdc_hits = 0usize;
        let mut hdc_total = 0usize;

        for (a, b) in dataset.pairs() {
            pairs.push((a, b));
            if pairs.len() >= batch_size {
                let flat = flatten_pairs(&pairs);
                graph.batch_link(&flat, state.dopamin, state.adrenaline);
                processed += pairs.len();
                if processed % 1_000_000 == 0 {
                    let sat = graph.saturation();
                    let coh = graph.coherence();
                    state.update_by_metrics(coh, sat);
                    let cuda_state = state.to_cuda();
                    graph.decay_weights(cuda_state.weight_decay);
                    graph.prune_weights(cuda_state.weight_prune);
                    if use_chem && state.panic_active {
                        graph.panic_clear(state.adrenaline, state.panic_threshold);
                    }
                }
                pairs.clear();
            }
        }
        if !pairs.is_empty() {
            let flat = flatten_pairs(&pairs);
            graph.batch_link(&flat, state.dopamin, state.adrenaline);
        }

        let sat = graph.saturation();
        let coh = graph.coherence();
        println!("  Train done: Sat={:.1}% Coh={:.1}% D={:.2} A={:.2} E={:.2}",
            sat * 100.0, coh * 100.0, state.dopamin, state.adrenaline, state.energy);

        // HDC-enhanced walk
        if use_chem {
            let mut seeds: Vec<u64> = Vec::new();
            for idx in (0..vocab_size).step_by(997) {
                let w = graph.get_weights(idx);
                if w.iter().filter(|&&v| v > 0.0).count() >= 5 {
                    seeds.push(idx as u64);
                    if seeds.len() >= 10 { break; }
                }
            }
            if seeds.is_empty() { seeds.push(5000); }

            for &seed in seeds.iter().take(5) {
                let mut cur = seed;
                let mut tokens: Vec<u64> = Vec::new();
                tokens.push(cur);

                for _ in 0..20 {
                    hdc_total += 1;
                    if hdc_mem.len() > 3 && tokens.len() >= 2 {
                        let query: Vec<u64> = tokens.iter().rev().take(3).copied().rev().collect();
                        if let Some(comp) = hdc_mem.complete(&query, 1) {
                            if let Some(&next_tok) = comp.first() {
                                if next_tok != cur {
                                    cur = next_tok;
                                    tokens.push(cur);
                                    hdc_hits += 1;
                                    continue;
                                }
                            }
                        }
                    }

                    graph.reset_activations();
                    let cuda_state = state.to_cuda();
                    let n_activated = graph.activate_chem(cur, 3, &cuda_state);
                    if n_activated <= 1 { break; }
                    let top = graph.get_top_activations(5);
                    let next = top.iter().find(|(id, _)| *id != cur);
                    if next.is_none() { break; }
                    cur = next.unwrap().0;
                    tokens.push(cur);
                }

                if tokens.len() > 1 {
                    hdc_mem.store(&tokens);
                    print!("    tok[{}]", tokens[0]);
                    for &t in tokens.iter().skip(1).take(10) {
                        print!(" → tok[{}]", t);
                    }
                    if tokens.len() > 11 { print!(" …"); }
                    println!(" ({} tok)", tokens.len());
                }
            }
            println!("  HDC hits: {}/{}", hdc_hits, hdc_total);
        }
    }

    println!();
    println!("  ═══════════════════════════════════════════════");
    println!("  Training complete!");
    println!("  Total pairs processed: {} × {} epochs", total_pairs, num_epochs);
    println!("  Unique tokens:      {}", graph.node_count());
    println!("  Saturation:         {:.1}%", graph.saturation() * 100.0);
    println!("  HDC patterns stored: {}", hdc_mem.len());

    if use_chem {
        println!();
        println!("  ─── Generated token sequences ───");
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for attempt in 0..5 {
            let mut cur: u64 = rng.gen_range(0..vocab_size) as u64;
            let mut tokens: Vec<u64> = Vec::new();
            tokens.push(cur);

            for _ in 0..20 {
                graph.reset_activations();
                let n_activated = graph.activate(cur, 3);
                if n_activated <= 1 { break; }
                let top = graph.get_top_activations(5);
                let next = top.iter().find(|(id, _)| *id != cur);
                if next.is_none() { break; }
                cur = next.unwrap().0;
                tokens.push(cur);
            }

            if tokens.len() > 1 {
                print!("    tok[{}]", tokens[0]);
                for &t in tokens.iter().skip(1).take(15) {
                    print!(" → tok[{}]", t);
                }
                if tokens.len() > 16 { print!(" …"); }
                println!(" ({} tokens, attempt {})", tokens.len(), attempt);
            }
        }
    }
    println!();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let use_gpu = !args.iter().any(|a| a == "--cpu");
    let train_real = args.iter().any(|a| a == "--train-real");
    let use_chem = args.iter().any(|a| a == "--chem");
    let use_tui = args.iter().any(|a| a == "--tui");

    if use_tui && use_gpu {
        let slots = 32;
        let max_nodes = 128;
        let words: [(&str, u32); 18] = [
            ("лицо", 0), ("face", 1), ("радость", 0), ("smile", 1),
            ("грусть", 0), ("frown", 1), ("код", 0), ("rust", 0),
            ("нейросеть", 0), ("ai", 3), ("ассоциация", 0), ("link", 3),
            ("ascii_face", 1), ("contour", 1), ("процессор", 0), ("gpu", 0),
            ("память", 0), ("slot", 3),
        ];
        let mut app = tui::App::new(&words, slots, max_nodes);
        let _ = app.run();
        return;
    }

    let concept_tui = args.iter().any(|a| a == "--concept-tui");

    if concept_tui {
        let path = args.get(2).cloned().unwrap_or_else(|| "../concept_data.tsv".to_string());
        let _slots = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
        let num_epochs = args.iter()
            .position(|a| a == "--epochs")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);

        let graph = cuda_bridge::CudaGraph::new(10000, 256);
        let mut thinker = thinker::Thinker::new();
        let pairs = load_concept_tsv(&path, &graph, &mut thinker);
        println!("  Loaded {} pairs, {} concepts", pairs.len(), graph.node_count());
        println!("  Training...");
        train_concept_graph(&graph, &pairs, num_epochs);

        let mut app = assoc_tui::AssocApp::new(graph, thinker);
        let _ = app.run();
        return;
    }

    let concept_data = args.iter().any(|a| a == "--concept-data");
    let run_coder = args.iter().any(|a| a == "--coder");
    let auto_exec = args.iter().any(|a| a == "--auto-exec");
    let use_wasm = args.iter().any(|a| a == "--wasm");
    let run_ascii_3d = args.iter().any(|a| a == "--ascii-3d");
    let hurricane_tui = args.iter().any(|a| a == "--hurricane-tui");

    if hurricane_tui {
        let path = args.get(2).cloned().unwrap_or_else(|| "concept_data.tsv".to_string());
        let graph = cuda_bridge::CudaGraph::new(10000, 256);
        let mut thinker = thinker::Thinker::new();
        let pairs = load_concept_tsv(&path, &graph, &mut thinker);
        println!("  Loaded {} pairs, {} concepts", pairs.len(), graph.node_count());
        println!("  Training...");
        train_concept_graph(&graph, &pairs, 2);
        println!("  ─── Hurricane 3D TUI ───");
        println!("  Type a concept name. 'quit' to exit.\n");

        use std::io::{self, Write};
        let mut frame: f32 = 0.0;
        let all_words: Vec<String> = thinker.concept_names();
        let mut active_hash: u64 = concept_hash("Rust");
        let stdin = io::stdin();
        let mut input = String::new();

        loop {
            // Run BFS — activate from seed
            let mut state = thinker::SystemState::new();
            let cuda_state = state.to_cuda();
            graph.reset_activations();
            graph.activate_chem(active_hash, 5, &cuda_state);
            graph.top_k(32);

            // Get ALL node activations
            let all_acts = graph.get_activations();
            let node_ids = graph.get_node_ids();
            let total_nodes = node_ids.len().max(1);

            // Build 3D points: one per node, energy = activation
            let points_3d: Vec<(u32, f32)> = node_ids.iter()
                .enumerate()
                .map(|(i, _id)| (i as u32, all_acts.get(i).copied().unwrap_or(0.0)))
                .collect();

            let graph_points = ascii_3d::map_graph_to_3d(&points_3d, total_nodes, 8.0);

            // Build edges from graph structure (consecutive in activation order for now)
            let edges: Vec<(usize, usize)> = (0..total_nodes.saturating_sub(1))
                .filter(|&i| points_3d[i].1 > 0.01 && points_3d[i + 1].1 > 0.01)
                .map(|i| (i, i + 1))
                .collect();

            // Find seed index
            let seed_idx = node_ids.iter().position(|&id| id == active_hash);

            let scene = ascii_3d::render_graph_scene(60, 25, &graph_points, &edges, frame, seed_idx);

            // Print frame
            print!("\x1B[2J\x1B[H");
            let active_count = all_acts.iter().filter(|&&a| a > 0.01).count();
            println!("  ╔═══ Hurricane 3D ═══ seed:{} active:{} tot:{} ═══",
                thinker.label(active_hash), active_count, total_nodes);
            println!("{}", scene);
            println!("  ───────────────────────────────────────────");
            print!("  > ");
            io::stdout().flush().ok();

            // Read query
            input.clear();
            if stdin.read_line(&mut input).ok().is_none() { break; }
            let trimmed = input.trim();
            if trimmed.is_empty() || trimmed == "quit" || trimmed == "exit" { break; }

            let id = concept_hash(trimmed);
            let label = thinker.label(id);
            active_hash = if label == "?" {
                let matches: Vec<ConceptId> = all_words.iter()
                    .filter(|w| w.contains(trimmed) || trimmed.contains(w.as_str()))
                    .map(|w| concept_hash(w))
                    .collect();
                if matches.is_empty() { continue; }
                matches[0]
            } else { id };

            frame += 0.5;
        }
        return;
    }

    if run_coder {
        let _max_nodes = 10000;
        let graph = cuda_bridge::CudaGraph::new(10000, 256);
        let mut thinker = thinker::Thinker::new();
        let mut board = blackboard::Blackboard::new(500);
        let path = args.get(2).cloned().unwrap_or_else(|| "concept_data.tsv".to_string());
        let pairs = load_concept_tsv(&path, &graph, &mut thinker);
        println!("  Loaded {} pairs, {} concepts", pairs.len(), graph.node_count());
        println!("  Training...");
        train_concept_graph(&graph, &pairs, 2);

        // Post training observation
        board.post("Graph training completed", blackboard::EntryType::Fact, blackboard::Source::System, None, None);

        println!("  Assembling chain → AST → code pipeline...\n");
        let coder = match coder::Coder::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("  Coder init error: {}", e); return; }
        };
        let seed = concept_hash("Rust");
        let state = thinker::SystemState::new();
        let cuda_state = state.to_cuda();
        graph.reset_activations();
        graph.activate_chem(seed, 5, &cuda_state);
        graph.top_k(16);
        let activated = graph.get_top_activations(16);
        let chain: Vec<(u64, &str, f32)> = activated.iter()
            .map(|(id, sc)| (*id, thinker.label(*id), *sc))
            .collect();
        println!("  Graph chain:");
        for (_id, name, sc) in &chain {
            println!("    {}[{}]", name, sc);
        }
        println!();
        println!("  AST:");
        let ast = coder.ast_gen.from_chain(&chain);
        println!("    {}", ast.render_text(0));
        println!();
        println!("  Generated Rust:");
        match coder.templates.render_chain(&ast, "rust") {
            Ok(code) => println!("{}", code),
            Err(e) => eprintln!("  Template error: {}", e),
        }
        println!();
        println!("  Generated Python:");
        match coder.templates.render_chain(&ast, "python") {
            Ok(code) => println!("{}", code),
            Err(e) => eprintln!("  Template error: {}", e),
        }
        println!();
        println!("  Description:");
        match coder.templates.render_chain(&ast, "desc") {
            Ok(desc) => println!("{}", desc),
            Err(e) => eprintln!("  Template error: {}", e),
        }
        // Post Uran+Coder result to Blackboard
        board.post("Coder assembled Rust+Python templates from Uran graph chain", blackboard::EntryType::Fact, blackboard::Source::Coder, Some(seed), None);
        println!();
        // Interactive coder REPL with Blackboard
        println!("  ─── Coder Interactive ───");
        println!("  Type a concept name to generate code from its associations.");
        println!("  Type 'quit' to exit. Type 'board' to show Blackboard entries.");
        use std::io::{self, Write};
        let stdin = io::stdin();
        let mut input = String::new();
        let all_words: Vec<String> = thinker.concept_names();
        loop {
            print!("  > ");
            io::stdout().flush().ok();
            input.clear();
            if stdin.read_line(&mut input).ok().is_none() { break; }
            let trimmed = input.trim();
            if trimmed.is_empty() || trimmed == "quit" || trimmed == "exit" { break; }
            if trimmed == "board" {
                println!("  ─── Blackboard ───");
                for e in board.read_all().iter().rev().take(16) {
                    println!("  [{:?}/{:?}] {}", e.source, e.entry_type, e.text);
                }
                continue;
            }
            let id = concept_hash(trimmed);
            let label = thinker.label(id);
            let search_id = if label == "?" {
                let matches: Vec<ConceptId> = all_words.iter()
                    .filter(|w| w.contains(trimmed) || trimmed.contains(w.as_str()))
                    .map(|w| concept_hash(w))
                    .collect();
                if matches.is_empty() {
                    println!("  Unknown concept '{}'", trimmed);
                    continue;
                }
                println!("  (using '{}')", thinker.label(matches[0]));
                matches[0]
            } else { id };
            graph.reset_activations();
            let cuda_state = state.to_cuda();
            graph.activate_chem(search_id, 5, &cuda_state);
            graph.top_k(10);
            let activated = graph.get_top_activations(10);
            let chain: Vec<(u64, &str, f32)> = activated.iter()
                .map(|(id, sc)| (*id, thinker.label(*id), *sc))
                .collect();
            // Post to Blackboard
            board.post(&trimmed, blackboard::EntryType::Command, blackboard::Source::User, Some(search_id), None);
            println!("  Chain:");
            let node = coder.ast_gen.from_chain(&chain);
            println!("  AST: {}", node.render_text(0));
            println!("  Rust:");
            match coder.templates.render_chain(&node, "rust") {
                Ok(code) => { for line in code.lines() { println!("    {}", line); } }
                Err(e) => eprintln!("    Template error: {}", e),
            }
            println!("  Python:");
            let python_code = match coder.templates.render_chain(&node, "python") {
                Ok(code) => { for line in code.lines() { println!("    {}", line); } Some(code) }
                Err(e) => { eprintln!("    Template error: {}", e); None }
            };
            board.post(&format!("Coder: generated code for '{}'", trimmed), blackboard::EntryType::Answer, blackboard::Source::Coder, Some(search_id), None);

            // ── Execute generated Python ──────────────────────────
            if let Some(ref code) = python_code {
                let exec_choice = if auto_exec {
                    "y".to_string()
                } else {
                    print!("  Execute Python? [Y/n/q]: ");
                    io::stdout().flush().ok();
                    let mut exec_input = String::new();
                    stdin.read_line(&mut exec_input).ok();
                    exec_input.trim().to_lowercase()
                };
                if exec_choice != "n" && exec_choice != "q" && exec_choice != "no" {
                    let exec = executor::Executor::new();
                    let result = if use_wasm {
                        println!("  ─── Executing in Wasm sandbox ───");
                        exec.run_wasm("compute.wasm", code)
                    } else {
                        println!("  ─── Executing Python ───");
                        exec.run_python(code)
                    };
                    if result.exit_code == 0 {
                        println!("  ✓ Exit: {}, Duration: {}ms", result.exit_code, result.duration_ms);
                        println!("  Output: {}", result.stdout.trim());
                        if !result.stderr.is_empty() {
                            println!("  stderr: {}", result.stderr.trim());
                        }

                        // ── Feedback: parse output → inject into graph ──
                        let output_tokens = executor::Executor::parse_output_to_tokens(&result.stdout);
                        if !output_tokens.is_empty() {
                            let out_concept_ids: Vec<ConceptId> = output_tokens.iter().map(|t| concept_hash(t)).collect();
                            println!("  → Injecting {} output tokens as concepts:", out_concept_ids.len());
                            for (tok, &cid) in output_tokens.iter().zip(out_concept_ids.iter()) {
                                let existing = thinker.label(cid);
                                if existing == "?" {
                                    graph.add_node(cid, tok, 0);
                                    thinker.add(cid, tok, graph::Modality::Text);
                                    println!("    + {} (new)", tok);
                                } else {
                                    println!("    · {} (exists)", existing);
                                }
                            }
                            // Link seed → output tokens via Hebbian
                            let mut learn_set: Vec<ConceptId> = vec![search_id];
                            learn_set.extend(&out_concept_ids);
                            let cuda_state2 = state.to_cuda();
                            graph.hebbian_learn(&learn_set, state.dopamin, state.adrenaline);
                            board.post(&format!("Exec output: {}", result.stdout.trim()), blackboard::EntryType::Fact, blackboard::Source::System, Some(search_id), None);

                            // Re-activate from first output token
                            if let Some(&first_out) = out_concept_ids.first() {
                                graph.reset_activations();
                                graph.activate_chem(first_out, 4, &cuda_state2);
                                let new_top = graph.get_top_activations(6);
                                println!("  → Now active from output:");
                                for (id, sc) in &new_top {
                                    println!("    {}[{}]", thinker.label(*id), sc);
                                }
                            }
                        }
                    } else {
                        println!("  ✗ Exit: {}, stderr: {}", result.exit_code, result.stderr.trim());
                        board.post(&format!("Exec error: {}", result.stderr.trim()), blackboard::EntryType::Fact, blackboard::Source::System, Some(search_id), None);
                    }
                }
                if exec_choice == "q" { break; }
            }
        }
        return;
    }

    if run_ascii_3d {
        println!("  ╔══════════════════════════════════════════════════╗");
        println!("  ║   3D ASCII Visualization                       ║");
        println!("  ╚══════════════════════════════════════════════════╝");
        println!();
        let width = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(60);
        let height = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(30);
        let frames = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(20);
        for frame in 0..frames {
            let angle = frame as f32 * 0.3;
            let scene = ascii_3d::render_scene_3d(width, height, angle);
            print!("\x1B[2J\x1B[H");
            println!("  Frame {}/{} (angle={:.1})", frame + 1, frames, angle);
            println!("{}", scene);
            std::thread::sleep(std::time::Duration::from_millis(150));
        }
        println!("  Done.");
        return;
    }

    if concept_data {
        let path = args.get(2).cloned().unwrap_or_else(|| "../concept_data.tsv".to_string());
        let slots = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(32);
        let num_epochs = args.iter()
            .position(|a| a == "--epochs")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);
        println!("  ╔══════════════════════════════════════════════════╗");
        println!("  ║   MUS Concept Graph — Semantic Pairs           ║");
        println!("  ║   GPU (CUDA) + HDC Pattern Memory              ║");
        println!("  ╚══════════════════════════════════════════════════╝");
        println!();
        run_concept_data(path, slots, num_epochs);
        return;
    }

    if train_real {
        let data_path = args.get(2)
            .cloned()
            .unwrap_or_else(|| "../data/train_cache.bin".to_string());
        let slots = args.get(3)
            .and_then(|s| s.parse().ok())
            .unwrap_or(256);

        // Parse --epochs N flag
        let num_epochs = args.iter()
            .position(|a| a == "--epochs")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);

        let vocab = 48000;

        if use_gpu {
            println!("  ╔══════════════════════════════════════════════════╗");
            println!("  ║   MUS Associative Core — Real Data Training   ║");
            println!("  ║   GPU (CUDA) + Bigram Hebbian                 ║");
            if use_chem {
                println!("  ║   Neurochemical System: ENABLED                ║");
            }
            println!("  ║   Slots/node: {}                              ║", slots);
            println!("  ║   Epochs:    {}                              ║", num_epochs);
            println!("  ╚══════════════════════════════════════════════════╝");
            println!();
            run_real_gpu(&data_path, slots, vocab, use_chem, num_epochs);
        } else {
            println!("  CPU real-data training not implemented; use GPU mode");
        }
        return;
    }

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
        if use_chem {
            println!("  ║   Neurochemical System: ENABLED                ║");
        }
        println!("  ║   Slots/node: {}                              ║", slots_per_node);
        println!("  ╚══════════════════════════════════════════════════╝");
        println!();
        run_gpu(&words, epochs, steps_per_epoch, slots_per_node, max_nodes);
    } else {
        println!("  ╔══════════════════════════════════════════════════╗");
        println!("  ║   MUS Associative Core — Hebbian Graph Engine  ║");
        println!("  ║   CPU (Rust)                                   ║");
        println!("  ║   Slots/node: {}                              ║", slots_per_node);
        println!("  ╚══════════════════════════════════════════════════╝");
        println!();
        run_cpu(&words, epochs, steps_per_epoch, slots_per_node, max_nodes);
    }
}
