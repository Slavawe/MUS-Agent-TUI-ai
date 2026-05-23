use crate::graph::{ConceptId, Modality};
use rand::Rng;
use std::collections::HashMap;

pub struct Thinker {
    labels: HashMap<ConceptId, String>,
    modalities: HashMap<ConceptId, Modality>,
    assoc: HashMap<ConceptId, Vec<ConceptId>>,
}

impl Thinker {
    pub fn new() -> Self {
        Thinker { labels: HashMap::new(), modalities: HashMap::new(), assoc: HashMap::new() }
    }

    pub fn add(&mut self, id: ConceptId, label: &str, modality: Modality) {
        self.labels.insert(id, label.to_string());
        self.modalities.insert(id, modality);
    }

    pub fn set_assoc(&mut self, src: ConceptId, dst: ConceptId) {
        self.assoc.entry(src).or_default().push(dst);
        self.assoc.entry(dst).or_default().push(src);
    }

    pub fn label(&self, id: ConceptId) -> &str {
        self.labels.get(&id).map(|s| s.as_str()).unwrap_or("???")
    }

    fn modality_name(&self, id: ConceptId) -> &'static str {
        match self.modalities.get(&id) {
            Some(Modality::Text) => "слово",
            Some(Modality::Vision) => "образ",
            Some(Modality::Audio) => "звук",
            Some(Modality::Composite) => "понятие",
            None => "объект",
        }
    }

    pub fn think(&self, seed: ConceptId, depth: usize) -> Vec<String> {
        let mut lines = Vec::new();
        if !self.labels.contains_key(&seed) {
            lines.push("... я не знаю, о чём думать.".to_string());
            return lines;
        }

        // BFS walk
        let mut visited = vec![seed];
        let mut queue: Vec<ConceptId> = vec![seed];
        let mut prev: HashMap<ConceptId, ConceptId> = HashMap::new();
        prev.insert(seed, seed);

        for _ in 0..depth {
            let mut next: Vec<ConceptId> = Vec::new();
            for &cur in &queue {
                if let Some(nbrs) = self.assoc.get(&cur) {
                    for &nbr in nbrs {
                        if !visited.contains(&nbr) {
                            visited.push(nbr);
                            prev.insert(nbr, cur);
                            next.push(nbr);
                        }
                    }
                }
            }
            queue = next;
            if queue.is_empty() { break; }
        }

        // Build path: seed → ... → farthest
        let path = if visited.len() <= 1 {
            vec![seed]
        } else {
            let mut chain = vec![visited.last().copied().unwrap()];
            while *chain.last().unwrap() != seed {
                let cur = *chain.last().unwrap();
                if let Some(&p) = prev.get(&cur) {
                    chain.push(p);
                } else { break; }
            }
            chain.reverse();
            chain
        };

        // Render path into thoughts
        let label0 = self.label(seed);
        let mod0 = self.modality_name(seed);
        lines.push(format!("Я думаю о «{}» — это {}.", label0, mod0));

        for (i, &node) in path.iter().enumerate().skip(1) {
            let lbl = self.label(node);
            let prev_lbl = self.label(path[i - 1]);
            let m = self.modality_name(node);

            let templates = match (self.modalities.get(&node), self.modalities.get(&path[i - 1])) {
                (Some(Modality::Vision), Some(Modality::Text)) => vec![
                    format!("Образ «{}» возникает из слова «{}».", lbl, prev_lbl),
                    format!("Я представляю «{}» когда слышу «{}».", lbl, prev_lbl),
                ],
                (Some(Modality::Text), Some(Modality::Vision)) => vec![
                    format!("«{}» — это слово для образа «{}».", lbl, prev_lbl),
                ],
                (Some(Modality::Composite), _) => vec![
                    format!("«{}» объединяет несколько смыслов, связанных с «{}».", lbl, prev_lbl),
                ],
                _ => vec![
                    format!("«{}» связано с «{}».", lbl, prev_lbl),
                    format!("{} «{}» напоминает мне о «{}».", m, lbl, prev_lbl),
                ],
            };

            let tpl = &templates[path.len() % templates.len()];
            lines.push(tpl.clone());
        }

        if path.len() >= 3 {
            let a = self.label(path[path.len() - 1]);
            let b = self.label(path[path.len() / 2]);
            let last = self.label(seed);
            lines.push(format!("Цепочка: {} → {} → {}. Мысль завершена.", b, a, last));
        }

        lines
    }
}
