use crate::graph::{ConceptId, Graph};

pub struct HebbianLearner {
    pub cooldown: usize,
    step: usize,
}

impl HebbianLearner {
    pub fn new(cooldown: usize) -> Self {
        HebbianLearner { cooldown, step: 0 }
    }

    pub fn observe(&mut self, graph: &mut Graph, active_set: &[ConceptId]) -> usize {
        self.step += 1;
        if self.step % self.cooldown != 0 {
            return 0;
        }

        let mut links = 0;
        for i in 0..active_set.len() {
            for j in (i + 1)..active_set.len() {
                let a = active_set[i];
                let b = active_set[j];
                graph.link(a, b);
                links += 1;
            }
        }
        links
    }
}
