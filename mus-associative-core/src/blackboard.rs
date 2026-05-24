use crate::thinker::ConceptId;
use std::collections::VecDeque;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntryType {
    Intent,
    Fact,
    Question,
    Answer,
    Command,
    Observation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    Uran,
    Coder,
    User,
    System,
}

#[derive(Debug, Clone)]
pub struct BlackboardEntry {
    pub id: u64,
    pub text: String,
    pub concept: Option<ConceptId>,
    pub entry_type: EntryType,
    pub source: Source,
    pub reply_to: Option<u64>,
    pub confidence: f32,
}

pub struct Blackboard {
    entries: VecDeque<BlackboardEntry>,
    max_entries: usize,
    next_id: u64,
}

impl Blackboard {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(max_entries),
            max_entries,
            next_id: 1,
        }
    }

    pub fn post(&mut self, text: &str, entry_type: EntryType, source: Source, concept: Option<ConceptId>, reply_to: Option<u64>) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.entries.push_back(BlackboardEntry {
            id,
            text: text.to_string(),
            concept,
            entry_type,
            source,
            reply_to,
            confidence: 1.0,
        });
        if self.entries.len() > self.max_entries {
            self.entries.pop_front();
        }
        id
    }

    pub fn read_all(&self) -> Vec<&BlackboardEntry> {
        self.entries.iter().collect()
    }

    pub fn filter(&self, entry_type: Option<&EntryType>, source: Option<&Source>, limit: usize) -> Vec<&BlackboardEntry> {
        self.entries.iter()
            .filter(|e| entry_type.map_or(true, |t| e.entry_type == *t))
            .filter(|e| source.map_or(true, |s| e.source == *s))
            .take(limit)
            .collect()
    }

    pub fn get(&self, id: u64) -> Option<&BlackboardEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    pub fn count(&self) -> usize {
        self.entries.len()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn recent_user_intents(&self, n: usize) -> Vec<&BlackboardEntry> {
        self.entries.iter()
            .filter(|e| e.source == Source::User && e.entry_type == EntryType::Intent)
            .rev()
            .take(n)
            .collect()
    }

    pub fn uran_facts(&self, n: usize) -> Vec<&BlackboardEntry> {
        self.entries.iter()
            .filter(|e| e.source == Source::Uran && e.entry_type == EntryType::Fact)
            .rev()
            .take(n)
            .collect()
    }
}
