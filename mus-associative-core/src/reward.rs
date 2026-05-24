use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

const CACHE_FILE: &str = "train_cache.bin";

#[derive(Debug, Clone)]
pub struct RewardEntry {
    pub pattern: Vec<u64>,
    pub dopamine: f32,
    pub reward: f32,
    pub timestamp: u64,
}

pub struct RewardSystem {
    pub entries: Vec<RewardEntry>,
    pub total_reward: f32,
    pub total_penalty: f32,
    pub streak: u32,
}

fn read_u32(buf: &[u8], pos: &mut usize) -> u32 {
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&buf[*pos..*pos + 4]);
    *pos += 4;
    u32::from_le_bytes(bytes)
}

fn read_u64(buf: &[u8], pos: &mut usize) -> u64 {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&buf[*pos..*pos + 8]);
    *pos += 8;
    u64::from_le_bytes(bytes)
}

fn read_f32(buf: &[u8], pos: &mut usize) -> f32 {
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&buf[*pos..*pos + 4]);
    *pos += 4;
    f32::from_le_bytes(bytes)
}

impl RewardSystem {
    pub fn new() -> Self {
        let entries = Self::load_cache().unwrap_or_default();
        RewardSystem {
            entries,
            total_reward: 0.0,
            total_penalty: 0.0,
            streak: 0,
        }
    }

    pub fn reward(&mut self, pattern: &[u64], current_dopamin: f32) -> f32 {
        let boost = 0.15 * (1.0 + self.streak as f32 * 0.1).min(2.0);
        self.streak += 1;
        self.total_reward += boost;

        self.entries.push(RewardEntry {
            pattern: pattern.to_vec(),
            dopamine: current_dopamin,
            reward: 1.0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        });

        self.trim_cache();
        self.save_cache().ok();
        boost
    }

    pub fn penalize(&mut self, pattern: &[u64], current_dopamin: f32) -> f32 {
        let penalty = 0.25;
        self.streak = 0;
        self.total_penalty += penalty;

        self.entries.push(RewardEntry {
            pattern: pattern.to_vec(),
            dopamine: current_dopamin,
            reward: 0.0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        });

        self.trim_cache();
        self.save_cache().ok();
        -penalty
    }

    pub fn get_best_patterns(&self, n: usize) -> Vec<Vec<u64>> {
        let mut best: Vec<&RewardEntry> = self.entries.iter()
            .filter(|e| e.reward > 0.5)
            .collect();
        best.sort_by(|a, b| b.dopamine.partial_cmp(&a.dopamine).unwrap_or(std::cmp::Ordering::Equal));
        best.iter().take(n).map(|e| e.pattern.clone()).collect()
    }

    fn trim_cache(&mut self) {
        if self.entries.len() > 1000 {
            self.entries.drain(0..self.entries.len() - 1000);
        }
    }

    fn save_cache(&self) -> std::io::Result<()> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for e in &self.entries {
            buf.extend_from_slice(&(e.pattern.len() as u32).to_le_bytes());
            for &id in &e.pattern {
                buf.extend_from_slice(&id.to_le_bytes());
            }
            buf.extend_from_slice(&e.dopamine.to_le_bytes());
            buf.extend_from_slice(&e.reward.to_le_bytes());
            buf.extend_from_slice(&e.timestamp.to_le_bytes());
        }
        let mut f = File::create(CACHE_FILE)?;
        f.write_all(&buf)?;
        Ok(())
    }

    fn load_cache() -> std::io::Result<Vec<RewardEntry>> {
        let path = Path::new(CACHE_FILE);
        if !path.exists() {
            return Ok(Vec::new());
        }
        let mut f = File::open(path)?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        let len = buf.len();
        let mut pos = 0;
        if pos + 4 > len { return Ok(Vec::new()); }
        let num = read_u32(&buf, &mut pos) as usize;
        let mut entries = Vec::with_capacity(num);
        for _ in 0..num {
            if pos + 4 > len { break; }
            let plen = read_u32(&buf, &mut pos) as usize;
            if pos + plen * 8 > len { break; }
            let mut pattern = Vec::with_capacity(plen);
            for _ in 0..plen {
                pattern.push(read_u64(&buf, &mut pos));
            }
            if pos + 4 + 4 + 8 > len { break; }
            let dopamine = read_f32(&buf, &mut pos);
            let reward = read_f32(&buf, &mut pos);
            let timestamp = read_u64(&buf, &mut pos);
            entries.push(RewardEntry { pattern, dopamine, reward, timestamp });
        }
        Ok(entries)
    }
}
