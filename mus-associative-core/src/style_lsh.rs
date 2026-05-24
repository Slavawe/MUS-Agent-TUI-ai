use std::collections::HashMap;

/// Locality-Sensitive Hashing for code style
/// Converts code style features into a compact binary fingerprint
#[derive(Debug, Clone)]
pub struct StyleFingerprint {
    /// Bit vector as u64 chunks (8 bytes each)
    pub bits: Vec<u64>,
    pub feature_count: usize,
}

impl StyleFingerprint {
    /// Compute Hamming distance (normalized 0.0 = identical, 1.0 =完全不同)
    pub fn distance(&self, other: &Self) -> f32 {
        let max_bits = self.bits.len().max(other.bits.len());
        if max_bits == 0 { return 0.0; }

        let mut diff = 0u64;
        let total_bits = max_bits * 64;

        for i in 0..self.bits.len().max(other.bits.len()) {
            let a = self.bits.get(i).copied().unwrap_or(0);
            let b = other.bits.get(i).copied().unwrap_or(0);
            diff += (a ^ b).count_ones() as u64;
        }

        diff as f32 / total_bits as f32
    }
}

pub struct StyleLSH {
    /// Random projection vectors for LSH
    projections: Vec<Vec<f32>>,
    dim: usize,
}

impl StyleLSH {
    pub fn new(dim: usize) -> Self {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let projections: Vec<Vec<f32>> = (0..dim).map(|_| {
            (0..128).map(|_| {
                if rng.gen::<f32>() > 0.5 { 1.0 } else { -1.0 }
            }).collect()
        }).collect();

        StyleLSH { projections, dim }
    }

    /// Analyze a code file and produce a style fingerprint
    pub fn analyze_code(&self, code: &str) -> StyleFingerprint {
        let features = self.extract_features(code);
        let bits = self.hash_features(&features);
        StyleFingerprint { bits, feature_count: features.len() }
    }

    /// Extract style features from code text
    fn extract_features(&self, code: &str) -> Vec<f32> {
        let mut features = Vec::new();

        let lines: Vec<&str> = code.lines().collect();
        if lines.is_empty() { return features; }

        // 1. Indentation: spaces vs tabs ratio
        let space_indent = lines.iter()
            .filter(|l| l.starts_with(' '))
            .count() as f32;
        let tab_indent = lines.iter()
            .filter(|l| l.starts_with('\t'))
            .count() as f32;
        features.push(if space_indent + tab_indent > 0.0 {
            space_indent / (space_indent + tab_indent)
        } else { 0.5 });

        // 2. Average line length
        let avg_len = lines.iter()
            .map(|l| l.len() as f32)
            .sum::<f32>() / lines.len().max(1) as f32;
        features.push((avg_len / 120.0).min(1.0));

        // 3. Brace style: same line vs new line
        let same_line = lines.iter()
            .filter(|l| l.contains('{') && !l.trim().ends_with('{') || l.ends_with(" {"))
            .count() as f32;
        let new_line = lines.iter()
            .filter(|l| l.trim() == "{")
            .count() as f32;
        features.push(if same_line + new_line > 0.0 {
            same_line / (same_line + new_line)
        } else { 0.5 });

        // 4. Comment density
        let comment_lines = lines.iter()
            .filter(|l| l.trim().starts_with("//") || l.trim().starts_with("#") || l.trim().starts_with("/*"))
            .count() as f32;
        features.push((comment_lines / lines.len().max(1) as f32).min(1.0));

        // 5. Empty line density
        let empty_lines = lines.iter()
            .filter(|l| l.trim().is_empty())
            .count() as f32;
        features.push((empty_lines / lines.len().max(1) as f32).min(1.0));

        // 6. Naming: snake_case vs camelCase ratio
        let snake = lines.iter()
            .filter(|l| l.contains('_'))
            .count() as f32;
        features.push((snake / lines.len().max(1) as f32).min(1.0));

        // 7. Semicolon density (C-style)
        let semi = code.bytes().filter(|&b| b == b';').count() as f32;
        features.push((semi / code.len().max(1) as f32 * 10.0).min(1.0));

        // 8. Paren density (function calls)
        let parens = code.bytes().filter(|&b| b == b'(' || b == b')').count() as f32;
        features.push((parens / code.len().max(1) as f32 * 10.0).min(1.0));

        features
    }

    /// Hash feature vector into binary fingerprint via random projections
    fn hash_features(&self, features: &[f32]) -> Vec<u64> {
        if features.is_empty() { return vec![0u64; 2]; }

        let mut bits = vec![0u64; self.dim / 64 + 1];

        for (i, proj) in self.projections.iter().enumerate() {
            let dot: f32 = features.iter()
                .zip(proj.iter())
                .map(|(f, p)| f * p)
                .sum();

            if dot > 0.0 {
                let word = i / 64;
                let bit = i % 64;
                if word < bits.len() {
                    bits[word] |= 1u64 << bit;
                }
            }
        }

        bits
    }
}

impl Default for StyleLSH {
    fn default() -> Self {
        Self::new(128)
    }
}
