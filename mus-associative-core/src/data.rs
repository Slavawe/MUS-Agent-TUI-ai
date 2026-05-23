use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

pub struct TokenDataset {
    mmap: Mmap,
    pub num_samples: usize,
    pub seq_len: usize,
}

impl TokenDataset {
    pub fn open(path: &str, seq_len: usize) -> std::io::Result<Self> {
        let file = File::open(Path::new(path))?;
        let mmap = unsafe { Mmap::map(&file)? };
        let token_bytes = 4; // i32
        let total_tokens = mmap.len() / token_bytes;
        let num_samples = total_tokens / seq_len;
        Ok(TokenDataset { mmap, num_samples, seq_len })
    }

    pub fn get_sample(&self, idx: usize) -> &[i32] {
        let offset = idx * self.seq_len;
        unsafe {
            std::slice::from_raw_parts(
                self.mmap.as_ptr().add(offset * 4) as *const i32,
                self.seq_len,
            )
        }
    }

    pub fn pairs(&self) -> PairIter<'_> {
        PairIter { data: self, sample_idx: 0, pair_idx: 0 }
    }
}

pub struct PairIter<'a> {
    data: &'a TokenDataset,
    sample_idx: usize,
    pair_idx: usize,
}

impl<'a> Iterator for PairIter<'a> {
    type Item = (u64, u64);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.sample_idx >= self.data.num_samples {
                return None;
            }
            let sample = self.data.get_sample(self.sample_idx);
            if self.pair_idx + 1 < self.data.seq_len {
                let a = sample[self.pair_idx] as u64;
                let b = sample[self.pair_idx + 1] as u64;
                self.pair_idx += 1;
                return Some((a, b));
            }
            self.sample_idx += 1;
            self.pair_idx = 0;
        }
    }
}
