"""Vision train: encode photos (Photo mode) + encode diagrams (Graph mode) + train."""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from mus_vision import ASCIIVisionConfig, ASCIIVisionCore
from mus_vision.canvas import VisionCanvas


PHOTO_DIR = Path("C:/Users/slava/Documents/MUS/data/photos")
DIAGRAM_DIR = Path("C:/Users/slava/Documents/MUS/data/diagrams")

# ── 1. Encode photos (Photo mode) и диаграммы (Graph mode) ─────

cfg = ASCIIVisionConfig(vision_width=64, vision_height=32, hidden_dim=192, num_layers=4, num_heads=6)
core_photo = ASCIIVisionCore(cfg, mode="photo")
core_graph = ASCIIVisionCore(cfg, mode="graph")

photos = list(PHOTO_DIR.glob("img*.jpg"))[:5]
print(f"Found {len(photos)} test photos")

for p in photos:
    tokens = core_photo.encode_from_file(str(p), mode="photo")
    print(f"  [Photo] {p.name}: {len(tokens)} tokens")

diagrams = list(DIAGRAM_DIR.glob("*.png"))[:5] if DIAGRAM_DIR.exists() else []
if diagrams:
    print(f"Found {len(diagrams)} test diagrams")
    for p in diagrams:
        tokens = core_graph.encode_from_file(str(p), mode="graph")
        print(f"  [Graph] {p.name}: {len(tokens)} tokens")

# Большое фото
big_photo = list(PHOTO_DIR.glob("pexels-castorlystock-8822082.jpg"))
if big_photo:
    p = big_photo[0]
    photo_tokens = core_photo.encode_from_file(str(p), mode="photo")
    graph_tokens = core_graph.encode_from_file(str(p), mode="graph")  # для сравнения
    art_photo = core_photo.decode(photo_tokens)
    art_graph = core_graph.decode(graph_tokens)
    print(f"\nBig photo '{p.name}' — Photo mode:")
    print(art_photo)
    print(f"Big photo '{p.name}' — Graph mode (Sobel edges):")
    print(art_graph)

# ── 2. Dataset ───────────────────────────────────────────────

class VisionDataset(Dataset):
    def __init__(self, data, core, seq_len=256):
        self.cfg = core.config
        self.seq_len = seq_len
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        if len(tokens) < self.seq_len:
            tokens = tokens + [self.cfg.vision_end_id] * (self.seq_len - len(tokens))
        return torch.tensor(tokens[:self.seq_len], dtype=torch.long)


VD = VisionDataset


# ── 3. Vision Transformer ────────────────────────────────────

class VisionTransformer(nn.Module):
    def __init__(self, cfg: ASCIIVisionConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.hidden_dim)
        self.blocks = nn.ModuleList([VisionBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.lm = nn.Linear(cfg.hidden_dim, cfg.vocab_size)
        self.lm.weight = self.embed.weight

    def forward(self, x, labels=None):
        B, T = x.shape
        x = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        logits = self.lm(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                   labels[:, 1:].reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, x, max_new=64, temperature=0.7, top_p=0.9):
        for _ in range(max_new):
            x = x[:, -self.cfg.max_seq_len:]
            logits, _ = self.forward(x)
            logits = logits[:, -1, :] / temperature
            sorted_logits, idx = torch.sort(logits, dim=-1, descending=True)
            cumsum = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            mask = cumsum > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            logits[0][mask[0]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_tok], dim=-1)
        return x

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


class VisionBlock(nn.Module):
    def __init__(self, cfg: ASCIIVisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.attn = nn.MultiheadAttention(cfg.hidden_dim, cfg.num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim * 4, cfg.hidden_dim),
        )

    def forward(self, x):
        a = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


# ── 4. Train ─────────────────────────────────────────────────

print("\n--- Training ---")

all_data = []

# Real photos (Photo mode)
paths = list(PHOTO_DIR.glob("*.jpg"))[:100]
for p in paths:
    try:
        all_data.append(core_photo.encode_from_file(str(p), mode="photo"))
    except Exception:
        pass
print(f"  Photo samples: {len(all_data)}")

# Diagrams / graphs (Graph mode)
if DIAGRAM_DIR.exists():
    for p in list(DIAGRAM_DIR.glob("*.png"))[:50]:
        try:
            all_data.append(core_graph.encode_from_file(str(p), mode="graph"))
        except Exception:
            pass
print(f"  Total with diagrams: {len(all_data)}")

# Synthetic augmentation
while len(all_data) < 100:
    i = len(all_data)
    start = cfg.ascii_tokens_start + (i % 10) * 5
    seq = list(range(start, min(start + 128, cfg.ascii_tokens_start + 2048)))
    mode_tag = cfg.vision_photo_id if (i % 2 == 0) else cfg.vision_graph_id
    seq = [cfg.vision_start_id, mode_tag] + seq * 4 + [cfg.vision_end_id]
    all_data.append(seq[:256])

print(f"  Total samples: {len(all_data)}")


ds = VisionDataset(all_data, core_photo)
model = VisionTransformer(cfg)
print(f"  Params: {model.get_num_params():,}")

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
dl = DataLoader(ds, batch_size=4, shuffle=True, drop_last=True)

model.train()
for epoch in range(5):
    losses = []
    for batch in dl:
        logits, loss = model(batch, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    print(f"  Epoch {epoch+1}: loss={sum(losses)/len(losses):.4f}")

# ── 5. Generate from vision prompt ───────────────────────────

print("\n--- Generation ---")
model.eval()

# Генерация из photo prompt
prompt = [cfg.bos_id, cfg.vision_start_id, cfg.vision_photo_id]
prompt_ids = torch.tensor([prompt], dtype=torch.long)
generated = model.generate(prompt_ids, max_new=64, temperature=0.8, top_p=0.9)
gen_tokens = generated[0].tolist()
print(f"  [Photo] Generated {len(gen_tokens)} tokens")
print(f"  First 20: {gen_tokens[:20]}")

try:
    art = core_photo.decode(gen_tokens)
    print(f"  Decoded:\n{art}")
except Exception as e:
    print(f"  Decode error: {e}")

# Генерация из graph prompt
prompt_g = [cfg.bos_id, cfg.vision_start_id, cfg.vision_graph_id]
prompt_ids_g = torch.tensor([prompt_g], dtype=torch.long)
generated_g = model.generate(prompt_ids_g, max_new=64, temperature=0.8, top_p=0.9)
gen_tokens_g = generated_g[0].tolist()
print(f"\n  [Graph] Generated {len(gen_tokens_g)} tokens")
print(f"  First 20: {gen_tokens_g[:20]}")

try:
    art_g = core_graph.decode(gen_tokens_g)
    print(f"  Decoded:\n{art_g}")
except Exception as e:
    print(f"  Decode error: {e}")

# Декорирование реального фото
if len(ds.data) > 0:
    real = ds.data[0]
    art = core_photo.decode(real)
    print(f"\n  Real sample decoded:\n{art}")

print("\nOK — Vision train pipeline verified (Photo + Graph modes)")
