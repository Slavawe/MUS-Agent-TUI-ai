from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ASCIIVisionConfig:
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    max_seq_len: int = 8192
    dropout: float = 0.1
    vocab_size: int = 25000
    vision_width: int = 64
    vision_height: int = 32
    ascii_palette: str = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#8&@%"
    ascii_tokens_start: int = 2001
    vision_start_id: int = 2101
    vision_end_id: int = 2102
    vision_photo_id: int = 2104
    vision_graph_id: int = 2105
    bos_id: int = 2001
    eos_id: int = 2002

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def palette_len(self) -> int:
        return len(self.ascii_palette)

    @property
    def ascii_tokens_end(self) -> int:
        return self.ascii_tokens_start + self.palette_len
