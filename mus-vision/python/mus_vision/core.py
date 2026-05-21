from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from mus_vision.config import ASCIIVisionConfig


class ASCIIVisionCore:
    """Pure-Python ASCII Vision Core с поддержкой режимов Photo/Graph.

    Режимы:
    - Photo: gamma sRGB -> BT.709 luminance -> sqrt quantization
    - Graph: raw sRGB luminance -> Sobel edge detection -> linear quantization
    """

    def __init__(self, config: Optional[ASCIIVisionConfig] = None, mode: str = "photo"):
        self.config = config or ASCIIVisionConfig()
        self._palette = list(self.config.ascii_palette)
        self._w = self.config.vision_width
        self._h = self.config.vision_height
        self.mode = mode

        # Gamma LUT (sRGB -> linear)
        self._gamma_lut = np.zeros(256, dtype=np.float32)
        for i in range(256):
            v = i / 255.0
            self._gamma_lut[i] = v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    # ── Palette ───────────────────────────────────────────────

    @staticmethod
    def auto_detect_palette(text: str) -> str:
        chars = sorted(set(c for c in text if not c.isspace() or c == ' '))
        return " " + "".join(c for c in chars if c != ' ')

    def set_palette(self, palette: str):
        self._palette = list(palette)

    # ── Image processing ──────────────────────────────────────

    def image_to_luminance(self, image: np.ndarray) -> np.ndarray:
        """BT.709 luminance (без gamma correction)."""
        if image.ndim == 3 and image.shape[2] >= 3:
            gray = (0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]).astype(np.float32)
        elif image.ndim == 2:
            gray = image.astype(np.float32)
        else:
            raise ValueError(f"Bad shape: {image.shape}")
        if gray.max() > 1.0:
            gray /= 255.0
        return np.clip(gray, 0.0, 1.0)

    def image_to_luminance_gamma(self, image: np.ndarray) -> np.ndarray:
        """Gamma-corrected BT.709 luminance (Photo mode)."""
        if image.ndim == 3 and image.shape[2] >= 3:
            r = self._gamma_lut[image[..., 0].clip(0, 255).astype(np.uint8)]
            g = self._gamma_lut[image[..., 1].clip(0, 255).astype(np.uint8)]
            b = self._gamma_lut[image[..., 2].clip(0, 255).astype(np.uint8)]
            gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
        elif image.ndim == 2:
            gray = self._gamma_lut[image.clip(0, 255).astype(np.uint8)]
        else:
            raise ValueError(f"Bad shape: {image.shape}")
        return np.clip(gray, 0.0, 1.0)

    def resize(self, matrix: np.ndarray) -> np.ndarray:
        h, w = matrix.shape
        if h == self._h and w == self._w:
            return matrix
        rh, rw = h / self._h, w / self._w
        out = np.zeros((self._h, self._w), dtype=np.float32)
        for y in range(self._h):
            sy = min(int(y * rh), h - 2)
            fy = y * rh - sy
            for x in range(self._w):
                sx = min(int(x * rw), w - 2)
                fx = x * rw - sx
                v00 = matrix[sy, sx]
                v10 = matrix[min(sy + 1, h - 1), sx]
                v01 = matrix[sy, min(sx + 1, w - 1)]
                v11 = matrix[min(sy + 1, h - 1), min(sx + 1, w - 1)]
                out[y, x] = v00 * (1 - fy) * (1 - fx) + v10 * fy * (1 - fx) + v01 * (1 - fy) * fx + v11 * fy * fx
        return out

    def sobel_edges(self, matrix: np.ndarray) -> np.ndarray:
        """Sobel edge detection. Возвращает magnitude [0, 1]."""
        gx = np.zeros_like(matrix)
        gy = np.zeros_like(matrix)
        h, w = matrix.shape

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                patch = matrix[y - 1:y + 2, x - 1:x + 2]
                gx[y, x] = (
                    -patch[0, 0] + patch[0, 2]
                    -2 * patch[1, 0] + 2 * patch[1, 2]
                    -patch[2, 0] + patch[2, 2]
                )
                gy[y, x] = (
                    -patch[0, 0] - 2 * patch[0, 1] - patch[0, 2]
                    +patch[2, 0] + 2 * patch[2, 1] + patch[2, 2]
                )

        mag = np.sqrt(gx ** 2 + gy ** 2)
        max_mag = mag.max()
        if max_mag > 1e-6:
            mag /= max_mag
        return mag

    def quantize_perceptual(self, matrix: np.ndarray) -> np.ndarray:
        """Perceptual quantization (sqrt curve) — Photo mode."""
        p = len(self._palette)
        perceptual = np.sqrt(matrix.clip(0, 1))
        return np.clip(np.round(perceptual * (p - 1)), 0, p - 1).astype(np.int32)

    def quantize_linear(self, matrix: np.ndarray) -> np.ndarray:
        """Linear quantization — Graph mode."""
        p = len(self._palette)
        return np.clip(np.round(matrix * (p - 1)), 0, p - 1).astype(np.int32)

    # ── Encode ────────────────────────────────────────────────

    def encode(self, image: np.ndarray, mode: Optional[str] = None) -> List[int]:
        """Изображение -> ASCII-токены в зависимости от режима."""
        _mode = mode or self.mode

        if _mode == "photo":
            gray = self.image_to_luminance_gamma(image)
            resized = self.resize(gray)
            q = self.quantize_perceptual(resized)
        elif _mode == "graph":
            gray = self.image_to_luminance(image)
            resized = self.resize(gray)
            edges = self.sobel_edges(resized)
            q = self.quantize_linear(edges)
        else:
            raise ValueError(f"Unknown mode: {_mode}")

        return (
            [self.config.vision_start_id]
            + [self.config.ascii_tokens_start + int(q[y, x])
               for y in range(self._h) for x in range(self._w)]
            + [self.config.vision_end_id]
        )

    def encode_from_file(self, path: str, mode: Optional[str] = None) -> List[int]:
        img = np.array(Image.open(path).convert("RGB"))
        return self.encode(img, mode=mode)

    def encode_from_text(self, text: str) -> List[int]:
        lines = text.split("\n")
        mat = np.zeros((self._h, self._w), dtype=np.float32)
        for y in range(min(len(lines), self._h)):
            for x in range(min(len(lines[y]), self._w)):
                ch = lines[y][x]
                if ch in self._palette:
                    mat[y, x] = self._palette.index(ch) / max(1, len(self._palette) - 1)
        return self.encode(mat)

    # ── Decode ────────────────────────────────────────────────

    def decode_matrix(self, token_ids: List[int]) -> np.ndarray:
        start, end = self.config.ascii_tokens_start, self.config.ascii_tokens_end
        tokens = [t for t in token_ids if start <= t <= end]
        idx = [t - start for t in tokens]
        n = self._h * self._w
        if len(idx) < n:
            idx.extend([0] * (n - len(idx)))
        return np.array(idx[:n], dtype=np.int32).reshape(self._h, self._w)

    def decode(self, token_ids: List[int]) -> str:
        mat = self.decode_matrix(token_ids)
        return self.matrix_to_ascii(mat)

    def matrix_to_ascii(self, matrix: np.ndarray) -> str:
        lines = []
        for y in range(matrix.shape[0]):
            lines.append("".join(self._palette[int(matrix[y, x]) % len(self._palette)]
                                 for x in range(matrix.shape[1])))
        return "\n".join(lines)

    # ── Shape generation ──────────────────────────────────────

    def generate_shape(self, shape: str = "circle", position: str = "center",
                       size: str = "medium", mode: Optional[str] = None) -> np.ndarray:
        mat = np.zeros((self._h, self._w), dtype=np.float32)
        cx, cy = self._h // 2, self._w // 2
        pos_map = {"top-left": (self._h // 4, self._w // 4),
                   "top-right": (self._h // 4, 3 * self._w // 4),
                   "center": (cx, cy),
                   "bottom-left": (3 * self._h // 4, self._w // 4),
                   "bottom-right": (3 * self._h // 4, 3 * self._w // 4)}
        if position in pos_map:
            cy, cx = pos_map[position]
        size_map = {"small": 0.15, "medium": 0.3, "large": 0.45}
        rel_size = size_map.get(size, 0.3)

        if shape == "circle":
            radius = int(min(self._h, self._w) * rel_size)
            for y in range(self._h):
                for x in range(self._w):
                    d = math.sqrt((y - cy) ** 2 + (x - cx) ** 2)
                    mat[y, x] = max(0, 1 - d / radius) if d <= radius else 0
        elif shape == "square":
            side = int(min(self._h, self._w) * rel_size * 2)
            y0, x0 = max(0, cy - side // 2), max(0, cx - side // 2)
            y1, x1 = min(self._h, y0 + side), min(self._w, x0 + side)
            mat[y0:y1, x0:x1] = 1.0
        elif shape == "triangle":
            side = int(min(self._h, self._w) * rel_size * 2)
            for y in range(self._h):
                for x in range(self._w):
                    dy = (y - cy) / max(1, side // 2)
                    dx = (x - cx) / max(1, side // 2)
                    inside = dy >= -1 and dy <= 0 and abs(dx) <= (1 + dy)
                    mat[y, x] = 1.0 if inside else 0.0
        elif shape == "diamond":
            radius = int(min(self._h, self._w) * rel_size)
            for y in range(self._h):
                for x in range(self._w):
                    d = abs(y - cy) + abs(x - cx)
                    mat[y, x] = max(0, 1 - d / radius) if d <= radius else 0
        else:
            mat[cy, cx] = 1.0

        _mode = mode or self.mode
        if _mode == "photo":
            return self.quantize_perceptual(mat)
        else:
            return self.quantize_linear(mat)

    def random_shape(self, mode: Optional[str] = None) -> Tuple[np.ndarray, str, str, str]:
        shapes = ["circle", "square", "triangle", "diamond"]
        positions = ["center", "top-left", "top-right", "bottom-left", "bottom-right"]
        sizes = ["small", "medium", "large"]
        s = random.choice(shapes)
        p = random.choice(positions)
        z = random.choice(sizes)
        return self.generate_shape(s, p, z, mode=mode), s, p, z

    # ── Simple canvas render ──────────────────────────────────

    def render_canvas(self, layers: List[Tuple[str, int, int]]) -> str:
        grid = [[' ' for _ in range(self._w)] for __ in range(self._h)]
        for ch, row, col in layers:
            if 0 <= row < self._h and 0 <= col < self._w:
                grid[row][col] = ch
        return "\n".join("".join(row) for row in grid)
