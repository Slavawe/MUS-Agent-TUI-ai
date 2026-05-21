"""
Uragan 1.0 — C++ Vision Tokenizer. Python обёртка над Rust-ускорителем.

Zero-copy интеграция с numpy через PyBuffer.
Режимы:
- Photo: perceptual pipeline (gamma + BT.709 + sqrt quantization)
- Graph: edge detection pipeline (Sobel + linear quantization)
"""
from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from mus_vision._core import CPPTokenizer as _RustTokenizer
from mus_vision._core import EncodeMode as _EncodeMode


_MODE_MAP = {
    "photo": _EncodeMode("photo"),
    "graph": _EncodeMode("graph"),
}


class CPPTokenizer:
    """Быстрый C++ токенизатор на Rust с прямой интеграцией numpy."""

    def __init__(
        self,
        width: int = 64,
        height: int = 32,
        palette: Optional[str] = None,
        cpp_start: int = 2001,
        vision_start: int = 2101,
        vision_end: int = 2102,
        frame_sep: int = 2103,
        mode: str = "photo",
    ):
        self._rust = _RustTokenizer(
            width=width,
            height=height,
            palette=palette,
            cpp_start=cpp_start,
            vision_start=vision_start,
            vision_end=vision_end,
            frame_sep=frame_sep,
            mode=mode,
        )
        self.width = width
        self.height = height
        self._mode = mode

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str):
        self._rust.set_mode(mode)
        self._mode = mode

    def encode_image(self, image: np.ndarray, mode: Optional[str] = None) -> List[int]:
        """RGB (H,W,3) или grayscale (H,W) → C++ токены. Zero-copy через PyBuffer."""
        if image.ndim == 3 and image.shape[2] >= 3:
            rgb = image[..., :3]
        elif image.ndim == 2:
            rgb = image
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)

        h, w = rgb.shape[:2]
        pixels = rgb.tobytes()

        if mode is not None and mode != self._mode:
            old_mode = self._mode
            self.set_mode(mode)
            result = self._rust.encode_image(pixels, w, h)
            self.set_mode(old_mode)
            return result

        return self._rust.encode_image(pixels, w, h)

    def encode_image_from_file(self, path: str, mode: Optional[str] = None) -> List[int]:
        """Файл изображения → C++ токены."""
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return self.encode_image(np.array(img), mode=mode)

    def encode_video(self, frames: List[np.ndarray]) -> List[int]:
        """Список кадров → видео-токены."""
        flat_frames = []
        widths = []
        heights = []
        for frame in frames:
            if frame.ndim == 3 and frame.shape[2] >= 3:
                rgb = frame[..., :3]
            elif frame.ndim == 2:
                rgb = frame
            else:
                raise ValueError(f"Unsupported frame shape: {frame.shape}")
            if not rgb.flags["C_CONTIGUOUS"]:
                rgb = np.ascontiguousarray(rgb)
            if rgb.dtype != np.uint8:
                rgb = (rgb.clip(0, 1) * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
            h, w = rgb.shape[:2]
            flat_frames.append(rgb.tobytes())
            widths.append(w)
            heights.append(h)
        return self._rust.encode_video(flat_frames, widths, heights)

    def encode_video_diff(self, frames: List[np.ndarray], threshold: float = 0.1) -> List[int]:
        """Видео с дельта-кодированием."""
        flat_frames = []
        widths = []
        heights = []
        for frame in frames:
            if frame.ndim == 3 and frame.shape[2] >= 3:
                rgb = frame[..., :3]
            elif frame.ndim == 2:
                rgb = frame
            else:
                raise ValueError(f"Unsupported frame shape: {frame.shape}")
            if not rgb.flags["C_CONTIGUOUS"]:
                rgb = np.ascontiguousarray(rgb)
            if rgb.dtype != np.uint8:
                rgb = (rgb.clip(0, 1) * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
            h, w = rgb.shape[:2]
            flat_frames.append(rgb.tobytes())
            widths.append(w)
            heights.append(h)
        return self._rust.encode_video_diff(flat_frames, widths, heights, threshold)

    def decode(self, tokens: List[int]) -> str:
        """Токены → арт строка."""
        return self._rust.decode([int(t) for t in tokens])

    def decode_video(self, tokens: List[int]) -> List[str]:
        """Видео-токены → список арт кадров."""
        return self._rust.decode_video([int(t) for t in tokens])

    def generate_shape(self, shape: str = "circle", position: str = "center", size: str = "medium") -> List[int]:
        """Процедурная фигура → токены."""
        return self._rust.generate_shape(shape, position, size)

    def info(self) -> str:
        return self._rust.info()
