from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple

try:
    from mus_vision._core import CanvasLayer as RustCanvasLayer
    from mus_vision._core import VisionCanvas as RustVisionCanvas
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


class LayerType(Enum):
    STATIC = "static"
    SPRITE = "sprite"
    OVERLAY = "overlay"
    BORDER = "border"


class SpriteAsset:
    library: Dict[str, SpriteAsset] = {}

    def __init__(self, name: str, lines: List[str]):
        self.name = name
        self.lines = lines

    @property
    def height(self) -> int:
        return len(self.lines)

    @property
    def width(self) -> int:
        return max(len(l) for l in self.lines) if self.lines else 0

    def to_matrix(self) -> List[List[str]]:
        return [list(line) for line in self.lines]

    @staticmethod
    def from_string(s: str) -> SpriteAsset:
        return SpriteAsset("custom", s.split("\n"))

    @staticmethod
    def register(name: str, asset: SpriteAsset):
        SpriteAsset.library[name] = asset

    @staticmethod
    def get(name: str) -> Optional[SpriteAsset]:
        return SpriteAsset.library.get(name)


# Predefined sprites
SpriteAsset.register("dot", SpriteAsset("dot", ["o"]))
SpriteAsset.register("star", SpriteAsset("star", [" * ", "***", " * "]))
SpriteAsset.register("cross", SpriteAsset("cross", ["  |  ", "--+--", "  |  "]))
SpriteAsset.register("smiley", SpriteAsset("smiley", [
    " ***** ",
    "*     *",
    "* o o *",
    "*  ^  *",
    "* \\_/ *",
    " ***** ",
]))
SpriteAsset.register("agent", SpriteAsset("agent", [
    "  O  ",
    " /|\\ ",
    " / \\ ",
]))


class VisionCanvas:
    def __init__(self, width: int = 40, height: int = 10, use_rust: bool = True):
        self.width = width
        self.height = height
        self._use_rust = use_rust and HAS_RUST
        self._layers: List[dict] = []

        if self._use_rust:
            self._rust = RustVisionCanvas(width, height)

    def add_layer(self, layer_type: str = "static", layer_id: str = "",
                  asset: str = "", row: int = 0, col: int = 0,
                  density: float = 0.2):
        if self._use_rust:
            rl = RustCanvasLayer(layer_id or f"layer_{len(self._layers)}",
                                 layer_type, asset, row, col, density)
            self._rust.add_layer(rl)
        self._layers.append({
            "id": layer_id or f"layer_{len(self._layers)}",
            "type": layer_type, "asset": asset,
            "row": row, "col": col, "density": density,
        })

    def add_static_bg(self, layer_id: str = "bg", density: float = 0.1):
        self.add_layer("static", layer_id, density=density)

    def place_sprite(self, layer_id: str, sprite_name: str,
                     row: int = 0, col: int = 0):
        sprite = SpriteAsset.get(sprite_name)
        if sprite is None:
            return False
        self.add_layer("sprite", layer_id, asset=sprite.name, row=row, col=col)
        return True

    def add_border(self):
        if self._use_rust:
            self._rust.add_border()
        self._layers.append({
            "id": "border", "type": "sprite",
            "asset": "__border__", "row": 0, "col": 0, "density": 0,
        })

    def remove_layer(self, layer_id: str) -> bool:
        if self._use_rust:
            self._rust.clear_layers()
            self._layers = [l for l in self._layers if l["id"] != layer_id]
            for l in self._layers:
                rl = RustCanvasLayer(l["id"], l["type"], l["asset"], l["row"], l["col"], l["density"])
                self._rust.add_layer(rl)
            return True
        self._layers = [l for l in self._layers if l["id"] != layer_id]
        return True

    def render(self) -> str:
        if self._use_rust and hasattr(self, '_rust'):
            return self._rust.render()
        return self._render_py()

    def _render_py(self) -> str:
        w, h = self.width, self.height
        grid = [[' ' for _ in range(w)] for __ in range(h)]

        DENSITY_CHARS = [' ', '.', ':', '-', '=', '+', '*', '%', '#', '@']

        for layer in sorted(self._layers, key=lambda l: 0 if l["type"] == "static" else 1):
            if layer["type"] == "static":
                idx = min(int(layer["density"] * (len(DENSITY_CHARS) - 1)), len(DENSITY_CHARS) - 1)
                ch = DENSITY_CHARS[idx]
                for row in grid:
                    for i in range(len(row)):
                        if row[i] == ' ':
                            row[i] = ch
            elif layer["type"] == "sprite":
                if layer["asset"] == "__border__":
                    self._draw_border(grid)
                else:
                    sprite = SpriteAsset.get(layer["asset"])
                    if sprite:
                        for r, line in enumerate(sprite.lines):
                            abs_r = layer["row"] + r
                            if abs_r >= h:
                                break
                            for c, ch in enumerate(line):
                                abs_c = layer["col"] + c
                                if abs_c >= w:
                                    break
                                if ch != ' ':
                                    grid[abs_r][abs_c] = ch

        return "\n".join("".join(row) for row in grid)

    def _draw_border(self, grid: List[List[str]]):
        h, w = len(grid), len(grid[0])
        for i in range(w):
            grid[0][i] = '-'
            grid[h - 1][i] = '-'
        for i in range(h):
            grid[i][0] = '|'
            grid[i][w - 1] = '|'
        grid[0][0] = '+'
        grid[0][w - 1] = '+'
        grid[h - 1][0] = '+'
        grid[h - 1][w - 1] = '+'

    def to_vision_block(self) -> str:
        return f"<vision>\n{self.render()}\n</vision>"

    def tokenize_state(self) -> List[int]:
        return []
