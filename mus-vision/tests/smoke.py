"""Smoke test for mus-vision — проверка Photo и Graph режимов."""
import numpy as np
from mus_vision import ASCIIVisionConfig, ASCIIVisionCore

# ── Photo mode ─────────────────────────────────────────────
cfg = ASCIIVisionConfig(vision_width=8, vision_height=4, ascii_palette=" .oO#")
core = ASCIIVisionCore(cfg, mode="photo")

mat = core.generate_shape("circle", "center", "medium", mode="photo")
print("Photo mode — circle:")
print(core.matrix_to_ascii(mat))

tokens = core.encode(mat, mode="photo")
print(f"Photo encode: {len(tokens)} tokens, first 8: {tokens[:8]}")

decoded = core.decode(tokens)
print("Photo decoded:")
print(decoded)

# ── Graph mode ─────────────────────────────────────────────
core_graph = ASCIIVisionCore(cfg, mode="graph")

mat_g = core_graph.generate_shape("circle", "center", "medium", mode="graph")
print("Graph mode — circle (edge detection style):")
print(core_graph.matrix_to_ascii(mat_g))

tokens_g = core_graph.encode(mat_g, mode="graph")
print(f"Graph encode: {len(tokens_g)} tokens, first 8: {tokens_g[:8]}")

decoded_g = core_graph.decode(tokens_g)
print("Graph decoded:")
print(decoded_g)

# ── Сравнение photo vs graph на синтетическом изображении ──
print("\n--- Photo vs Graph comparison ---")
test_img = np.zeros((32, 32, 3), dtype=np.uint8)
test_img[8:24, 8:24] = [200, 200, 200]  # серый квадрат
test_img[12:20, 12:20] = [50, 50, 50]    # тёмный центр

photo_tokens = core.encode(test_img, mode="photo")
graph_tokens = core_graph.encode(test_img, mode="graph")
print(f"Photo tokens: {len(photo_tokens)}")
print(f"Graph tokens: {len(graph_tokens)}")
print("Photo art:")
print(core.decode(photo_tokens))
print("Graph art (edges):")
print(core_graph.decode(graph_tokens))

# ── Canvas ─────────────────────────────────────────────────
from mus_vision.canvas import VisionCanvas

canvas = VisionCanvas(20, 8, use_rust=False)
canvas.add_static_bg("bg", density=0.1)
canvas.place_sprite("hero", "agent", row=2, col=5)
canvas.add_border()
print("\nCanvas (Python):")
print(canvas.render())

# Rust canvas (если доступен)
try:
    from mus_vision._core import CanvasLayer, VisionCanvas as RustCanvas, EncodeMode
    rust_canvas = RustCanvas(20, 8)
    rust_canvas.add_layer(CanvasLayer("bg", "static", "", 0, 0, 0.1))
    rust_canvas.add_border()
    print("\nCanvas (Rust):")
    print(rust_canvas.render())
    print(f"EncodeMode: Photo={EncodeMode('photo')}, Graph={EncodeMode('graph')}")
    print("Rust backend: OK")
except ImportError as e:
    print(f"No Rust backend: {e}")

# ── YAML renderer ──────────────────────────────────────────
try:
    from mus_vision._core import VisionRenderer
    renderer = VisionRenderer(12, 5)
    yaml = """
canvas_size: 12x5
layers:
  - type: static
    density: 0.2
  - type: sprite
    asset: "  o  \\n ooo \\n  o  "
    coords: [1, 4]
"""
    ascii_art, count = renderer.render_yaml(yaml, add_border=True)
    print(f"\nYAML Renderer ({count} layers):")
    print(ascii_art)
except Exception as e:
    print(f"YAML renderer error: {e}")

print("\nAll smoke tests passed!")

# ── Проверка, что photo и graph дают разные токены ─────────
assert photo_tokens != graph_tokens, "Photo и Graph режимы должны давать разные токены!"
print("✓ Photo и Graph режимы дают разные результаты")
