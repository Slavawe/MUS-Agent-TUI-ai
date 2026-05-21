"""Тесты для Python обёртки над Rust C++ токенизатором (если билд доступен).

Запуск: pytest tests/test_tokenizer.py
Если Rust не собран — тесты пропускаются.
"""
import numpy as np
import pytest

from mus_vision.cpp_tokenizer import CPPTokenizer

pytestmark = pytest.mark.skip(reason="Rust билд не собран на этой машине")


def test_init_defaults():
    tok = CPPTokenizer()
    assert tok.width == 64
    assert tok.height == 32
    assert tok.mode == "photo"


def test_init_graph():
    tok = CPPTokenizer(mode="graph")
    assert tok.mode == "graph"


def test_set_mode():
    tok = CPPTokenizer(mode="photo")
    assert tok.mode == "photo"
    tok.set_mode("graph")
    assert tok.mode == "graph"


def test_set_mode_invalid():
    tok = CPPTokenizer()
    with pytest.raises(ValueError):
        tok.set_mode("unknown")


def test_encode_photo_rgb():
    tok = CPPTokenizer(width=8, height=4, mode="photo")
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[2:6, 2:6] = [255, 255, 255]
    tokens = tok.encode_image(img)
    assert len(tokens) == 8 * 4 + 2
    assert tokens[0] == 2101
    assert tokens[-1] == 2102
    for t in tokens[1:-1]:
        assert 2001 <= t <= 2100, f"Токен {t} вне CPP диапазона"


def test_encode_graph_rgb():
    tok = CPPTokenizer(width=8, height=4, mode="graph")
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[2:6, 2:6] = [255, 255, 255]
    tokens = tok.encode_image(img)
    assert len(tokens) == 8 * 4 + 2
    assert tokens[0] == 2101
    assert tokens[-1] == 2102


def test_photo_and_graph_differ():
    tok_p = CPPTokenizer(width=8, height=4, mode="photo")
    tok_g = CPPTokenizer(width=8, height=4, mode="graph")
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[4:12, 4:12] = [200, 200, 200]
    p_tokens = tok_p.encode_image(img)
    g_tokens = tok_g.encode_image(img)
    assert p_tokens != g_tokens


def test_encode_with_mode_override():
    tok = CPPTokenizer(width=8, height=4, mode="photo")
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[2:6, 2:6] = [255, 255, 255]
    g_tokens = tok.encode_image(img, mode="graph")
    assert tok.mode == "photo"
    p_tokens = tok.encode_image(img, mode="photo")
    assert p_tokens != g_tokens


def test_encode_grayscale():
    tok = CPPTokenizer(width=8, height=4)
    gray = np.zeros((8, 8), dtype=np.uint8)
    gray[2:6, 2:6] = 200
    tokens = tok.encode_image(gray)
    assert len(tokens) == 8 * 4 + 2


def test_encode_from_file(tmp_path, core_photo):
    from PIL import Image
    path = tmp_path / "test.png"
    img = Image.fromarray(np.full((16, 16, 3), 150, dtype=np.uint8))
    img.save(path)
    tok = CPPTokenizer(width=8, height=4)
    tokens = tok.encode_image_from_file(str(path))
    assert len(tokens) == 8 * 4 + 2


def test_decode():
    tok = CPPTokenizer(width=4, height=2)
    tokens = [2101, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2102]
    art = tok.decode(tokens)
    lines = art.split("\n")
    assert len(lines) == 2
    assert len(lines[0]) == 4
    assert len(lines[1]) == 4


def test_generate_shape():
    tok = CPPTokenizer(width=8, height=4)
    tokens = tok.generate_shape("circle", "center", "medium")
    assert len(tokens) == 8 * 4 + 2
    assert tokens[0] == 2101
    assert tokens[-1] == 2102


def test_generate_shape_all():
    tok = CPPTokenizer(width=8, height=4)
    shapes = ["circle", "square", "diamond", "triangle", "gradient",
              "waves", "checkerboard", "cross", "spiral"]
    for s in shapes:
        tokens = tok.generate_shape(s, "center", "medium")
        assert len(tokens) == 8 * 4 + 2, f"Shape {s}"


def test_info():
    tok = CPPTokenizer(width=8, height=4, mode="photo")
    info = tok.info()
    assert "8x4" in info
    assert "photo" in info or "Photo" in info


def test_info_graph():
    tok = CPPTokenizer(width=8, height=4, mode="graph")
    info = tok.info()
    assert "graph" in info or "Graph" in info


def test_encode_video():
    tok = CPPTokenizer(width=4, height=2)
    frames = [
        np.full((4, 4, 3), 255, dtype=np.uint8),
        np.full((4, 4, 3), 0, dtype=np.uint8),
    ]
    tokens = tok.encode_video(frames)
    assert 2103 in tokens
    assert tokens[0] == 2101
    assert tokens[-1] == 2102


def test_decode_video():
    tok = CPPTokenizer(width=4, height=2)
    tokens = [2101, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008,
              2103,
              2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008,
              2102]
    frames = tok.decode_video(tokens)
    assert len(frames) == 2
    for f in frames:
        lines = f.split("\n")
        assert len(lines) == 2


def test_encode_video_diff():
    tok = CPPTokenizer(width=4, height=2)
    frames = [
        np.full((4, 4, 3), 255, dtype=np.uint8),
        np.full((4, 4, 3), 255, dtype=np.uint8),
    ]
    tokens = tok.encode_video_diff(frames, threshold=0.5)
    assert len(tokens) >= 2
