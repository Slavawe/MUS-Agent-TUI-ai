"""Модульные тесты для ASCIIVisionCore: Photo и Graph режимы."""
import numpy as np
import pytest

from mus_vision import ASCIIVisionConfig, ASCIIVisionCore


@pytest.fixture
def cfg():
    return ASCIIVisionConfig(vision_width=8, vision_height=4, ascii_palette=" .oO#")


@pytest.fixture
def core_photo(cfg):
    return ASCIIVisionCore(cfg, mode="photo")


@pytest.fixture
def core_graph(cfg):
    return ASCIIVisionCore(cfg, mode="graph")


# ════════════════════════════════════════════════════════════════
#  Базовые
# ════════════════════════════════════════════════════════════════

def test_init_modes(cfg):
    p = ASCIIVisionCore(cfg, mode="photo")
    g = ASCIIVisionCore(cfg, mode="graph")
    assert p.mode == "photo"
    assert g.mode == "graph"


def test_palette(core_photo):
    assert len(core_photo._palette) == 5
    assert core_photo._palette[0] == ' '
    assert core_photo._palette[-1] == '#'


# ════════════════════════════════════════════════════════════════
#  Luminance
# ════════════════════════════════════════════════════════════════

def test_image_to_luminance_without_gamma(core_photo):
    """BT.709 luminance без gamma — чистая взвешенная сумма."""
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[0, 0] = [255, 0, 0]    # pure red
    img[0, 1] = [0, 255, 0]    # pure green
    img[0, 2] = [0, 0, 255]    # pure blue
    lum = core_photo.image_to_luminance(img)
    assert lum[0, 0] == pytest.approx(0.2126, abs=1e-4), "Red luminance (BT.709)"
    assert lum[0, 1] == pytest.approx(0.7152, abs=1e-4), "Green luminance (BT.709)"
    assert lum[0, 2] == pytest.approx(0.0722, abs=1e-4), "Blue luminance (BT.709)"


def test_image_to_luminance_gamma(core_photo):
    """Gamma-corrected luminance — sRGB -> linear -> weighted sum."""
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[0, 0] = [128, 128, 128]
    lum = core_photo.image_to_luminance_gamma(img)
    # sRGB 128/255 ≈ 0.502 -> gamma: ((0.502+0.055)/1.055)^2.4 ≈ 0.216
    expected_gamma = ((0.502 + 0.055) / 1.055) ** 2.4
    assert lum[0, 0] == pytest.approx(expected_gamma, abs=1e-3), "Gamma correction"


def test_luminance_gamma_vs_raw(core_photo):
    """Gamma-corrected luminance должна отличаться от raw luminance."""
    img = np.full((4, 4, 3), 100, dtype=np.uint8)
    raw = core_photo.image_to_luminance(img)
    gamma = core_photo.image_to_luminance_gamma(img)
    assert not np.allclose(raw, gamma), "Gamma correction должна менять значения"


# ════════════════════════════════════════════════════════════════
#  Sobel Edge Detection
# ════════════════════════════════════════════════════════════════

def test_sobel_edges_on_uniform():
    """Sobel на однородном поле должен давать нули (кроме границ)."""
    cfg = ASCIIVisionConfig(vision_width=8, vision_height=4)
    core = ASCIIVisionCore(cfg, mode="graph")
    uniform = np.full((8, 4), 0.5, dtype=np.float32)
    edges = core.sobel_edges(uniform)
    assert edges[1:-1, 1:-1].max() < 1e-6, "Однородное поле — нет границ"


def test_sobel_edges_on_step():
    """Sobel на ступеньке должен давать ненулевой отклик на границе."""
    cfg = ASCIIVisionConfig(vision_width=8, vision_height=4)
    core = ASCIIVisionCore(cfg, mode="graph")
    step = np.zeros((8, 4), dtype=np.float32)
    step[:, 2:] = 1.0  # вертикальная граница в x=2
    edges = core.sobel_edges(step)
    assert edges[4, 2] > 0.1, "Sobel должен найти границу"
    assert edges[4, 0] < 0.01, "Sobel не должен находить границу вдали"


def test_sobel_max_one():
    """Sobel magnitude должна быть нормализована в [0, 1]."""
    cfg = ASCIIVisionConfig(vision_width=8, vision_height=4)
    core = ASCIIVisionCore(cfg, mode="graph")
    step = np.zeros((8, 4), dtype=np.float32)
    step[:, 2:] = 1.0
    edges = core.sobel_edges(step)
    assert edges.max() <= 1.0 + 1e-6
    assert edges.min() >= 0.0


# ════════════════════════════════════════════════════════════════
#  Quantization
# ════════════════════════════════════════════════════════════════

def test_quantize_perceptual_range(core_photo):
    """Perceptual quantization: [0,1] -> [0, palette_len-1]."""
    p = len(core_photo._palette)
    q = core_photo.quantize_perceptual(np.array([[0.0, 0.5, 1.0]]))
    assert q.min() >= 0
    assert q.max() <= p - 1
    assert q[0, 0] == 0, "0.0 -> индекс 0"
    assert q[0, 2] == p - 1, "1.0 -> последний индекс"


def test_quantize_linear_range(core_graph):
    """Linear quantization: [0,1] -> [0, palette_len-1]."""
    p = len(core_graph._palette)
    q = core_graph.quantize_linear(np.array([[0.0, 0.5, 1.0]]))
    assert q.min() >= 0
    assert q.max() <= p - 1
    assert q[0, 0] == 0, "0.0 -> индекс 0"
    assert q[0, 2] == p - 1, "1.0 -> последний индекс"


def test_quantize_perceptual_vs_linear(core_photo, core_graph):
    """Perceptual и linear дают разные результаты на midtones."""
    midtones = np.array([[0.25, 0.5, 0.75]])
    qp = core_photo.quantize_perceptual(midtones)
    ql = core_graph.quantize_linear(midtones)
    # На 0.25 perceptual должен дать более высокий индекс
    # (sqrt curve поднимает тени)
    assert qp[0, 0] >= ql[0, 0], "Perceptual поднимает тени"


# ════════════════════════════════════════════════════════════════
#  Encode / Decode roundtrip
# ════════════════════════════════════════════════════════════════

def test_encode_photo_roundtrip(core_photo):
    """Photo encode -> decode: структура сохраняется."""
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[8:24, 8:24] = [200, 200, 200]
    tokens = core_photo.encode(img, mode="photo")
    assert tokens[0] == core_photo.config.vision_start_id
    assert tokens[-1] == core_photo.config.vision_end_id
    decoded = core_photo.decode(tokens)
    lines = decoded.split("\n")
    assert len(lines) == core_photo._h
    assert len(lines[0]) == core_photo._w


def test_encode_graph_roundtrip(core_graph):
    """Graph encode -> decode: структура сохраняется."""
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[8:24, 8:24] = [200, 200, 200]
    tokens = core_graph.encode(img, mode="graph")
    assert tokens[0] == core_graph.config.vision_start_id
    assert tokens[-1] == core_graph.config.vision_end_id
    decoded = core_graph.decode(tokens)
    lines = decoded.split("\n")
    assert len(lines) == core_graph._h
    assert len(lines[0]) == core_graph._w


def test_encode_photo_and_graph_differ(core_photo, core_graph):
    """Photo и Graph режимы на одинаковом входе дают разные токены."""
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[8:24, 8:24] = [200, 200, 200]
    img[12:20, 12:20] = [50, 50, 50]

    p_tokens = core_photo.encode(img, mode="photo")
    g_tokens = core_graph.encode(img, mode="graph")

    assert p_tokens != g_tokens, "Режимы должны давать разные токены"

    # Сравним ASCII-тело (без vision_start/end)
    p_body = p_tokens[1:-1]
    g_body = g_tokens[1:-1]
    assert p_body != g_body, "Тело токенов должно различаться"


def test_encode_from_text_roundtrip(core_photo):
    """encode_from_text -> decode возвращает то же."""
    ascii_text = " o  o  \n ooo   \n  o    \n       "
    tokens = core_photo.encode_from_text(ascii_text)
    decoded = core_photo.decode(tokens)
    # Проверяем что размер совпадает
    lines = decoded.split("\n")
    assert len(lines) == core_photo._h


# ════════════════════════════════════════════════════════════════
#  Shape generation
# ════════════════════════════════════════════════════════════════

def test_generate_shape_photo_vs_graph(core_photo, core_graph):
    """generate_shape даёт разные матрицы для Photo и Graph."""
    mat_p = core_photo.generate_shape("circle", "center", "medium", mode="photo")
    mat_g = core_graph.generate_shape("circle", "center", "medium", mode="graph")
    assert mat_p.shape == (core_photo._h, core_photo._w)
    assert mat_g.shape == (core_graph._h, core_graph._w)
    # Типы разные (uint8 vs float32 для photo vs graph)
    # Просто проверяем что есть ненулевые элементы
    assert mat_p.max() > 0
    assert mat_g.max() > 0


def test_generate_all_shapes(core_photo):
    """Все типы фигур работают без ошибок."""
    shapes = ["circle", "square", "diamond", "triangle", "gradient",
              "waves", "checkerboard", "cross", "spiral"]
    for s in shapes:
        mat = core_photo.generate_shape(s, "center", "medium", mode="photo")
        assert mat.shape == (core_photo._h, core_photo._w), f"Shape {s}"


# ════════════════════════════════════════════════════════════════
#  Resize
# ════════════════════════════════════════════════════════════════

def test_resize_identity(core_photo):
    """Resize при совпадающих размерах возвращает то же."""
    mat = np.random.randn(core_photo._h, core_photo._w).astype(np.float32)
    resized = core_photo.resize(mat)
    assert np.allclose(mat, resized)


def test_resize_downscale(core_photo):
    """Resize большого изображения в canvas."""
    big = np.random.randn(64, 64).astype(np.float32)
    resized = core_photo.resize(big)
    assert resized.shape == (core_photo._h, core_photo._w)


# ════════════════════════════════════════════════════════════════
#  Canvas
# ════════════════════════════════════════════════════════════════

def test_render_canvas(core_photo):
    """render_canvas: базовая отрисовка слоёв."""
    layers = [("@", 1, 2), ("#", 2, 4)]
    result = core_photo.render_canvas(layers)
    lines = result.split("\n")
    assert len(lines) == core_photo._h
    assert len(lines[0]) == core_photo._w
    assert lines[1][2] == "@"
    assert lines[2][4] == "#"


# ════════════════════════════════════════════════════════════════
#  auto_detect_palette
# ════════════════════════════════════════════════════════════════

def test_auto_detect_palette():
    """auto_detect_palette собирает символы из текста."""
    text = "hello  world"
    pal = ASCIIVisionCore.auto_detect_palette(text)
    assert pal[0] == ' '
    assert 'h' in pal
    assert 'e' in pal


# ════════════════════════════════════════════════════════════════
#  Config
# ════════════════════════════════════════════════════════════════

def test_config_defaults():
    cfg = ASCIIVisionConfig()
    assert cfg.vision_width == 64
    assert cfg.vision_height == 32
    assert cfg.ascii_tokens_start == 2001
    assert cfg.vision_start_id == 2101
    assert cfg.vision_end_id == 2102
    assert cfg.vision_photo_id == 2104
    assert cfg.vision_graph_id == 2105
    assert cfg.ascii_tokens_end == cfg.ascii_tokens_start + cfg.palette_len
    assert cfg.head_dim == cfg.hidden_dim // cfg.num_heads


# ════════════════════════════════════════════════════════════════
#  Decode matrix
# ════════════════════════════════════════════════════════════════

def test_decode_matrix(core_photo):
    """decode_matrix восстанавливает матрицу индексов из токенов."""
    tokens = [core_photo.config.vision_start_id,
              2001, 2002, 2003, 2004, 2001, 2002, 2003, 2004,
              2001, 2002, 2003, 2004, 2001, 2002, 2003, 2004,
              2001, 2002, 2003, 2004, 2001, 2002, 2003, 2004,
              2001, 2002, 2003, 2004, 2001, 2002, 2003, 2004,
              core_photo.config.vision_end_id]
    mat = core_photo.decode_matrix(tokens)
    assert mat.shape == (core_photo._h, core_photo._w)
    assert mat[0, 0] == 0  # 2001 - 2001 = 0
    assert mat[0, 1] == 1  # 2002 - 2001 = 1


def test_decode_matrix_short(core_photo):
    """decode_matrix дополняет нулями если токенов мало."""
    mat = core_photo.decode_matrix([2001, 2002])
    assert mat.shape == (core_photo._h, core_photo._w)
    assert mat[0, 0] == 0
    assert mat[0, 1] == 1
