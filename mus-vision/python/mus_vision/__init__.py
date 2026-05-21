"""Uragan 1.0 — C++ Vision: кодирование/декодирование, canvas, генерация фигур.

Режимы кодирования:
- Photo: perceptual pipeline (gamma + BT.709 + sqrt quantization) — для фотографий
- Graph: edge detection (Sobel + linear quantization) — для схем, графов, диаграмм
"""

from mus_vision.config import CPPVisionConfig
from mus_vision.core import CPPVisionCore
