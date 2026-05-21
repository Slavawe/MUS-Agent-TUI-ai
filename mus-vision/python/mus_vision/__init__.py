"""MUS Vision: ASCII art encoding/decoding, canvas rendering, shape generation.

Режимы кодирования:
- Photo: perceptual pipeline (gamma + BT.709 + sqrt quantization) — для фотографий
- Graph: edge detection (Sobel + linear quantization) — для схем, графов, диаграмм
"""

from mus_vision.config import ASCIIVisionConfig
from mus_vision.core import ASCIIVisionCore
