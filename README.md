![Uragan Logo]( assets/MUS.jpg)

# Uragan 1.0 — C++ CUDA Transformer

**English** · [Русский](#русский) · [中文](#中文) · [Español](#español)

---

<a id="english"></a>

## 🇬🇧 English

Uragan 1.0 is a production-grade multimodal transformer trained on consumer GPUs.

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Kernels** | C++17 / CUDA 12.5 | Custom FP32/FP16 kernels, cuBLAS GEMM |
| **Training** | CUDA C++ | Weighted CE, AdamW, RoPE, SwiGLU, RMSNorm |
| **Vision** | Rust + PyO3 | Image → C++ Token encoding (Photo/Graph modes) |
| **Python** | Python 3.11+ | NumPy, Pillow, PyTorch 2.0+ |
| **Build** | CMake + Maturin | Cross-platform compilation |
| **Cloud** | Kaggle / Colab | Automated training pipeline |
| **Storage** | Hugging Face Hub | Weight distribution |
| **CI/CD** | GitHub Actions | Build, lint, test, release |
| **Serialization** | Serde / YAML | Config & canvas rendering |

### Architecture

```
┌─────────────────────────────────────────────┐
│  Python (mus_vision)  ◄── PyO3 ── Rust     │
│  ├─ CPPVisionConfig                        │
│  ├─ CPPVisionCore    (Photo / Graph)        │
│  └─ VisionCanvas     (Layers / Sprites)     │
├─────────────────────────────────────────────┤
│  C++ / CUDA (mus-cuda)                      │
│  ├─ CPPTokenizer     (Vision → tokens)      │
│  ├─ MUSConfig        (Auto VRAM detect)     │
│  ├─ FP16 Kernels     (Attention, MLP, CE)   │
│  └─ Trainers         (400M / 700M / 1B)     │
├─────────────────────────────────────────────┤
│  Scripts & Cloud                            │
│  ├─ Kaggle / Colab pipelines               │
│  └─ Hugging Face upload                     │
└─────────────────────────────────────────────┘
```

### Model Presets

| Model | Params | VRAM | GPU |
|-------|--------|------|-----|
| 400M | 376M | 3.4 GB | GTX 1060, 1660 Ti |
| 500M | 498M | 4.1 GB | RTX 3050, 4050 |
| 700M | 685M | 5.8 GB | RTX 3060, 4060, T4 |
| 1B | 1.0B | 7.8 GB | RTX 4090, A100 |

### Token Ranges

| Range | Type | Description |
|-------|------|-------------|
| 0–2000 | AER | Audio / Event / Registry entities |
| 2001–2100 | CPP | C++ Vision tokens (palette encoding) |
| 2101–2200 | MM | Multimodal tags (vision, audio, system) |
| 2201–48000 | BPE | Text vocabulary |

---

<a id="русский"></a>

## 🇷🇺 Русский

Uragan 1.0 — production-grade мультимодальный трансформер, обучаемый на потребительских GPU.

### Технологический стек

| Слой | Технология | Назначение |
|------|-----------|-----------|
| **Ядра** | C++17 / CUDA 12.5 | Собственные FP32/FP16 ядра, cuBLAS GEMM |
| **Обучение** | CUDA C++ | Weighted CE, AdamW, RoPE, SwiGLU, RMSNorm |
| **Зрение** | Rust + PyO3 | Изображение → C++ токены (Photo/Graph) |
| **Python** | Python 3.11+ | NumPy, Pillow, PyTorch 2.0+ |
| **Сборка** | CMake + Maturin | Кроссплатформенная компиляция |
| **Облако** | Kaggle / Colab | Автоматический пайплайн обучения |
| **Хранилище** | Hugging Face Hub | Распространение весов |
| **CI/CD** | GitHub Actions | Сборка, линтинг, тесты, релизы |
| **Сериализация** | Serde / YAML | Конфиги и рендеринг canvas |

### Архитектура

```
┌─────────────────────────────────────────────┐
│  Python (mus_vision)  ◄── PyO3 ── Rust     │
│  ├─ CPPVisionConfig                        │
│  ├─ CPPVisionCore    (Photo / Graph)        │
│  └─ VisionCanvas     (Слои / Спрайты)       │
├─────────────────────────────────────────────┤
│  C++ / CUDA (mus-cuda)                      │
│  ├─ CPPTokenizer     (Зрение → токены)      │
│  ├─ MUSConfig        (Автоопределение VRAM) │
│  ├─ FP16 Kernels     (Attention, MLP, CE)   │
│  └─ Trainers         (400M / 700M / 1B)     │
├─────────────────────────────────────────────┤
│  Скрипты и облако                           │
│  ├─ Kaggle / Colap пайплайны               │
│  └─ Загрузка на Hugging Face               │
└─────────────────────────────────────────────┘
```

### Пресеты моделей

| Модель | Параметры | VRAM | GPU |
|--------|-----------|------|-----|
| 400M | 376M | 3.4 GB | GTX 1060, 1660 Ti |
| 500M | 498M | 4.1 GB | RTX 3050, 4050 |
| 700M | 685M | 5.8 GB | RTX 3060, 4060, T4 |
| 1B | 1.0B | 7.8 GB | RTX 4090, A100 |

### Диапазоны токенов

| Диапазон | Тип | Описание |
|----------|-----|----------|
| 0–2000 | AER | Сущности аудио/событий/реестра |
| 2001–2100 | CPP | C++ токены зрения (палитра) |
| 2101–2200 | MM | Мультимодальные теги |
| 2201–48000 | BPE | Текстовый словарь |

---

<a id="中文"></a>

## 🇨🇳 中文

Uragan 1.0 是一个可在消费级 GPU 上训练的生产级多模态 transformer。

### 技术栈

| 层 | 技术 | 用途 |
|-----|---------|---------|
| **内核** | C++17 / CUDA 12.5 | 自定义 FP32/FP16 内核，cuBLAS GEMM |
| **训练** | CUDA C++ | 加权 CE、AdamW、RoPE、SwiGLU、RMSNorm |
| **视觉** | Rust + PyO3 | 图像 → C++ 令牌编码 (Photo/Graph 模式) |
| **Python** | Python 3.11+ | NumPy、Pillow、PyTorch 2.0+ |
| **构建** | CMake + Maturin | 跨平台编译 |
| **云** | Kaggle / Colab | 自动化训练流水线 |
| **存储** | Hugging Face Hub | 权重分发 |
| **CI/CD** | GitHub Actions | 构建、检查、测试、发布 |
| **序列化** | Serde / YAML | 配置和 canvas 渲染 |

### 架构

```
┌─────────────────────────────────────────────┐
│  Python (mus_vision)  ◄── PyO3 ── Rust     │
│  ├─ CPPVisionConfig                        │
│  ├─ CPPVisionCore    (Photo / Graph)        │
│  └─ VisionCanvas     (图层 / 精灵)          │
├─────────────────────────────────────────────┤
│  C++ / CUDA (mus-cuda)                      │
│  ├─ CPPTokenizer     (视觉 → 令牌)          │
│  ├─ MUSConfig        (自动 VRAM 检测)       │
│  ├─ FP16 Kernels     (Attention, MLP, CE)   │
│  └─ Trainers         (400M / 700M / 1B)     │
├─────────────────────────────────────────────┤
│  脚本与云                                   │
│  ├─ Kaggle / Colab 流水线                  │
│  └─ Hugging Face 上传                       │
└─────────────────────────────────────────────┘
```

### 模型预设

| 模型 | 参数 | 显存 | GPU |
|------|------|------|-----|
| 400M | 3.76亿 | 3.4 GB | GTX 1060, 1660 Ti |
| 500M | 4.98亿 | 4.1 GB | RTX 3050, 4050 |
| 700M | 6.85亿 | 5.8 GB | RTX 3060, 4060, T4 |
| 1B | 10亿 | 7.8 GB | RTX 4090, A100 |

### 令牌范围

| 范围 | 类型 | 描述 |
|------|------|------|
| 0–2000 | AER | 音频/事件/注册实体 |
| 2001–2100 | CPP | C++ 视觉令牌（调色板编码） |
| 2101–2200 | MM | 多模态标签 |
| 2201–48000 | BPE | 文本词汇表 |

---

<a id="español"></a>

## 🇪🇸 Español

Uragan 1.0 es un transformer multimodal de grado de producción entrenable en GPUs de consumo.

### Stack Tecnológico

| Capa | Tecnología | Propósito |
|------|-----------|-----------|
| **Kernels** | C++17 / CUDA 12.5 | Kernels FP32/FP16 personalizados, cuBLAS GEMM |
| **Entrenamiento** | CUDA C++ | CE ponderada, AdamW, RoPE, SwiGLU, RMSNorm |
| **Visión** | Rust + PyO3 | Imagen → tokens C++ (modos Photo/Graph) |
| **Python** | Python 3.11+ | NumPy, Pillow, PyTorch 2.0+ |
| **Build** | CMake + Maturin | Compilación multiplataforma |
| **Nube** | Kaggle / Colab | Pipeline automatizado de entrenamiento |
| **Almacenamiento** | Hugging Face Hub | Distribución de pesos |
| **CI/CD** | GitHub Actions | Build, lint, test, release |
| **Serialización** | Serde / YAML | Config y renderizado de canvas |

### Arquitectura

```
┌─────────────────────────────────────────────┐
│  Python (mus_vision)  ◄── PyO3 ── Rust     │
│  ├─ CPPVisionConfig                        │
│  ├─ CPPVisionCore    (Photo / Graph)        │
│  └─ VisionCanvas     (Capas / Sprites)      │
├─────────────────────────────────────────────┤
│  C++ / CUDA (mus-cuda)                      │
│  ├─ CPPTokenizer     (Visión → tokens)      │
│  ├─ MUSConfig        (Detección VRAM auto)  │
│  ├─ FP16 Kernels     (Attention, MLP, CE)   │
│  └─ Trainers         (400M / 700M / 1B)     │
├─────────────────────────────────────────────┤
│  Scripts y Nube                             │
│  ├─ Pipelines Kaggle / Colab               │
│  └─ Subida a Hugging Face                  │
└─────────────────────────────────────────────┘
```

### Preajustes de Modelo

| Modelo | Parámetros | VRAM | GPU |
|--------|-----------|------|-----|
| 400M | 376M | 3.4 GB | GTX 1060, 1660 Ti |
| 500M | 498M | 4.1 GB | RTX 3050, 4050 |
| 700M | 685M | 5.8 GB | RTX 3060, 4060, T4 |
| 1B | 1.0B | 7.8 GB | RTX 4090, A100 |

### Rangos de Tokens

| Rango | Tipo | Descripción |
|-------|------|-------------|
| 0–2000 | AER | Entidades de audio/eventos/registro |
| 2001–2100 | CPP | Tokens C++ de visión (codificación de paleta) |
| 2101–2200 | MM | Etiquetas multimodales |
| 2201–48000 | BPE | Vocabulario de texto |

---

### Quick Start

```bash
git clone https://github.com/Slavawe/MUS-Agent-TUI-ai.git
cd MUS-Agent-TUI-ai/mus-cuda
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./build/mus_train_4_6gb
```

### License

MIT / Apache 2.0 (Dual)
