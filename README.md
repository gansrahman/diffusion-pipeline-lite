# Diffusion Pipeline Lite

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A lightweight inference pipeline for **Stable Diffusion**, optimized for GPUs with limited VRAM (6-8GB). Designed for researchers and developers who need fast image generation without high-end hardware.

## Features

- **FP16 inference** with automatic memory management
- **Sequential CPU offloading** for low-VRAM devices
- **Attention slicing** for memory-efficient generation
- Supports **SD 1.5**, **SD 2.1**, and **SDXL** models
- Configurable guidance scale, steps, and seed
- Batch generation support

## Installation

```bash
git clone https://github.com/gansrahman/diffusion-pipeline-lite.git
cd diffusion-pipeline-lite
pip install -r requirements.txt
```

## Quick Start

```python
from generate import load_pipeline, generate

pipe = load_pipeline("stabilityai/stable-diffusion-2-1")
img = generate(pipe, "a futuristic city at sunset", steps=30)
img.save("output.png")
```

## CLI Usage

```bash
python generate.py --prompt "a futuristic city at sunset" --steps 30 --output city.png
python generate.py --prompt "anime character" --steps 50 --guidance 8.0 --seed 42
```

## Memory Optimization

For GPUs with 6GB VRAM:
```python
pipe = load_pipeline()
pipe.enable_model_cpu_offload()  # Offload to CPU when not in use
```

## Benchmarks

| GPU | Resolution | Steps | Time |
|-----|-----------|-------|------|
| RTX 3060 6GB | 512x512 | 20 | ~8s |
| RTX 4090 24GB | 512x512 | 20 | ~2s |
| A100 40GB | 512x512 | 20 | ~3s |

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ or ROCm 5.7+
- 6GB+ VRAM recommended

## License

MIT License - see [LICENSE](LICENSE) for details.
