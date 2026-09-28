# E-Commerce Product Image Standardization & Deep Inpainting Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0+-green.svg)](https://opencv.org/)
[![Pillow](https://img.shields.io/badge/Pillow-12.0+-orange.svg)](https://python-pillow.org/)
[![rembg](https://img.shields.io/badge/rembg-2.0+-red.svg)](https://github.com/danielgatis/rembg)
[![LaMa Inpainting](https://img.shields.io/badge/LaMa-Deep%20Inpainting-purple.svg)](https://github.com/advimman/lama)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An automated, production-ready Python image processing pipeline designed to ingest raw, catalog-extracted, or low-quality product images and output standardized, commercial e-commerce assets matching strict marketplace standards (e.g., Amazon, Shopify, Trendyol, Hepsiburada).

---

## 🚀 Key Features

- **Multi-Cue Catalog Sticker & Label Detection:** Fuses HSV color saturation and Canny high-frequency edge density with heavy morphological dilation ($25\times 25$ kernel) and safety padding to capture the entire sticker bounding box (eliminating green headers, energy rating bars, typography, QR codes, and borders).
- **SOTA Deep Surface Inpainting (LaMa):** Uses Large Mask Inpainting (LaMa) to reconstruct curved plastic chassis, panels, and surfaces with photorealistic lighting gradients. Features an automatic fallback to 2D vertical gradient interpolation and Telea edge blending.
- **Deep-Learning Background Isolation:** Employs salient object segmentation (`rembg` with `u2net`) with alpha threshold cleaning to remove faint shadows and catalog artifacts.
- **Strict Bounding Box & Aspect-Ratio Scaling:** Scales products proportionally (strictly preserving aspect ratio) to occupy an exact target fill ratio (default: `85%`) inside a square canvas (default: `1500x1500px`).
- **Precision Centering on Pure `#FFFFFF` Canvas:** Places the product at the mathematical center with balanced margins and 100% `#FFFFFF` (`255, 255, 255`) outer canvas purity.
- **White-on-White Edge Preservation:** Prevents white products (air conditioners, home appliances) from washing out into the white background via subtle boundary micro-contrast enhancement.
- **Batch & CLI Processing:** Seamlessly processes single files or entire folders with high-quality JPEG (`quality=98`, 4:4:4 subsampling) or lossless PNG export.

---

## 📐 Architecture & Pipeline Flow

```mermaid
flowchart TD
    A[Raw Catalog Image] --> B[Front-Panel ROI Extraction]
    B --> C1[HSV Saturation Mask]
    B --> C2[Canny Edge Detection for Text & QR]
    C1 & C2 --> D[Bitwise OR + 25x25 Morphological Dilation]
    D --> E[Full Sticker BBox + 8px Safety Padding]
    E --> F[LaMa Deep Inpainting / 2D Gradient Fallback]
    F --> G[Rembg Salient Object Segmentation]
    G --> H[Alpha Noise Cleaning & Tight Product BBox]
    H --> I[Proportional Scaling: 85% Fill via Lanczos]
    I --> J[White-on-White Edge Contrast & Sharpening]
    J --> K[Centering on 1500x1500 Pure #FFFFFF Canvas]
    K --> L[Export High-Quality JPEG / PNG]
```

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/alierentugrul/product-image-standardization.git
cd product-image-standardization
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 CLI Reference & Usage

### Standard E-Commerce Execution
```bash
python standardize_pipeline.py --input DoraNEwinverterAC_1.webp --output output/Dora_ECommerce_Perfect.jpg
```
*Or using `pipeline.py`:*
```bash
python pipeline.py --input DoraNEwinverterAC_1.webp --output output/Dora_ECommerce_Perfect.jpg --size 1500 --fill 0.85
```

### Directory Batch Execution
Process an entire folder of catalog images:
```bash
python pipeline.py --input ./raw_catalog --output ./standardized_catalog --size 1500 --fill 0.85
```

### CLI Arguments Summary

| Argument | Flag | Default | Description |
| :--- | :---: | :---: | :--- |
| `--input` | `-i` | `DoraNEwinverterAC_1.webp` | Path to an input image or directory of images. |
| `--output` | `-o` | `output/Dora_ECommerce_Perfect.jpg` | Output file path or target directory. |
| `--size` | `-s` | `1500` | Target square canvas dimension in pixels. |
| `--fill` | `-f` | `0.85` | Product bounding box max fill ratio (e.g. 0.85 = 85%). |
| `--inpaint-stickers` | | `True` | Multi-cue sticker detection and deep inpainting. |
| `--model` | | `u2net` | Rembg segmentation model (`u2net`, `birefnet-general`). |

---

## 📊 Evaluation & Verification Checklist (TASKTWO.md)

Benchmark results on reference test asset `DoraNEwinverterAC_1.webp`:

| Acceptance Criteria | Specification | Result |
| :--- | :--- | :---: |
| **No Remnant Typography or Badges** | Zero visible text, QR code, green header, or sticker borders | ✅ Passed |
| **Plastic Curvature & Shading** | Inpainted chassis matches ambient lighting gradients smoothly | ✅ Passed |
| **Pure White Background** | Canvas perimeter is 100% `#FFFFFF` (`RGB 255, 255, 255`) | ✅ Passed |
| **Mathematical Centering** | 1500x1500px, exactly 85% fill, balanced L/R and T/B margins | ✅ Passed |
| **Edge Sharpness** | Top and lateral AC contours distinct against pure white background | ✅ Passed |

---

## 📁 Repository Structure

```text
├── DoraNEwinverterAC_1.webp       # Reference test catalog asset
├── pipeline.py                    # Core multi-cue inpainting & standardization script
├── standardize_pipeline.py        # Task execution wrapper
├── requirements.txt               # Pinned dependencies (including LaMa & Rembg)
├── .gitignore                     # Git ignore rules
├── TASK.md                        # Phase 1 specifications
├── TASKTWO.md                     # Phase 2 advanced inpainting specifications
├── README.md                      # Comprehensive project documentation
└── output/                        # Verified marketplace outputs
    ├── Dora_ECommerce_Perfect.jpg # Inpainted & standardized (LaMa SOTA)
    ├── Dora_ECommerce_Standard.jpg
    └── Dora_ECommerce_NoStickers.jpg
```

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.
