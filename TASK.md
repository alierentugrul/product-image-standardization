\# Project: E-Commerce Product Image Standardization \& AI Preprocessing Pipeline



\## 1. Project Overview \& Goal

Build an automated, production-ready Python image processing pipeline designed to ingest raw, low-quality, or catalog/PDF-extracted product images and output standardized, commercial e-commerce assets matching strict marketplace standards.



The initial reference test case is `DoraNEwinverterAC\_1.webp` (an air conditioner unit with catalog stickers and edge contrast challenges on white backgrounds).



\---



\## 2. Core Requirements \& Acceptance Criteria

1\. \*\*Background Isolation:\*\*

&#x20;  - Detect and segment the primary product foreground cleanly.

&#x20;  - Eliminate all original background elements, shadows, and catalog artifacts.

&#x20;  - Place the isolated asset onto a pure white canvas (`#FFFFFF` / `255, 255, 255`).



2\. \*\*Unified Framing, Scaling \& Centering:\*\*

&#x20;  - Determine the strict bounding box of the non-transparent/foreground pixels.

&#x20;  - Scale the product proportionally (strictly preserve aspect ratio, no stretching/distortion).

&#x20;  - Ensure the product occupies a configurable percentage of the canvas (default: `85%` fill ratio).

&#x20;  - Position the bounding box at the exact center (both horizontal and vertical) of a square canvas (default: `1500x1500px`).



3\. \*\*Artifact \& Sticker Inpainting (Optional / Modular):\*\*

&#x20;  - Provide a modular step to remove unwanted catalog stickers (such as energy rating stickers, QR codes, or promo tags) without damaging the underlying surface geometry.

&#x20;  - Blend inpainted regions seamlessly with the surrounding texture/color using OpenCV inpainting (`cv2.inpaint`) or lightweight mask-based inpainting.



4\. \*\*Edge Preservation \& White-on-White Contrast:\*\*

&#x20;  - Prevent pure white products from blending or dissolving into the pure white background.

&#x20;  - Apply unsharp masking / local contrast enhancement to maintain crisp product boundaries, vents, text logos, and seams.



5\. \*\*Batch Execution \& CLI Interface:\*\*

&#x20;  - Support both single-image processing and batch processing of an entire directory.

&#x20;  - Save outputs in high-quality JPEG (`quality=95`) or PNG.



\---



\## 3. Technology Stack \& Dependencies

\- \*\*Language:\*\* Python 3.10+

\- \*\*Core Libraries:\*\*

&#x20; - `opencv-python` (Geometric transformations, contour bounding boxes, inpainting, filtering)

&#x20; - `pillow` (Canvas manipulation, compositing, ICC profile / color preservation)

&#x20; - `numpy` (Pixel matrix manipulation, mask thresholding)

&#x20; - `rembg` (Deep-learning salient object segmentation using `u2net` or `birefnet-general`)



\---



\## 4. Pipeline Architecture \& Implementation Steps



\### Step 1: Environment Setup

Create a virtual environment and install required packages:

```bash

pip install opencv-python pillow numpy rembg

Step 2: Foreground Segmentation

Load input image (RGBA).



Run rembg.remove with alpha matting enabled if necessary to preserve semi-transparent edges.



Extract the alpha channel mask to isolate product boundaries.



Step 3: Bounding Box \& Scale Computation

Find non-zero coordinates of the alpha mask:



Python

coords = cv2.findNonZero(alpha\_mask)

x, y, w, h = cv2.boundingRect(coords)

Crop the product tightly to (x, y, w, h).



Calculate resize dimensions to fit canvas\_size \* fill\_ratio while maintaining aspect ratio via Lanczos resampling.



Step 4: Canvas Compositing \& Centering

Instantiate a blank RGBA image with color (255, 255, 255, 255).



Calculate offsets:



Python

offset\_x = (canvas\_w - resized\_w) // 2

offset\_y = (canvas\_h - resized\_h) // 2

Paste the resized product onto the center of the canvas using its alpha channel as a mask.



Step 5: Post-Processing \& Edge Sharpening

Convert the composite to RGB.



Apply an Unsharp Mask filter (or PIL.ImageEnhance.Sharpness with factor 1.3 - 1.5) to restore micro-details and logo sharpness.



Ensure the boundary between the white product edges and white background remains distinct.



5\. CLI Script Specification (pipeline.py)

The script should accept command-line arguments via argparse:



\--input / -i: Path to an image file or directory of images (Required).



\--output / -o: Output file path or directory (Default: ./output).



\--size / -s: Target canvas dimension in pixels (Default: 1500).



\--fill: Max fill ratio inside the canvas (Default: 0.85).



\--inpaint-stickers: Boolean flag to trigger automatic color-threshold sticker detection and inpainting.



6\. Verification \& Test Case

Run the pipeline against the test file:



Bash

python pipeline.py --input DoraNEwinverterAC\_1.webp --output output/Dora\_ECommerce\_Standard.jpg --size 1500 --fill 0.85

Evaluation Checklist:



\[ ] Product is perfectly centered in a 1500x1500px square.



\[ ] Background is pure #FFFFFF across all outer pixels.



\[ ] Product edges are clean without dark fringing or halo artifacts.



\[ ] The top edge of the AC remains visible against the white background.

