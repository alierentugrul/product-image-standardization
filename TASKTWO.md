Markdown
# Task: Automated E-Commerce Image Standardization & Deep Inpainting Pipeline

## 1. Objective & Problem Definition
The goal is to build a fully automated, production-grade Python image processing pipeline that ingests catalog/brochure product assets and outputs standardized e-commerce images (1500x1500px, pure white `#FFFFFF` background, centered, sharp edges).

### The Critical Bug in Previous Iteration:
A naive color-thresholding approach (filtering only green/yellow HSV values) failed on `DoraNEwinverterAC_1.webp`. It erased the energy efficiency color bars but left the green header, black typography, QR code, and sticker border behind, creating an unacceptable blotchy artifact.

### Required Resolution:
Implement a multi-cue morphological detection strategy that locks onto the **entire sticker bounding box** (combining color saturation, high-frequency text/QR edges, and morphological dilation) followed by deep surface inpainting (LaMa / Large Mask Inpainting) to seamlessly reconstruct the curved plastic chassis.

---

## 2. Technical Architecture & Pipeline Stages

### Stage 1: Multi-Cue Sticker Bounding Box Detection
- Restrict search to the front panel region of interest (ROI) (horizontal: 5% to 40%, vertical: 25% to 70%).
- Extract high-saturation regions (colored rating bars) using HSV thresholding.
- Extract high-frequency textual/QR edges using Canny edge detection.
- Merge both signals (`cv2.bitwise_or`) and apply a heavy rectangular morphological dilation kernel (`25x25`) to fuse discrete text lines, borders, and QR codes into a single solid mask component.
- Find the enclosing bounding box (`cv2.boundingRect`) and expand it with safety padding (`pad = 8px`) to ensure zero border remnants.

### Stage 2: SOTA Surface Inpainting
- Primary Engine: Use `simple-lama-inpainting` (LaMa - Large Mask Inpainting) to hallucinate and reconstruct the AC's curved plastic casing with realistic lighting gradients and zero discoloration.
- Fallback Engine: If PyTorch/LaMa dependencies are missing, use 2D vertical gradient interpolation across the patch boundaries followed by edge blending via `cv2.inpaint`.

### Stage 3: Salient Foreground Segmentation
- Run `rembg` on the inpainted image to isolate the product foreground from any background cast or shadows.
- Extract the alpha channel to compute the exact product footprint.

### Stage 4: Strict Aspect-Ratio Preserving Centering & Sizing
- Compute `cv2.boundingRect` on the non-zero alpha coordinates.
- Proportionally resize the asset using Lanczos resampling so that its longest dimension fits within `85%` of the target canvas dimension (e.g., 1275px inside a 1500x1500px canvas).
- Calculate horizontal and vertical offsets to place the asset at the mathematical center.

### Stage 5: Contrast Recovery & Edge Sharpening
- Composite the resized asset onto a pure white canvas (`255, 255, 255`).
- Apply sharpness enhancement (factor: `1.3`) so product contours, branding logos, and vents remain crisp and legible against white backgrounds.

---

## 3. Environment & Dependencies

Install the required packages in your Python environment:
```bash
pip install opencv-python pillow numpy rembg simple-lama-inpainting
4. Complete Reference Implementation (standardize_pipeline.py)
Python
import argparse
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from rembg import remove

def detect_sticker_bbox(image_bgr):
    """
    Detects the complete bounding box of the catalog sticker by fusing
    color saturation and high-frequency edge density.
    """
    h, w = image_bgr.shape[:2]
    y1, y2 = int(h * 0.25), int(h * 0.70)
    x1, x2 = int(w * 0.05), int(w * 0.40)
    roi = image_bgr[y1:y2, x1:x2]

    # 1. Color saturation mask (bars)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat_mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 255, 255]))

    # 2. Text and QR code edges
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # 3. Fuse signals
    combined = cv2.bitwise_or(sat_mask, edges)

    # 4. Morphological fusion to merge all sticker elements into one block
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    dilated = cv2.dilate(combined, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_c = max(contours, key=cv2.contourArea)
    rx, ry, rw, rh = cv2.boundingRect(largest_c)

    # Add safety padding
    pad = 8
    bx = max(0, x1 + rx - pad)
    by = max(0, y1 + ry - pad)
    bw = min(w - bx, rw + (pad * 2))
    bh = min(h - by, rh + (pad * 2))

    return (bx, by, bw, bh)

def inpaint_panel(image_bgr, bbox):
    """
    Inpaints the sticker region using SOTA LaMa model or gradient interpolation fallback.
    """
    x, y, w, h = bbox
    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    mask[y:y+h, x:x+w] = 255

    try:
        from simple_lama_inpainting import SimpleLama
        lama = SimpleLama()
        img_rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask)
        inpainted_pil = lama(img_rgb, mask_pil)
        return cv2.cvtColor(np.array(inpainted_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[!] LaMa unavailable ({e}), using gradient fallback...")
        border_top = image_bgr[max(0, y-10):y, x:x+w].mean(axis=(0, 1))
        border_bottom = image_bgr[y+h:min(image_bgr.shape[0], y+h+10), x:x+w].mean(axis=(0, 1))

        gradient_fill = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(3):
            gradient_fill[:, :, c] = np.linspace(border_top[c], border_bottom[c], h)[:, None]

        temp = image_bgr.copy()
        temp[y:y+h, x:x+w] = gradient_fill

        border_mask = np.zeros_like(mask)
        cv2.rectangle(border_mask, (x, y), (x+w, y+h), 255, 3)
        return cv2.inpaint(temp, border_mask, 5, cv2.INPAINT_TELEA)

def process_image(input_path, output_path, canvas_size=1500, fill_ratio=0.85):
    print(f"[*] Processing: {input_path}")
    orig_bgr = cv2.imread(input_path)
    if orig_bgr is None:
        raise FileNotFoundError(f"Cannot read image at {input_path}")

    # Stage 1 & 2: Detect & Inpaint sticker
    bbox = detect_sticker_bbox(orig_bgr)
    if bbox is not None:
        print(f"[+] Sticker located at: {bbox}. Inpainting...")
        clean_bgr = inpaint_panel(orig_bgr, bbox)
    else:
        print("[-] No sticker detected, skipping inpainting.")
        clean_bgr = orig_bgr

    # Stage 3: Segment background
    clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
    no_bg = remove(Image.fromarray(clean_rgb))

    # Stage 4: Centering and unified scaling
    alpha = np.array(no_bg)[:, :, 3]
    coords = cv2.findNonZero(alpha)
    rx, ry, rw, rh = cv2.boundingRect(coords)
    cropped = no_bg.crop((rx, ry, rx + rw, ry + rh))

    target_dim = (canvas_size, canvas_size)
    max_dim = int(canvas_size * fill_ratio)
    cropped.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", target_dim, (255, 255, 255, 255))
    offset_x = (canvas_size - cropped.size[0]) // 2
    offset_y = (canvas_size - cropped.size[1]) // 2
    canvas.paste(cropped, (offset_x, offset_y), mask=cropped)

    # Stage 5: Edge enhancement & Output
    final_rgb = canvas.convert("RGB")
    enhancer = ImageEnhance.Sharpness(final_rgb)
    final_output = enhancer.enhance(1.3)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    final_output.save(output_path, "JPEG", quality=98)
    print(f"[✓] Successfully exported to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E-Commerce Asset Standardization Pipeline")
    parser.add_argument("--input", "-i", default="DoraNEwinverterAC_1.webp", help="Path to input image")
    parser.add_argument("--output", "-o", default="output/Dora_ECommerce_Perfect.jpg", help="Path to output image")
    parser.add_argument("--size", "-s", type=int, default=1500, help="Canvas size in pixels")
    parser.add_argument("--fill", "-f", type=float, default=0.85, help="Fill ratio")
    args = parser.parse_args()

    process_image(args.input, args.output, args.size, args.fill)
5. Execution Command
Bash
python standardize_pipeline.py --input DoraNEwinverterAC_1.webp --output output/Dora_ECommerce_Perfect.jpg
6. Acceptance Criteria Checklist
[ ] No remnant text, green header, or QR code visible on the AC unit.

[ ] Plastic curvature and shading across the inpainted area match the rest of the unit.

[ ] Background is pure #FFFFFF (RGB 255, 255, 255) across the entire canvas perimeter.

[ ] Product is mathematically centered with a consistent 85% scale factor.

[ ] Top/side borders remain sharp and discernible against the pure white background.