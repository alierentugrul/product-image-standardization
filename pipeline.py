"""
E-Commerce Product Image Standardization & Deep Inpainting Pipeline
Automated, production-grade image processing pipeline for marketplace readiness.
Features:
- Multi-cue catalog sticker detection (HSV saturation + Canny edges + morphological fusion)
- Deep surface inpainting (LaMa SOTA / 2D vertical gradient interpolation fallback)
- AI salient object segmentation (rembg)
- Aspect-ratio preserving scaling to 85% fill on 1500x1500px pure white (#FFFFFF) canvas
- White-on-white edge preservation and sharpness enhancement
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

# Ensure safe console output across all Windows encodings
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import rembg


def get_rembg_session(model_name: str = "u2net"):
    """Initialize and return a rembg session with the specified model."""
    try:
        return rembg.new_session(model_name)
    except Exception as e:
        print(f"[Warning] Failed to initialize rembg session with '{model_name}': {e}. Falling back to default 'u2net'.")
        return rembg.new_session("u2net")


def detect_sticker_bbox(image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detects the complete bounding box of the catalog sticker by fusing
    color saturation and high-frequency edge density within the front-panel ROI.
    """
    h, w = image_bgr.shape[:2]
    # Restrict search to front-panel ROI where catalog stickers are placed
    y1, y2 = int(h * 0.25), int(h * 0.70)
    x1, x2 = int(w * 0.05), int(w * 0.40)
    roi = image_bgr[y1:y2, x1:x2]

    # 1. Color saturation mask (catches colored energy rating bars and green headers)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat_mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 255, 255]))

    # 2. Text and QR code high-frequency edges
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # 3. Fuse both signals
    combined = cv2.bitwise_or(sat_mask, edges)

    # 4. Heavy rectangular morphological dilation to fuse discrete text, QR, and bars into one solid mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    dilated = cv2.dilate(combined, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Filter out tiny noise contours; find the primary sticker block
    valid_contours = [c for c in contours if cv2.contourArea(c) > 1500]
    if not valid_contours:
        return None

    largest_c = max(valid_contours, key=cv2.contourArea)
    rx, ry, rw, rh = cv2.boundingRect(largest_c)

    # 5. Add safety padding (pad = 8px) to eliminate all edge/border remnants
    pad = 8
    bx = max(0, x1 + rx - pad)
    by = max(0, y1 + ry - pad)
    bw = min(w - bx, rw + (pad * 2))
    bh = min(h - by, rh + (pad * 2))

    return (bx, by, bw, bh)


def inpaint_panel(image_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Inpaints the sticker region using SOTA LaMa model or 2D gradient interpolation fallback.
    Seamlessly reconstructs the AC unit's curved plastic casing with matching lighting gradients.
    """
    x, y, w, h = bbox
    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    mask[y:y+h, x:x+w] = 255

    # 1. Primary Engine: LaMa (Large Mask Inpainting)
    try:
        from simple_lama_inpainting import SimpleLama
        lama = SimpleLama()
        img_rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask)
        inpainted_pil = lama(img_rgb, mask_pil)
        print("  -> Surface inpainting succeeded with LaMa deep model.")
        return cv2.cvtColor(np.array(inpainted_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"  -> LaMa unavailable ({e}), using smooth 2D gradient interpolation fallback...")

    # 2. Fallback Engine: 2D vertical gradient interpolation across patch boundaries + Telea edge blending
    border_top = image_bgr[max(0, y-10):y, x:x+w].mean(axis=(0, 1))
    border_bottom = image_bgr[y+h:min(image_bgr.shape[0], y+h+10), x:x+w].mean(axis=(0, 1))

    gradient_fill = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        gradient_fill[:, :, c] = np.linspace(border_top[c], border_bottom[c], h)[:, None]

    temp = image_bgr.copy()
    temp[y:y+h, x:x+w] = np.clip(gradient_fill, 0, 255).astype(np.uint8)

    # Blend outer seam transition with Telea inpainting
    border_mask = np.zeros_like(mask)
    cv2.rectangle(border_mask, (x, y), (x+w, y+h), 255, 3)
    blended = cv2.inpaint(temp, border_mask, 5, cv2.INPAINT_TELEA)

    return blended


def preserve_white_on_white_edges(image_rgba: Image.Image) -> Image.Image:
    """
    Prevents pure white products (e.g. top edge of an AC unit) from washing out
    or dissolving into the pure #FFFFFF background.
    """
    img_np = np.array(image_rgba, dtype=np.float32)
    rgb = img_np[:, :, :3]
    alpha = img_np[:, :, 3]

    # Binary mask of foreground product
    binary_mask = (alpha > 30).astype(np.uint8) * 255

    # Compute outer boundary contour (gradient of mask, ~2px width)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(binary_mask, kernel, iterations=1)
    boundary = (binary_mask - eroded) > 0

    # Luminance calculation
    luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    # Delineate only boundary pixels that are near-white (> 240)
    bright_boundary = boundary & (luminance > 240)
    if np.any(bright_boundary):
        shading_factor = np.ones_like(luminance)
        shading_factor[bright_boundary] = 0.92  # Soft 8% edge contour definition
        for c in range(3):
            rgb[:, :, c] = np.clip(rgb[:, :, c] * shading_factor, 0, 255)

    result_np = np.dstack([rgb.astype(np.uint8), alpha.astype(np.uint8)])
    return Image.fromarray(result_np, mode="RGBA")


def standardize_product_image(
    input_path: Path,
    output_path: Path,
    canvas_size: int = 1500,
    fill_ratio: float = 0.85,
    inpaint_stickers: bool = True,
    rembg_session=None,
) -> bool:
    """
    Execute full 5-stage e-commerce image standardization pipeline on a single image.
    """
    try:
        print(f"[*] Processing: {input_path}")
        orig_bgr = cv2.imread(str(input_path))
        if orig_bgr is None:
            raise FileNotFoundError(f"Cannot read image at {input_path}")

        # Stage 1 & 2: Multi-Cue Sticker Detection & SOTA Inpainting
        if inpaint_stickers:
            bbox = detect_sticker_bbox(orig_bgr)
            if bbox is not None:
                bx, by, bw, bh = bbox
                print(f"[+] Multi-cue sticker bounding box located: x={bx}, y={by}, w={bw}, h={bh}. Inpainting...")
                clean_bgr = inpaint_panel(orig_bgr, bbox)
            else:
                print("[-] No catalog sticker detected in ROI. Skipping inpainting.")
                clean_bgr = orig_bgr
        else:
            clean_bgr = orig_bgr

        # Stage 3: Salient Foreground Segmentation
        print("  -> Isolating foreground with rembg...")
        clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
        clean_pil = Image.fromarray(clean_rgb)

        if rembg_session:
            no_bg = rembg.remove(clean_pil, session=rembg_session)
        else:
            no_bg = rembg.remove(clean_pil)

        # White-on-white edge contrast protection
        no_bg = preserve_white_on_white_edges(no_bg)

        # Stage 4: Strict Aspect-Ratio Preserving Centering & Sizing
        alpha = np.array(no_bg)[:, :, 3]
        # Filter out faint rembg shadow noise (alpha <= 15)
        clean_alpha = np.where(alpha > 15, alpha, 0).astype(np.uint8)
        no_bg.putalpha(Image.fromarray(clean_alpha))

        coords = cv2.findNonZero((clean_alpha > 0).astype(np.uint8))
        if coords is None:
            print(f"[Error] No foreground detected in {input_path.name}")
            return False

        rx, ry, rw, rh = cv2.boundingRect(coords)
        print(f"  -> Tight product bounding box: x={rx}, y={ry}, w={rw}, h={rh}")
        cropped = no_bg.crop((rx, ry, rx + rw, ry + rh))

        target_max_dim = int(canvas_size * fill_ratio)
        # Proportionally resize strictly maintaining aspect ratio
        scale = min(target_max_dim / rw, target_max_dim / rh)
        new_w = max(1, int(round(rw * scale)))
        new_h = max(1, int(round(rh * scale)))

        print(f"  -> Scaling to {new_w}x{new_h} (Fill ratio: {fill_ratio:.2f}, Canvas: {canvas_size}x{canvas_size})...")
        resized_product = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Stage 5: Detail Enhancement & Clean Canvas Compositing
        product_rgb = resized_product.convert("RGB")
        enhancer = ImageEnhance.Sharpness(product_rgb)
        sharpened_rgb = enhancer.enhance(1.3)
        sharpened_product = Image.merge("RGBA", (*sharpened_rgb.split(), resized_product.split()[-1]))

        # Instantiate pure white canvas (#FFFFFF)
        canvas = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 255))
        offset_x = (canvas_size - new_w) // 2
        offset_y = (canvas_size - new_h) // 2

        print(f"  -> Centering on canvas at offset ({offset_x}, {offset_y})...")
        canvas.paste(sharpened_product, (offset_x, offset_y), mask=sharpened_product.split()[-1])

        # Final export
        final_rgb = canvas.convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix.lower() in [".jpg", ".jpeg"]:
            final_rgb.save(output_path, "JPEG", quality=98, optimize=True, subsampling=0)
        else:
            final_rgb.save(output_path, "PNG", optimize=True)

        print(f"[SUCCESS] Successfully exported standardized image to: {output_path}")
        return True

    except Exception as e:
        print(f"[Error] Failed to process {input_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="E-Commerce Asset Standardization & Deep Inpainting Pipeline"
    )
    parser.add_argument(
        "--input", "-i", default="DoraNEwinverterAC_1.webp",
        help="Path to input image or directory of images."
    )
    parser.add_argument(
        "--output", "-o", default="output/Dora_ECommerce_Perfect.jpg",
        help="Path to output image or directory."
    )
    parser.add_argument(
        "--size", "-s", type=int, default=1500,
        help="Target square canvas dimension in pixels (default: 1500)."
    )
    parser.add_argument(
        "--fill", "-f", type=float, default=0.85,
        help="Max fill ratio inside canvas (default: 0.85)."
    )
    parser.add_argument(
        "--inpaint-stickers", action=argparse.BooleanOptionalAction, default=True,
        help="Enable/disable multi-cue catalog sticker detection and inpainting (default: True)."
    )
    parser.add_argument(
        "--model", default="u2net", type=str,
        help="Rembg segmentation model name (default: u2net)."
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[Error] Input path '{input_path}' does not exist.")
        sys.exit(1)

    print("[Init] Initializing segmentation engine...")
    session = get_rembg_session(args.model)

    supported_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

    if input_path.is_file():
        if output_path.is_dir() or output_path.suffix == "":
            out_file = output_path / f"{input_path.stem}_Standard.jpg"
        else:
            out_file = output_path

        success = standardize_product_image(
            input_path=input_path,
            output_path=out_file,
            canvas_size=args.size,
            fill_ratio=args.fill,
            inpaint_stickers=args.inpaint_stickers,
            rembg_session=session,
        )
        sys.exit(0 if success else 1)

    elif input_path.is_dir():
        image_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in supported_extensions]
        print(f"[Batch] Found {len(image_files)} images in directory {input_path}")

        output_path.mkdir(parents=True, exist_ok=True)
        success_count = 0

        for img_file in image_files:
            out_file = output_path / f"{img_file.stem}_Standard.jpg"
            if standardize_product_image(
                input_path=img_file,
                output_path=out_file,
                canvas_size=args.size,
                fill_ratio=args.fill,
                inpaint_stickers=args.inpaint_stickers,
                rembg_session=session,
            ):
                success_count += 1

        print(f"[Batch Complete] Successfully processed {success_count}/{len(image_files)} images.")
        sys.exit(0 if success_count == len(image_files) else 1)


if __name__ == "__main__":
    main()
