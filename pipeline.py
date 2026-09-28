"""
E-Commerce Product Image Standardization & Deep Inpainting Pipeline
Production-grade image processing pipeline for commercial marketplace standards.

Key Features:
- Segment-First Silhouette Lock: Background is segmented first; the raw alpha contour is locked.
- Contour Clamping: Inpainting masks are strictly clamped to the product alpha channel (cv2.bitwise_and)
  and kept safely below the chassis ceiling line (y > y_top + 18%), preventing any outer contour warping ("horn" artifacts).
- SOTA Surface Inpainting: Uses LaMa (Large Mask Inpainting) with automatic Telea fallback.
- Pure White Canvas: 1500x1500px (#FFFFFF / 255, 255, 255) pure background.
- Precision Framing: Strictly aspect-ratio-locked proportional scaling (85% fill) and mathematical centering.
- White-on-White Edge Preservation: Boundary contrast enhancement keeps white products distinct on white backgrounds.
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


def extract_clamped_sticker_mask(
    seg_np: np.ndarray,
    x_ac: int,
    y_ac: int,
    w_ac: int,
    h_ac: int,
) -> Optional[np.ndarray]:
    """
    Detects the sticker inside the front-panel ROI safely below the top edge of the product,
    and clamps the resulting mask strictly to the product alpha channel (cv2.bitwise_and).
    Guarantees zero bleed outside the product contour.
    """
    alpha = seg_np[:, :, 3]

    # Constrain search strictly below the top ceiling contour (y > y_ac + 18% of product height)
    y1 = y_ac + int(h_ac * 0.18)
    y2 = y_ac + int(h_ac * 0.70)
    x1 = x_ac + int(w_ac * 0.02)
    x2 = x_ac + int(w_ac * 0.25)

    roi_rgb = seg_np[y1:y2, x1:x2, :3]
    roi_hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)

    # 1. Color saturation mask (catches rating bars and green headers)
    sat_mask = cv2.inRange(roi_hsv, np.array([0, 35, 40]), np.array([180, 255, 255]))

    # 2. Text and QR code high-frequency edges
    edges = cv2.Canny(roi_gray, 40, 120)

    # 3. Fuse signals
    combined = cv2.bitwise_or(sat_mask, edges)

    # 4. Morphological fusion to merge all sticker components into one solid block
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(combined, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    valid_contours = [c for c in contours if cv2.contourArea(c) > 500]
    if not valid_contours:
        return None

    largest_c = max(valid_contours, key=cv2.contourArea)
    rx, ry, rw, rh = cv2.boundingRect(largest_c)

    # Add safety padding
    pad = 6
    bx = max(x1, x1 + rx - pad)
    by = max(y1, y1 + ry - pad)
    bw = min(x2 - bx, rw + (pad * 2))
    bh = min(y2 - by, rh + (pad * 2))

    inpaint_mask = np.zeros(alpha.shape, dtype=np.uint8)
    inpaint_mask[by:by+bh, bx:bx+bw] = 255

    # CRITICAL: Clamp strictly with product alpha channel
    # Prevents any pixel outside the product from being included in the mask
    clamped_mask = cv2.bitwise_and(inpaint_mask, alpha)

    print(f"  -> Sticker bounding box: x={bx}, y={by}, w={bw}, h={bh} (Clamped to alpha channel)")
    return clamped_mask


def inpaint_surface(seg_np: np.ndarray, inpaint_mask: np.ndarray) -> np.ndarray:
    """
    Inpaints the sticker region using SOTA LaMa model or Telea fallback.
    Reconstructs the surface while preserving the locked alpha channel.
    """
    alpha = seg_np[:, :, 3]
    rgb = seg_np[:, :, :3]
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    try:
        from simple_lama_inpainting import SimpleLama
        lama = SimpleLama()
        pil_img = Image.fromarray(rgb)
        pil_mask = Image.fromarray(inpaint_mask)
        clean_rgb = np.array(lama(pil_img, pil_mask))
        print("  -> Surface inpainting succeeded with LaMa deep model.")
    except Exception as e:
        print(f"  -> LaMa unavailable ({e}), using Telea edge blending fallback...")
        clean_bgr = cv2.inpaint(rgb_bgr, inpaint_mask, 7, cv2.INPAINT_TELEA)
        clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)

    # Re-attach the pristine locked alpha channel
    return np.dstack((clean_rgb, alpha))


def preserve_white_on_white_edges(image_rgba: Image.Image) -> Image.Image:
    """
    Prevents pure white products from washing out into the pure #FFFFFF background.
    Adds subtle edge contour delineation to extremely bright boundary pixels.
    """
    img_np = np.array(image_rgba, dtype=np.float32)
    rgb = img_np[:, :, :3]
    alpha = img_np[:, :, 3]

    binary_mask = (alpha > 30).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(binary_mask, kernel, iterations=1)
    boundary = (binary_mask - eroded) > 0

    luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
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
    Standardize product image with Silhouette Lock and Contour Clamping.
    """
    try:
        print(f"[*] Processing: {input_path}")
        raw_img = Image.open(input_path).convert("RGBA")

        # 1. Segment-First: Isolate foreground and lock the exact silhouette
        print("  -> Step 1: Isolating foreground and locking silhouette...")
        if rembg_session:
            segmented = rembg.remove(raw_img, session=rembg_session)
        else:
            segmented = rembg.remove(raw_img)

        seg_np = np.array(segmented)
        alpha = seg_np[:, :, 3]

        # Clean faint shadow noise (alpha <= 15) to isolate the true physical product
        alpha = np.where(alpha > 15, alpha, 0).astype(np.uint8)
        seg_np[:, :, 3] = alpha

        coords = cv2.findNonZero(alpha)
        if coords is None:
            print(f"[Error] No foreground detected in {input_path.name}")
            return False

        x_ac, y_ac, w_ac, h_ac = cv2.boundingRect(coords)
        print(f"  -> Locked product bounds: x={x_ac}, y={y_ac}, w={w_ac}, h={h_ac}")

        # 2. Clamped Inpainting: Inpaint strictly inside the product body below ceiling line
        if inpaint_stickers:
            print("  -> Step 2: Detecting sticker and clamping mask...")
            clamped_mask = extract_clamped_sticker_mask(seg_np, x_ac, y_ac, w_ac, h_ac)
            if clamped_mask is not None and np.any(clamped_mask > 0):
                clean_rgba_np = inpaint_surface(seg_np, clamped_mask)
            else:
                print("  -> No sticker detected in ROI. Skipping inpainting.")
                clean_rgba_np = seg_np
        else:
            clean_rgba_np = seg_np

        clean_pil = Image.fromarray(clean_rgba_np, mode="RGBA")

        # 3. White-on-White Edge Preservation
        clean_pil = preserve_white_on_white_edges(clean_pil)

        # 4. Aspect-Ratio Locked Proportional Scaling & Centering
        cropped = clean_pil.crop((x_ac, y_ac, x_ac + w_ac, y_ac + h_ac))

        target_max_dim = int(canvas_size * fill_ratio)
        scale = min(target_max_dim / w_ac, target_max_dim / h_ac)
        new_w = max(1, int(round(w_ac * scale)))
        new_h = max(1, int(round(h_ac * scale)))

        print(f"  -> Scaling to {new_w}x{new_h} (Fill ratio: {fill_ratio:.2f}, Canvas: {canvas_size}x{canvas_size})...")
        resized_product = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Detail enhancement on product layer
        product_rgb = resized_product.convert("RGB")
        enhancer = ImageEnhance.Sharpness(product_rgb)
        sharpened_rgb = enhancer.enhance(1.25)
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
        "--output", "-o", default="output/Dora_ECommerce_Fixed.jpg",
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
