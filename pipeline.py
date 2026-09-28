"""
E-Commerce Product Image Standardization & AI Preprocessing Pipeline
Standardizes product images: background removal, centering, proportional scaling,
edge preservation on white backgrounds, and optional sticker inpainting.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

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


def remove_background(image: Image.Image, session=None) -> Image.Image:
    """
    Remove background from the input PIL Image and return an RGBA Image.
    Uses rembg with alpha matting enabled if needed for crisp borders.
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    
    # Run rembg foreground extraction
    kwargs = {
        "alpha_matting": True,
        "alpha_matting_foreground_threshold": 240,
        "alpha_matting_background_threshold": 10,
        "alpha_matting_erode_size": 5,
    }
    try:
        if session:
            kwargs["session"] = session
        result = rembg.remove(image, **kwargs)
    except Exception:
        # Fallback without alpha matting if model doesn't support specific matting args
        kwargs.pop("alpha_matting", None)
        kwargs.pop("alpha_matting_foreground_threshold", None)
        kwargs.pop("alpha_matting_background_threshold", None)
        kwargs.pop("alpha_matting_erode_size", None)
        result = rembg.remove(image, **kwargs)

    return result


def detect_and_inpaint_stickers(image_rgba: Image.Image) -> Image.Image:
    """
    Modular step to detect and inpaint unwanted catalog stickers/energy rating labels.
    Stickers typically have high saturation (yellow, red, green) or barcode patterns.
    """
    img_np = np.array(image_rgba)
    rgb = img_np[:, :, :3]
    alpha = img_np[:, :, 3]

    # Convert to HSV for vibrant sticker / energy label detection
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Stickers are typically high saturation & moderate/high brightness on white/gray products
    # Product surface is mostly neutral (low saturation), stickers have s > 65 and v > 60
    color_mask = (s > 60) & (v > 60) & (alpha > 128)
    color_mask = color_mask.astype(np.uint8) * 255

    # Filter out tiny noise and retain rectangular sticker blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours that qualify as stickers (area > 1500 px, bounding box inside product)
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    inpaint_mask = np.zeros(alpha.shape, dtype=np.uint8)

    sticker_found = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            # Avoid inpainting the entire product: width/height should be reasonable fraction
            if w < img_np.shape[1] * 0.4 and h < img_np.shape[0] * 0.4:
                cv2.drawContours(inpaint_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                sticker_found = True

    if not sticker_found:
        return image_rgba

    # Dilate mask slightly to cover edge transitions
    inpaint_mask = cv2.dilate(inpaint_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    # Inpaint RGB channels
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    inpainted_bgr = cv2.inpaint(bgr, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)

    result_np = np.dstack([inpainted_rgb, alpha])
    return Image.fromarray(result_np, mode="RGBA")


def preserve_white_on_white_edges(image_rgba: Image.Image) -> Image.Image:
    """
    Enhance white-on-white edge contrast:
    Ensures that white products (e.g. top edge of an AC unit) do not wash out
    or dissolve into a pure #FFFFFF background.
    Adds subtle edge shading to perimeter pixels that are extremely bright.
    """
    img_np = np.array(image_rgba, dtype=np.float32)
    rgb = img_np[:, :, :3]
    alpha = img_np[:, :, 3]

    # Binary mask of the product
    binary_mask = (alpha > 30).astype(np.uint8) * 255

    # Compute outer boundary contour (gradient of mask, ~2px width)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(binary_mask, kernel, iterations=1)
    boundary = (binary_mask - eroded) > 0

    # Luminance of RGB pixels
    luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    # Where boundary pixels are extremely bright (> 245), apply subtle shading
    bright_boundary = boundary & (luminance > 240)
    
    # Create smooth edge definition factor (darken bright edges by 6-9% to delineate against pure #FFF)
    if np.any(bright_boundary):
        shading_factor = np.ones_like(luminance)
        shading_factor[bright_boundary] = 0.92  # Soft 8% edge contour definition
        for c in range(3):
            rgb[:, :, c] = np.clip(rgb[:, :, c] * shading_factor, 0, 255)

    result_np = np.dstack([rgb.astype(np.uint8), alpha.astype(np.uint8)])
    result_img = Image.fromarray(result_np, mode="RGBA")

    # Local detail & logo enhancement
    enhancer = ImageEnhance.Sharpness(result_img)
    result_img = enhancer.enhance(1.35)

    return result_img


def standardize_product_image(
    input_path: Path,
    output_path: Path,
    canvas_size: int = 1500,
    fill_ratio: float = 0.85,
    inpaint_stickers: bool = False,
    rembg_session=None,
) -> bool:
    """
    Standardize a single product image according to e-commerce marketplace standards.
    """
    try:
        print(f"[Processing] Loading {input_path.name}...")
        orig_img = Image.open(input_path)

        # 1. Background Isolation
        print(f"  -> Isolating foreground with rembg...")
        rgba_img = remove_background(orig_img, session=rembg_session)

        # 2. Sticker Inpainting (Modular)
        if inpaint_stickers:
            print(f"  -> Inpainting stickers/catalog artifacts...")
            rgba_img = detect_and_inpaint_stickers(rgba_img)

        # 3. White-on-White Edge Preservation & Detail Enhancement
        print(f"  -> Enhancing edge contrast and details...")
        rgba_img = preserve_white_on_white_edges(rgba_img)

        # 4. Strict Bounding Box & Scale Computation
        alpha_channel = np.array(rgba_img.split()[-1])
        # Find non-zero / solid foreground pixels
        foreground_pts = cv2.findNonZero((alpha_channel > 10).astype(np.uint8))
        
        if foreground_pts is None:
            print(f"[Error] No foreground detected in {input_path.name}")
            return False

        x, y, w, h = cv2.boundingRect(foreground_pts)
        print(f"  -> Foreground detected at x={x}, y={y}, w={w}, h={h}")

        # Crop tightly to product bounding box
        cropped_product = rgba_img.crop((x, y, x + w, y + h))

        # Target box dimension based on fill_ratio
        target_fill_dim = int(canvas_size * fill_ratio)

        # Calculate proportional scale (strictly preserving aspect ratio)
        scale = min(target_fill_dim / w, target_fill_dim / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        print(f"  -> Scaling to {new_w}x{new_h} (Fill ratio: {fill_ratio:.2f}, Canvas: {canvas_size}x{canvas_size})...")
        resized_product = cropped_product.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 5. Canvas Compositing & Centering
        # Create pure white canvas (#FFFFFF)
        canvas = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 255))

        # Exact center offset
        offset_x = (canvas_size - new_w) // 2
        offset_y = (canvas_size - new_h) // 2

        # 5. Detail Enhancement & Sharpening on Product Layer
        # Applying unsharp mask to product layer avoids bleeding into the canvas background
        product_rgb = resized_product.convert("RGB")
        unsharp_filter = ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=2)
        sharpened_rgb = product_rgb.filter(unsharp_filter)
        
        # Re-attach alpha channel
        sharpened_product = Image.merge("RGBA", (*sharpened_rgb.split(), resized_product.split()[-1]))

        # 6. Canvas Compositing & Centering
        # Create pure white canvas (#FFFFFF)
        canvas = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 255))

        # Exact center offset
        offset_x = (canvas_size - new_w) // 2
        offset_y = (canvas_size - new_h) // 2

        print(f"  -> Centering on canvas at offset ({offset_x}, {offset_y})...")
        canvas.paste(sharpened_product, (offset_x, offset_y), mask=sharpened_product.split()[-1])

        # Convert to RGB (pure white background preserved)
        final_img = canvas.convert("RGB")

        # 7. Output directory and file saving
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix.lower() in [".jpg", ".jpeg"]:
            final_img.save(output_path, "JPEG", quality=95, optimize=True, subsampling=0)
        else:
            final_img.save(output_path, "PNG", optimize=True)

        print(f"[Success] Saved standardized image: {output_path}")
        return True

    except Exception as e:
        print(f"[Error] Failed to process {input_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="E-Commerce Product Image Standardization & AI Preprocessing Pipeline"
    )
    parser.add_argument(
        "-i", "--input", required=True, type=str,
        help="Path to an image file or directory of images."
    )
    parser.add_argument(
        "-o", "--output", default="./output", type=str,
        help="Output file path or directory (default: ./output)."
    )
    parser.add_argument(
        "-s", "--size", default=1500, type=int,
        help="Target square canvas dimension in pixels (default: 1500)."
    )
    parser.add_argument(
        "--fill", default=0.85, type=float,
        help="Max fill ratio inside canvas (default: 0.85)."
    )
    parser.add_argument(
        "--inpaint-stickers", action="store_true",
        help="Trigger color-threshold sticker detection and inpainting."
    )
    parser.add_argument(
        "--model", default="u2net", type=str,
        help="Rembg segmentation model name (default: u2net, options: birefnet-general, isnet-general-use, u2net)."
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
        # Single image processing
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
        # Directory batch processing
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
