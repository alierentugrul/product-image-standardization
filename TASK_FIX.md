# Bug Fix: Prevent Silhouette Distortion During Inpainting

## Problem Analysis
In the previous run, the bounding box of the sticker dilated beyond the top edge of the air conditioner chassis. The inpainter filled this background region, and `rembg` subsequently traced around the infilled artifact, creating a warped "horn" on the top-left contour.

## Strict Constraints
1. **Silhouette Lock:** The outer boundary of the product must remain 100% identical to the raw segmentation output.
2. **Contour Clamping:** The inpainting mask must be strictly clipped inside the product body (`cv2.bitwise_and` with alpha channel). The inpainting mask must NEVER reach the top boundary coordinates of the AC.
3. **Inpainting Scope:** Inpaint ONLY inside the isolated sticker bounding box, strictly constrained to `y > y_top + 30px`.

---

## Drop-in Python Script (`fix_pipeline.py`)

```python
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from rembg import remove

def standardize_and_inpaint(input_path, output_path):
    print(f"[*] İşleniyor: {input_path}")
    raw_img = Image.open(input_path).convert("RGBA")

    # 1. ADIM: Önce Arka Planı Kusursuz Ayır (Silüeti Kilitle)
    segmented = remove(raw_img)
    seg_np = np.array(segmented)
    alpha = seg_np[:, :, 3]

    # Ürünün gerçek dış sınırlarını bul
    coords = cv2.findNonZero(alpha)
    x_ac, y_ac, w_ac, h_ac = cv2.boundingRect(coords)

    # 2. ADIM: Etiket Maskesini SADECE Ürün İçinde Çıkar
    # Etiket, klimanın sol ön yüzünde yer alır. 
    # Tavan çizgisine asla yaklaşmaması için y başlangıcını güvenli mesafede tutuyoruz.
    y1 = y_ac + int(h_ac * 0.18)
    y2 = y_ac + int(h_ac * 0.70)
    x1 = x_ac + int(w_ac * 0.02)
    x2 = x_ac + int(w_ac * 0.25)

    roi_rgb = seg_np[y1:y2, x1:x2, :3]
    roi_hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)

    # Renkli barlar (yeşil/sarı/kırmızı)
    sat_mask = cv2.inRange(roi_hsv, np.array([0, 35, 40]), np.array([180, 255, 255]))
    # QR kod ve metin kenarları
    edges = cv2.Canny(roi_gray, 40, 120)
    combined = cv2.bitwise_or(sat_mask, edges)

    # Etiketi tek blok haline getir
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(combined, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    inpaint_mask = np.zeros(alpha.shape, dtype=np.uint8)
    if contours:
        c = max(contours, key=cv2.contourArea)
        rx, ry, rw, rh = cv2.boundingRect(c)
        pad = 6
        bx = max(x1, x1 + rx - pad)
        by = max(y1, y1 + ry - pad)
        bw = min(x2 - bx, rw + pad * 2)
        bh = min(y2 - by, rh + pad * 2)

        # Maskeyi çiz
        inpaint_mask[by:by+bh, bx:bx+bw] = 255
        
        # KRİTİK KONTROL: Maskeyi ürünün orijinal alfa sınırıyla kes
        # Asla klimanın dışına tek bir piksel taşamaz!
        inpaint_mask = cv2.bitwise_and(inpaint_mask, alpha)

    # 3. ADIM: Inpaint Uygula (Sadece Etiket Alanına)
    rgb_bgr = cv2.cvtColor(seg_np[:, :, :3], cv2.COLOR_RGB2BGR)
    
    try:
        from simple_lama_inpainting import SimpleLama
        lama = SimpleLama()
        pil_img = Image.fromarray(seg_np[:, :, :3])
        pil_mask = Image.fromarray(inpaint_mask)
        clean_rgb = np.array(lama(pil_img, pil_mask))
    except Exception:
        # Fallback Telea
        clean_bgr = cv2.inpaint(rgb_bgr, inpaint_mask, 7, cv2.INPAINT_TELEA)
        clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)

    # Temizlenmiş gövdeyi orijinal alfa maskesiyle tekrar birleştir
    clean_rgba = np.dstack((clean_rgb, alpha))
    clean_pil = Image.fromarray(clean_rgba)

    # 4. ADIM: Oran Korumalı Ölçekle ve 1500x1500px Tuvalde Tam Merkeze Yerleştir
    cropped = clean_pil.crop((x_ac, y_ac, x_ac + w_ac, y_ac + h_ac))
    
    canvas_size = (1500, 1500)
    fill_ratio = 0.85
    max_w = int(canvas_size[0] * fill_ratio)
    max_h = int(canvas_size[1] * fill_ratio)
    cropped.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", canvas_size, (255, 255, 255, 255))
    offset_x = (canvas_size[0] - cropped.size[0]) // 2
    offset_y = (canvas_size[1] - cropped.size[1]) // 2
    canvas.paste(cropped, (offset_x, offset_y), mask=cropped)

    # 5. ADIM: Kontrast ve Keskinlik
    final = canvas.convert("RGB")
    enhancer = ImageEnhance.Sharpness(final)
    output = enhancer.enhance(1.25)
    
    output.save(output_path, "JPEG", quality=98)
    print(f"[✓] Kusursuz çıktı hazırlandı: {output_path}")

if __name__ == "__main__":
    standardize_and_inpaint("DoraNEwinverterAC_1.webp", "Dora_ECommerce_Fixed.jpg")