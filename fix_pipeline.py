"""
Standalone reference script implementing Silhouette Lock & Contour Clamping as specified in TASK_FIX.md.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import rembg


def standardize_and_inpaint(input_path: str, output_path: str, canvas_size: int = 1500, fill_ratio: float = 0.85):
    print(f"[*] İşleniyor: {input_path}")
    raw_img = Image.open(input_path).convert("RGBA")

    # 1. ADIM: Önce Arka Planı Kusursuz Ayır (Silüeti Kilitle)
    print("  -> 1. Adım: rembg ile silüet kilitleniyor...")
    session = rembg.new_session("u2net")
    segmented = rembg.remove(raw_img, session=session)
    seg_np = np.array(segmented)
    alpha = seg_np[:, :, 3]

    # Faint shadow noise temizle
    alpha = np.where(alpha > 15, alpha, 0).astype(np.uint8)
    seg_np[:, :, 3] = alpha

    # Ürünün gerçek dış sınırlarını bul
    coords = cv2.findNonZero(alpha)
    if coords is None:
        raise ValueError(f"Ön plan tespit edilemedi: {input_path}")

    x_ac, y_ac, w_ac, h_ac = cv2.boundingRect(coords)
    print(f"  -> Kilitlenen ürün sınırlayıcı kutusu: x={x_ac}, y={y_ac}, w={w_ac}, h={h_ac}")

    # 2. ADIM: Etiket Maskesini SADECE Ürün İçinde Çıkar
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
        print(f"  -> Etiket maskesi kilitlendi ve kenetlendi: x={bx}, y={by}, w={bw}, h={bh}")

    # 3. ADIM: Inpaint Uygula (Sadece Etiket Alanına)
    print("  -> 3. Adım: Inpaint uygulanıyor...")
    rgb_bgr = cv2.cvtColor(seg_np[:, :, :3], cv2.COLOR_RGB2BGR)

    try:
        from simple_lama_inpainting import SimpleLama
        lama = SimpleLama()
        pil_img = Image.fromarray(seg_np[:, :, :3])
        pil_mask = Image.fromarray(inpaint_mask)
        clean_rgb = np.array(lama(pil_img, pil_mask))
        print("  -> Inpainting LaMa modeli ile tamamlandı.")
    except Exception as e:
        print(f"  -> LaMa kullanılamadı ({e}), Telea fallback uygulanıyor...")
        clean_bgr = cv2.inpaint(rgb_bgr, inpaint_mask, 7, cv2.INPAINT_TELEA)
        clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)

    # Temizlenmiş gövdeyi orijinal kilitli alfa maskesiyle tekrar birleştir
    clean_rgba = np.dstack((clean_rgb, alpha))
    clean_pil = Image.fromarray(clean_rgba)

    # 4. ADIM: Oran Korumalı Ölçekle ve Tuvalde Tam Merkeze Yerleştir
    cropped = clean_pil.crop((x_ac, y_ac, x_ac + w_ac, y_ac + h_ac))

    target_dim = (canvas_size, canvas_size)
    max_dim = int(canvas_size * fill_ratio)
    cropped.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", target_dim, (255, 255, 255, 255))
    offset_x = (canvas_size - cropped.size[0]) // 2
    offset_y = (canvas_size - cropped.size[1]) // 2
    canvas.paste(cropped, (offset_x, offset_y), mask=cropped)

    # 5. ADIM: Kontrast ve Keskinlik
    final = canvas.convert("RGB")
    enhancer = ImageEnhance.Sharpness(final)
    output = enhancer.enhance(1.25)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    output.save(output_path, "JPEG", quality=98)
    print(f"[SUCCESS] Kusursuz çıktı hazırlandı: {output_path}")


if __name__ == "__main__":
    standardize_and_inpaint("DoraNEwinverterAC_1.webp", "output/Dora_ECommerce_Fixed.jpg")
