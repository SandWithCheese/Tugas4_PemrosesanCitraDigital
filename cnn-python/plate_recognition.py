import easyocr
import cv2
import matplotlib.pyplot as plt
import numpy as np
import re

def extract_text_easyocr_smart(image_path):
    print("Sedang memproses...")

    reader = easyocr.Reader(['en'], gpu=False)
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Gambar tidak ditemukan!")
        return "(gambar tidak ditemukan)"

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Deteksi teks
    results = reader.readtext(img_gray, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    # Cari tinggi teks terbesar
    max_height = 0
    for (bbox, text, prob) in results:
        h = bbox[2][1] - bbox[0][1]
        max_height = max(max_height, h)

    print(f"Tinggi Teks Maksimum: {max_height}")

    height_threshold = max_height * 0.6
    raw_texts = []

    for (bbox, text, prob) in results:
        h = bbox[2][1] - bbox[0][1]
        if prob > 0.4 and h > height_threshold:
            raw_texts.append(text)

    # Gabungkan teks
    full_string = "".join(raw_texts).upper()
    clean_string = re.sub(r'[^A-Z0-9]', '', full_string)

    # Pola plat
    pattern = r"^([A-Z]{1,2})(\d{1,4})([A-Z]*)"
    match = re.search(pattern, clean_string)

    if match:
        area = match.group(1)
        number = match.group(2)
        suffix = match.group(3)
        final = f"{area} {number} {suffix}".strip()
    else:
        final = "(format tidak cocok)"

    # Visualisasi
    plt.figure(figsize=(8, 5))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"OCR: {final}")
    plt.axis('off')
    plt.show()

    return final
