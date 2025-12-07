import easyocr
import cv2
import matplotlib.pyplot as plt
import numpy as np
import re

def extract_text_easyocr_smart(image_path):
    print("Sedang memproses...")

    reader = easyocr.Reader(['en'], gpu=False)
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Deteksi Semua Teks
    results = reader.readtext(img_gray, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')


    max_height = 0
    valid_detections = []

    for (bbox, text, prob) in results:
        h = bbox[2][1] - bbox[0][1] 
        
        if h > max_height:
            max_height = h
            
    print(f"Tinggi Teks Maksimum (Acuan): {max_height} px")

    height_threshold = max_height * 0.6

    raw_texts = []
    print("\n--- DETEKSI SETELAH FILTER UKURAN ---")
    
    for (bbox, text, prob) in results:
        h = bbox[2][1] - bbox[0][1]
        
        if prob > 0.4 and h > height_threshold:
            print(f"✔ Diambil: '{text}' (Tinggi: {h}px)")
            raw_texts.append(text)
        else:
            print(f"❌ Dibuang: '{text}' (Tinggi: {h}px -> Terlalu kecil/Masa berlaku)")

    full_string = "".join(raw_texts).upper()
    clean_string = re.sub(r'[^A-Z0-9]', '', full_string)
    
    print(f"\nString Bersih: {clean_string}")

    plate_pattern = r"^([A-Z]{1,2})(\d{1,4})([A-Z]*)"
    
    match = re.search(plate_pattern, clean_string)

    if match:
        area_code = match.group(1)
        reg_number = match.group(2)
        suffix = match.group(3)

        final_plate = f"{area_code} {reg_number} {suffix}".strip()
        
        print(f"\n✔ Plat Terdeteksi: {final_plate}")
        
    else:
        final_plate = "(format tidak cocok)"
        print("\n❌ Gagal memisahkan format.")

    # Visualisasi
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Result: {final_plate}")
    plt.axis('off')
    plt.show()

    return final_plate

extract_text_easyocr_smart("C:/Users/HaikalAssyauqi/OneDrive - TSP/Pictures/Screenshots/Screenshot 2025-12-07 134340.png")