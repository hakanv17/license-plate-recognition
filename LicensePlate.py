# Gerekli kütüphaneler:
# pip install paddleocr
# pip install paddlepaddle -f https://paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
# pip install ultralytics opencv-python matplotlib

import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Model yolları
model_path = "/Users/hakanvanli/Desktop/Plaka Okuma/best.pt"
model = YOLO(model_path)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Kamera açılıyor
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Camera didnt open!")

print("License Plate System. Press 'q' to quit")

frame_count = 0
process_every_n_frame = 10  # Her 10 karede bir YOLO+OCR uygula
scale_factor = 1.5          # OCR için yeniden boyutlandırma oranı
padding = 10                # Plaka kenarlarına ek boşluk

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü küçült (hız için)
    frame = cv2.resize(frame, (640, 480))
    frame_count += 1

    if frame_count % process_every_n_frame == 0:
        results = model.predict(source=frame, save=False, verbose=False)
        xyxy = results[0].boxes.xyxy.cpu().numpy().astype(int)

        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = box

            # Kırpılacak alan
            x1_pad = max(x1 - padding, 0)
            y1_pad = max(y1 - padding, 0)
            x2_pad = min(x2 + padding, frame.shape[1])
            y2_pad = min(y2 + padding, frame.shape[0])

            crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            if crop.size == 0:
                continue

            h, w = crop.shape[:2]
            new_size = (int(w * scale_factor), int(h * scale_factor))
            crop_resized = cv2.resize(crop, new_size, interpolation=cv2.INTER_LINEAR)

            # OCR işlemi
            result = ocr.predict(crop_resized)
            rec_texts = result[0].get('rec_texts', [])
            rec_scores = result[0].get('rec_scores', [])

            import re  # dosyanın başında varsa tekrar eklemene gerek yok

            raw_text = "".join(rec_texts).replace(" ", "")
            combined_text = re.sub(r'[^A-Za-z0-9]', '', raw_text)
            avg_score = sum(rec_scores) / len(rec_scores) if rec_scores else 0

            # Yalnızca güven skoru yüksek olanlar
            if combined_text and avg_score >= 0.90:
              print(f"Plaka: {combined_text} | Güven: {avg_score:.2f}")
              cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
              cv2.putText(frame, combined_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Kamerayı göster
    cv2.imshow("Live License Plate Stream 'q' for quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
