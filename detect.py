import cv2
import os
from ultralytics import YOLO
import easyocr
from datetime import datetime
import pandas as pd

# === CONFIGURATION ===
video_path = r'C:\Users\mouad.DESKTOP-VVUTG6U\Downloads\nohe.mp4'
model_path = 'best.pt'  # Path to your trained YOLOv8 model
output_folder = 'violations'
csv_path = 'violations.csv'

os.makedirs(output_folder, exist_ok=True)
reader = easyocr.Reader(['en'])

# === LOAD MODEL ===
model = YOLO(model_path)

# === OPEN VIDEO ===
cap = cv2.VideoCapture(video_path)
frame_count = 0
violation_count = 0
log_data = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Run YOLO inference
    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = result.names[cls]

            if class_name == 'no_helmet' and conf > 0.5:
                # Annotate the frame with bounding boxes
                annotated_frame = result.plot()

                # Save annotated frame
                timestamp = datetime.now()
                filename = f"violation_{violation_count+1}.jpg"
                frame_path = os.path.join(output_folder, filename)
                cv2.imwrite(frame_path, annotated_frame)


                # Run OCR on entire frame (or optionally crop plate region)
                ocr_result = reader.readtext(frame)
                plate_number = ocr_result[0][1] if ocr_result else "UNKNOWN"

                # Log the violation
                log_data.append({
                    'image': filename,
                    'plate': plate_number,
                    'year': timestamp.year,
                    'month': timestamp.month,
                    'day': timestamp.day,
                    'hour': timestamp.hour,
                    'minute': timestamp.minute
                })

                violation_count += 1
                break  # Save only once per frame

# Save CSV log
df = pd.DataFrame(log_data)
df.to_csv(csv_path, index=False)

cap.release()
print(f"[✅] Done. {violation_count} violations saved to {output_folder} and logged in {csv_path}.")
