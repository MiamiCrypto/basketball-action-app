## detect.py
from ultralytics import YOLO
import cv2
import os

def run_yolo_on_frames(input_dir, output_dir):
    model = YOLO("yolov8n.pt")  # Load the lightweight model

    frame_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")])
    results = []

    for file in frame_files:
        input_path = os.path.join(input_dir, file)
        frame = cv2.imread(input_path)
        detections = model(frame)[0]

        for box in detections.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # Class 0 = person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Player", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        output_path = os.path.join(output_dir, file)
        cv2.imwrite(output_path, frame)
        results.append(output_path)

    return results
