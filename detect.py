## detect.py
from ultralytics import YOLO
import cv2
import os
import streamlit as st

def run_yolo_on_frames(input_dir, output_dir):
    st.write("🚀 Loading YOLOv8 model...")
    try:
        model = YOLO.from_pretrained("yolov8n.pt")  # Safe model loading in PyTorch 2.6+
    except Exception as e:
        st.error(f"❌ Failed to load YOLO model: {e}")
        return []

    frame_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")])
    results = []

    if not frame_files:
        st.warning("⚠️ No frames found in the input directory.")
        return []

    st.write(f"📸 Processing {len(frame_files)} frames for player detection...")

    for file in frame_files:
        input_path = os.path.join(input_dir, file)
        frame = cv2.imread(input_path)
        if frame is None:
            st.warning(f"⚠️ Could not read frame: {file}")
            continue

        try:
            detections = model(frame)[0]
        except Exception as e:
            st.error(f"❌ Detection failed on {file}: {e}")
            continue

        person_found = False
        for box in detections.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # Class 0 = person
                person_found = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Player", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if not person_found:
            st.info(f"No players detected in frame: {file}")

        output_path = os.path.join(output_dir, file)
        cv2.imwrite(output_path, frame)
        results.append(output_path)

    st.success("✅ Detection complete.")
    return results
