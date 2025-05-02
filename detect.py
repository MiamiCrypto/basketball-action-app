## detect.py
from ultralytics import YOLO
import cv2
import os
import streamlit as st

def run_yolo_on_frames(input_dir, output_dir):
    st.write("🚀 Loading YOLOv8 model...")
    try:
        model = YOLO("yolov8n.pt")  # Load YOLOv8n (lightweight version)
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

        for box in detections.boxes:
            cls = int(box.cls[0])
            if cls != 0:
                continue  # only detect people

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_height = y2 - y1
            box_center_y = (y1 + y2) // 2

            # Rule 1: Skip small boxes (likely distant fans)
            if box_height < 100:
                continue

            # Rule 2: Skip people low on screen (likely fans)
            if box_center_y > frame.shape[0] * 0.85:
                continue

            # Rule 3 (optional): Filter out referees by average gray color
            roi = frame[y1:y2, x1:x2]
            avg_color = roi.mean(axis=(0, 1)) if roi.size else [0, 0, 0]

            # Filter out gray (referee-like) colors
            if 90 < avg_color[0] < 160 and 90 < avg_color[1] < 160 and 90 < avg_color[2] < 160:
                continue

            # Passed all filters, label as Player
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Player", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        output_path = os.path.join(output_dir, file)
        cv2.imwrite(output_path, frame)
        results.append(output_path)

    st.success("✅ Detection complete.")
    return results
