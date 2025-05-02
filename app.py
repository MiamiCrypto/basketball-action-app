## app.py
import streamlit as st
from detect import run_yolo_on_frames
from pose_utils import extract_pose_from_image
import os
import shutil
from pathlib import Path
import cv2

st.set_page_config(page_title="🏀 Basketball Player Detection", layout="wide")
st.title("🏀 Basketball Player Detection")

uploaded_video = st.file_uploader("Upload a basketball video (MP4)", type=["mp4", "mov"])

if uploaded_video:
    # Save uploaded video
    temp_video_path = "temp_input.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_video.read())

    st.video(temp_video_path)

    # Prepare frame output directory
    frames_dir = "frames"
    output_dir = "outputs"
    shutil.rmtree(frames_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(frames_dir, exist_ok=True)

    # Extract every 3rd frame
    cap = cv2.VideoCapture(temp_video_path)
    i = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % 3 == 0:
            path = os.path.join(frames_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(path, frame)
            saved += 1
        i += 1
    cap.release()

    st.write(f"Extracted {saved} frames")

    # Run YOLOv8 Detection
    os.makedirs(output_dir, exist_ok=True)
    result_paths = run_yolo_on_frames(frames_dir, output_dir)

    st.success("Detection complete! Preview below:")
    for path in result_paths[:5]:
        st.image(path, caption=Path(path).name)
        
    if result_paths:
        with open(result_paths[0], "rb") as f:
            st.download_button("Download First Annotated Frame", f, "annotated_frame.jpg")
    else: 
        st.warning("No annotated frames to download.")       

    # Pose Estimation
    if st.button("🧍 Extract Pose from First Annotated Frame"):
        pose_frame, joint_coords = extract_pose_from_image(result_paths[0])
        if pose_frame is not None:
            st.image(pose_frame, caption="Pose Detected")
            st.write("Sample Joint Coordinates (normalized):")
            st.json(joint_coords[:5])
        else:
            st.warning("No pose landmarks detected.")


## detect.py
from ultralytics import YOLO
import cv2
import os
import streamlit as st

def run_yolo_on_frames(input_dir, output_dir):
    st.write("🚀 Loading YOLOv8 model...")
    try:
        model = YOLO("yolov8n.pt", task="detect")  # Load the lightweight model
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


## pose_utils.py
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

def extract_pose_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, None

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            return image, None

        # Draw landmarks
        annotated_image = image.copy()
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract coordinates
        joints = []
        for lm in results.pose_landmarks.landmark:
            joints.append((lm.x, lm.y))

        return annotated_image, joints
