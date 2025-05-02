## app.py
import streamlit as st
from detect import run_yolo_on_frames
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

    with open(result_paths[0], "rb") as f:
        st.download_button("Download First Annotated Frame", f, "annotated_frame.jpg")
