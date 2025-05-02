import streamlit as st
import os
from detect import run_yolo_on_frames
import shutil
import uuid

st.set_page_config(page_title="Basketball Player Detection", layout="centered")
st.title("🏀 Basketball Player Detection")

# Create working directories
TEMP_DIR = "temp"
FRAME_DIR = os.path.join(TEMP_DIR, "frames")
OUTPUT_DIR = os.path.join("outputs")
os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Choose between sample video and upload
st.subheader("Select Video Source")
video_source = st.radio("Choose a video input:", ["Use built-in Lakers clip", "Upload your own video"])

if video_source == "Upload your own video":
    uploaded_file = st.file_uploader("Upload a basketball video", type=["mp4"])
    if uploaded_file is not None:
        video_id = str(uuid.uuid4())
        video_path = os.path.join(TEMP_DIR, f"uploaded_{video_id}.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        st.video(video_path)
    else:
        st.stop()
else:
    video_path = "test_videos/Lakers_short_clip.mp4"
    st.info("Using built-in Lakers sample clip.")
    st.video(video_path)

# Process button
if st.button("Run Detection"):
    st.write("📽️ Extracting frames from video...")

    # Extract frames using OpenCV
    import cv2
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(FRAME_DIR, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    cap.release()
    st.success(f"Extracted {frame_count} frames")

    # Run detection
    result_paths = run_yolo_on_frames(FRAME_DIR, OUTPUT_DIR)

    if result_paths:
        st.success("Detection complete! Preview below:")
        for i, path in enumerate(result_paths[:5]):
            st.image(path, caption=f"Annotated Frame {i+1}", use_column_width=True)

        with open(result_paths[0], "rb") as f:
            st.download_button("📥 Download First Annotated Frame", f, file_name="annotated_frame.jpg")

        if st.button("🕴️ Extract Pose from First Annotated Frame"):
            st.warning("Pose extraction is coming soon.")
    else:
        st.warning("No annotated frames to display or download.")

# Cleanup function on rerun
if st.button("Clear Temporary Files"):
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    os.makedirs(FRAME_DIR, exist_ok=True)
    st.info("Temporary files cleared.")

