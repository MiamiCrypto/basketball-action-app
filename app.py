import streamlit as st
from detect import run_yolo_on_frames
import os
import shutil

st.title("🏀 Basketball Player Detection")

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov"])
if uploaded_video:
    temp_video = "temp_input.mp4"
    with open(temp_video, "wb") as f:
        f.write(uploaded_video.read())

    st.video(temp_video)

    # Extract every 3rd frame (you can call your frame_extractor function here)
    # Then run detection
    detected_frames = run_yolo_on_frames("frames/", "outputs/")

    st.success("Detection complete. Sample output:")
    for path in detected_frames[:5]:
        st.image(path)

    with open(detected_frames[0], "rb") as f:
        st.download_button("Download First Annotated Frame", f, "annotated_frame.jpg")
