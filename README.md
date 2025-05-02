# 🏀 Basketball Action App

This is a Streamlit-based computer vision application for detecting basketball players in game footage using YOLOv8. It allows you to upload a video clip, extract frames, run player detection, and preview/download annotated images.

## 🚀 Features

- Upload basketball video clips (MP4 or MOV)
- Extract every 3rd frame for efficient processing
- Detect players in each frame using YOLOv8
- Annotate frames with bounding boxes labeled "Player"
- Preview results in the browser
- Download sample annotated frame

## 🧰 Technologies

- [Streamlit](https://streamlit.io/)
- [YOLOv8 (Ultralytics)](https://docs.ultralytics.com/)
- OpenCV
- Python

## 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/basketball-action-app.git
cd basketball-action-app
pip install -r requirements.txt
streamlit run app.py
