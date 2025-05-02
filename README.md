# 🏀 Basketball Action App

Basketball Action Detection App

This application allows you to upload a basketball game video, detect players on the court, and annotate their positions frame-by-frame using YOLOv8.

📂 Folder Structure

basketball-action-detector/
├── app.py                  # Streamlit UI
├── detect.py               # Detection logic (YOLOv8)
├── utils.py                # Frame extraction and helper tools
├── model/                  # YOLOv8 model weights (yolov8n.pt)
├── test_videos/            # 🟡 Place your video files here
├── outputs/                # Annotated frame results
├── requirements.txt        # Dependencies
└── README.md               # This file

🚀 How to Use

Use the Built-in Sample Video or Upload Your OwnThe app includes a built-in sample clip (sample_clip.mp4) located in the test_videos/ folder. This video will load automatically if no upload is provided.

Alternatively, you may upload your own .mp4 basketball clip directly through the app interface.

Run the Streamlit App

streamlit run app.py

Upload a Clip

Choose your .mp4 file from the test_videos/ folder

The app will extract frames and perform YOLO detection

Download/Preview Results

You can download the first annotated frame

Optionally extract pose landmarks from it (coming soon)

🔧 About the Download / Pose Buttons

Download First Annotated Frame: Saves the first detection result so you can verify the labeling quickly without downloading all outputs.

Extract Pose from First Annotated Frame (WIP): Will eventually run pose estimation (e.g., MediaPipe) on that same frame to classify actions like dribble, walk, or run.
