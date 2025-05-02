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
