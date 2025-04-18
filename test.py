from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

# Tạo đối tượng pose detector
base_options = python.BaseOptions(model_asset_path=r'E:\PROJECT_GITHUB\Project_detection_body\pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# Mở webcam hoặc video
use_webcam = True  # Đổi thành False nếu bạn muốn dùng video file
video_path = r"E:\PROJECT_GITHUB\Project_detection_body\a.mp4"  # Đường dẫn video nếu dùng file

cap = cv2.VideoCapture(video_path)

# Thiết lập kích thước mới cho video (ví dụ 640x480)
output_width = 640
output_height = 480

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Không đọc được frame từ camera/video.")
        break

    # Thay đổi kích thước frame
    frame_resized = cv2.resize(frame, (output_width, output_height))

    rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)
    annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)

    # Hiển thị frame đã được thay đổi kích thước
    cv2.imshow("Pose Detection", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
