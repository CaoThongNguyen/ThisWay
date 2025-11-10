#!/usr/bin/env python3
import cv2
import sys
from src.detection import TrafficDetector


def main():
    if len(sys.argv) < 2:
        print("Usage: python detect.py <video_path> [model_path]")
        print("Example: python detect.py video.mp4")
        return

    video_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "models/yolov5m.pt"

    detector = TrafficDetector(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Không mở được video!")
        return

    print("Bắt đầu detection. Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, score, counts = detector.process_frame(frame)
        cv2.imshow("Smart Traffic Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()