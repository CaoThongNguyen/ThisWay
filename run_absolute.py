import cv2
import torch
import sys
import os
import pathlib
import numpy as np # Import numpy
from traffic_light_system import TrafficLightSystem

# ==========================================
# 👇 CẤU HÌNH ĐƯỜNG DẪN & FIX SYSTEM
# ==========================================
pathlib.PosixPath = pathlib.WindowsPath
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_DIR = os.path.join(BASE_DIR, 'yolov5')
MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')
VIDEO_PATH = os.path.join(BASE_DIR, 'vid_GiaoThong', '1-1.mp4') 

print(f"📍 Đang chạy tại: {BASE_DIR}")

if not os.path.exists(os.path.join(YOLO_DIR, 'hubconf.py')):
    print(f"❌ LỖI: Không thấy file hubconf.py trong {YOLO_DIR}")
    sys.exit()
# ==========================================

def draw_info(frame, result, count):
    # 👇 FIX LỖI READ-ONLY (QUAN TRỌNG NHẤT)
    # Kiểm tra xem ảnh có cho phép ghi không, nếu không thì tạo bản sao
    if not frame.flags.writeable:
        frame = np.array(frame) # Tạo mảng mới cho phép ghi

    # Vẽ bảng thông tin
    cv2.rectangle(frame, (10, 10), (350, 180), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (350, 180), (255, 255, 255), 1)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"So xe: {count}", (30, 50), font, 1.0, (0, 255, 255), 2)
    
    if result:
        green = result['output']['green']
        red = result['output']['red']
        msg = result['status']['message']
        
        cv2.putText(frame, f"XANH: {green}s", (30, 90), font, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"DO:   {red}s", (30, 130), font, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, msg, (30, 160), font, 0.5, (200, 200, 200), 1)
    
    return frame

def main():
    print(f"🚀 Đang tải model từ {MODEL_PATH}...")

    try:
        model = torch.hub.load(YOLO_DIR, 'custom', path=MODEL_PATH, source='local')
        model.conf = 0.5 
    except Exception as e:
        print(f"\n❌ Lỗi tải model: {e}")
        return

    print("✅ Tải model thành công! Đang mở video...")
    
    traffic_brain = TrafficLightSystem()
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"❌ Không mở được video tại: {VIDEO_PATH}")
        return

    frame_count = 0
    cached_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Hết video.")
            break

        frame = cv2.resize(frame, (1020, 600))
        
        # Nhận diện
        results = model(frame)
        
        df = results.pandas().xyxy[0]
        vehicle_count = len(df)
        
        # Lấy ảnh kết quả từ YOLO
        rendered_frame = results.render()[0]
        
        # 👇 Ép kiểu sang mảng NumPy mới để tránh lỗi Read-only
        frame_clean = np.array(rendered_frame)

        if frame_count % 5 == 0:
            cached_result = traffic_brain.calculate(num_vehicles=vehicle_count)

        # Truyền frame đã làm sạch vào hàm vẽ
        final_frame = draw_info(frame_clean, cached_result, vehicle_count)

        cv2.imshow("He thong den giao thong AI", final_frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()