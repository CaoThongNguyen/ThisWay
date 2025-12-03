import cv2
import torch
import sys
import os
import pathlib
import numpy as np
from traffic_light_system import TrafficLightSystem
# 👇 IMPORT THÊM THƯ VIỆN PILLOW ĐỂ VẼ TIẾNG VIỆT
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# 👇 CẤU HÌNH ĐƯỜNG DẪN & FIX SYSTEM
# ==========================================
pathlib.PosixPath = pathlib.WindowsPath
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_DIR = os.path.join(BASE_DIR, 'yolov5')
MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')
VIDEO_PATH = os.path.join(BASE_DIR, 'vid_GiaoThong', '1-3.mp4')
# 👇 Lấy trực tiếp font Arial từ hệ thống Windows (Chắc chắn có)
FONT_PATH = r"C:\Windows\Fonts\arial.ttf"

print(f"📍 Đang chạy tại: {BASE_DIR}")

if not os.path.exists(os.path.join(YOLO_DIR, 'hubconf.py')):
    print(f"❌ LỖI: Không thấy file hubconf.py trong {YOLO_DIR}")
    sys.exit()
# ==========================================

# 👇 HÀM MỚI: VẼ CHỮ TIẾNG VIỆT BẰNG PILLOW
def draw_text_vi(img, text, pos, font_path, font_size, color):
    # 1. Chuyển ảnh từ OpenCV (BGR) sang Pillow (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 2. Tải font chữ
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        # Nếu không tìm thấy font, dùng font mặc định (sẽ không có tiếng Việt đẹp)
        print(f"⚠️ Cảnh báo: Không tìm thấy file font {font_path}. Đang dùng font mặc định.")
        font = ImageFont.load_default()

    # 3. Vẽ chữ lên ảnh
    draw.text(pos, text, font=font, fill=color)

    # 4. Chuyển ngược lại từ Pillow (RGB) sang OpenCV (BGR)
    img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv2

def draw_info(frame, result, count):
    # Fix lỗi read-only
    if not frame.flags.writeable:
        frame = np.array(frame)

    # Vẽ bảng thông tin nền đen
    cv2.rectangle(frame, (10, 10), (380, 200), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (380, 200), (255, 255, 255), 1)
    
    # 👇 SỬA LẠI MÀU SẮC CHUẨN (RGB) CHO PILLOW
    # Màu Vàng (R=255, G=255, B=0)
    frame = draw_text_vi(frame, f"Số xe: {count}", (30, 30), FONT_PATH, 32, (255, 255, 0))
    
    if result:
        green = result['output']['green']
        red = result['output']['red']
        msg = result['status']['message']
        
        # Màu Xanh Lá (R=0, G=255, B=0)
        frame = draw_text_vi(frame, f"XANH: {green}s", (30, 80), FONT_PATH, 28, (0, 255, 0))
        # Màu Đỏ (R=255, G=0, B=0) -> Sửa chỗ này
        frame = draw_text_vi(frame, f"ĐỎ:    {red}s", (30, 120), FONT_PATH, 28, (255, 0, 0))
        
        # Màu Xám trắng
        frame = draw_text_vi(frame, msg, (30, 160), FONT_PATH, 20, (200, 200, 200))
    
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
        
        rendered_frame = results.render()[0]
        frame_clean = np.array(rendered_frame)

        if frame_count % 5 == 0:
            cached_result = traffic_brain.calculate(num_vehicles=vehicle_count)

        # Truyền frame vào hàm vẽ thông tin mới
        final_frame = draw_info(frame_clean, cached_result, vehicle_count)

        cv2.imshow("He thong den giao thong AI", final_frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()