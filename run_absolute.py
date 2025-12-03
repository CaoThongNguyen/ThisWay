"""
run_absolute.py - HỆ THỐNG ĐÈN GIAO THÔNG AI
Sử dụng YOLOv5 custom model + Traffic Light System

Phím tắt:
    q - Thoát
    p - Pause/Resume
    s - Screenshot
    r - Reset statistics
"""

import cv2
import torch
import sys
import os
import pathlib
import numpy as np
from datetime import datetime
from traffic_light_system import TrafficLightSystem

# ==========================================
# 👇 CẤU HÌNH ĐƯỜNG DẪN & FIX SYSTEM
# ==========================================
pathlib.PosixPath = pathlib.WindowsPath
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_DIR = os.path.join(BASE_DIR, 'yolov5')
MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')
VIDEO_PATH = os.path.join(BASE_DIR, 'vid_GiaoThong', '5-2.mp4') 

print("=" * 70)
print("TRAFFIC LIGHT AI SYSTEM - YOLOv5 + QCVN 41:2019")
print("=" * 70)
print(f"Running at: {BASE_DIR}")

# Kiểm tra file cần thiết
if not os.path.exists(os.path.join(YOLO_DIR, 'hubconf.py')):
    print(f"ERROR: hubconf.py not found in {YOLO_DIR}")
    sys.exit(1)

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    sys.exit(1)

if not os.path.exists(VIDEO_PATH):
    print(f"ERROR: Video not found at {VIDEO_PATH}")
    sys.exit(1)

print("All files checked OK")
# ==========================================


def draw_traffic_light_box(frame, x, y, active_light='red'):
    """
    Vẽ cột đèn giao thông 3D đẹp mắt
    
    Args:
        frame: Frame để vẽ
        x, y: Tọa độ góc trên trái
        active_light: 'red', 'yellow', 'green'
    """
    # Kích thước
    box_width = 80
    box_height = 200
    light_radius = 25
    
    # Vẽ cột đèn (shadow + border 3D)
    # Shadow
    cv2.rectangle(frame, (x + 5, y + 5), (x + box_width + 5, y + box_height + 5), (50, 50, 50), -1)
    
    # Nền chính - gradient effect (vẽ nhiều lớp)
    for i in range(10):
        alpha = 1.0 - (i * 0.05)
        color_val = int(30 + i * 3)
        cv2.rectangle(frame, (x + i, y + i), 
                     (x + box_width - i, y + box_height - i), 
                     (color_val, color_val, color_val), 2)
    
    # Border ngoài sáng bóng
    cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), (100, 100, 100), 3)
    cv2.rectangle(frame, (x + 2, y + 2), (x + box_width - 2, y + box_height - 2), (180, 180, 180), 1)
    
    # Vị trí 3 đèn
    center_x = x + box_width // 2
    positions = {
        'red': y + 40,
        'yellow': y + 100,
        'green': y + 160
    }
    
    # Vẽ từng đèn với hiệu ứng phát sáng
    lights_config = {
        'red': ((0, 0, 255), (0, 0, 100)),      # (active_color, dim_color)
        'yellow': ((0, 255, 255), (0, 100, 100)),
        'green': ((0, 255, 0), (0, 100, 0))
    }
    
    for light_name, (active_col, dim_col) in lights_config.items():
        center_y = positions[light_name]
        is_active = (light_name == active_light)
        
        if is_active:
            # Hiệu ứng phát sáng (glow effect) - vẽ nhiều vòng tròn mờ dần
            for i in range(5, 0, -1):
                glow_radius = light_radius + (i * 5)
                alpha = 0.1 * i
                overlay = frame.copy()
                cv2.circle(overlay, (center_x, center_y), glow_radius, active_col, -1)
                frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
            
            # Đèn chính - sáng
            cv2.circle(frame, (center_x, center_y), light_radius, active_col, -1)
            
            # Highlight (điểm sáng phản chiếu)
            cv2.circle(frame, (center_x - 8, center_y - 8), 6, (255, 255, 255), -1)
            cv2.circle(frame, (center_x - 8, center_y - 8), 6, active_col, 1)
        else:
            # Đèn tắt - tối
            cv2.circle(frame, (center_x, center_y), light_radius, dim_col, -1)
        
        # Border đèn
        cv2.circle(frame, (center_x, center_y), light_radius, (80, 80, 80), 2)
        cv2.circle(frame, (center_x, center_y), light_radius - 2, (120, 120, 120), 1)
    
    return frame


def draw_enhanced_info(frame, result, count, fps=0, frame_num=0, paused=False, active_light='green'):
    """
    Vẽ thông tin chi tiết lên frame với đèn giao thông đẹp
    
    Args:
        frame: Frame gốc (từ YOLO render)
        result: Kết quả từ TrafficLightSystem
        count: Số xe phát hiện
        fps: FPS hiện tại
        frame_num: Số frame hiện tại
        paused: Trạng thái pause
        active_light: Đèn đang sáng ('green', 'yellow', 'red')
    """
    # Fix lỗi read-only
    if not frame.flags.writeable:
        frame = np.array(frame)
    
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # ============================================
    # ĐÈN GIAO THÔNG 3D BÊN TRÁI (dùng active_light từ tham số)
    # ============================================
    traffic_light_x = 20
    traffic_light_y = 20
    
    frame = draw_traffic_light_box(frame, traffic_light_x, traffic_light_y, active_light)
    
    # ============================================
    # PANEL THÔNG TIN BÊN PHẢI ĐÈN
    # ============================================
    info_x = 120
    info_y = 20
    panel_width = 350
    panel_height = 220
    
    # Vẽ nền panel với gradient
    overlay = frame.copy()
    
    # Gradient background (vẽ nhiều lớp)
    for i in range(0, panel_height, 10):
        alpha = 0.8 - (i / panel_height) * 0.3
        gray = int(20 + (i / panel_height) * 30)
        cv2.rectangle(overlay, (info_x, info_y + i), 
                     (info_x + panel_width, info_y + i + 10), 
                     (gray, gray, gray), -1)
    
    frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
    
    # Border với hiệu ứng 3D
    cv2.rectangle(frame, (info_x, info_y), (info_x + panel_width, info_y + panel_height), (100, 100, 100), 3)
    cv2.rectangle(frame, (info_x + 2, info_y + 2), 
                 (info_x + panel_width - 2, info_y + panel_height - 2), (180, 180, 180), 1)
    
    # Header với background
    cv2.rectangle(frame, (info_x + 10, info_y + 10), 
                 (info_x + panel_width - 10, info_y + 40), (50, 50, 50), -1)
    cv2.putText(frame, "TRAFFIC CONTROL SYSTEM", (info_x + 20, info_y + 32), 
               font, 0.65, (0, 255, 255), 2)
    
    y_pos = info_y + 65
    
    # Số xe với icon
    cv2.rectangle(frame, (info_x + 15, y_pos - 20), (info_x + 45, y_pos + 5), (0, 100, 200), -1)
    cv2.putText(frame, "CAR", (info_x + 18, y_pos - 2), font, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"Vehicles: {count}", (info_x + 55, y_pos), font, 0.75, (255, 255, 255), 2)
    y_pos += 35
    
    # Thông tin đèn
    if result:
        green = result['output']['green']
        yellow = result['output']['yellow']
        red = result['output']['red']
        cycle = result['output']['cycle']
        
        # Đèn xanh
        cv2.circle(frame, (info_x + 30, y_pos - 5), 8, (0, 255, 0), -1)
        cv2.circle(frame, (info_x + 30, y_pos - 5), 8, (255, 255, 255), 1)
        cv2.putText(frame, f"GREEN:  {green:>5.1f}s", (info_x + 50, y_pos), font, 0.65, (150, 255, 150), 2)
        y_pos += 30
        
        # Đèn vàng
        cv2.circle(frame, (info_x + 30, y_pos - 5), 8, (0, 255, 255), -1)
        cv2.circle(frame, (info_x + 30, y_pos - 5), 8, (255, 255, 255), 1)
        cv2.putText(frame, f"YELLOW: {yellow:>5.1f}s", (info_x + 50, y_pos), font, 0.65, (150, 255, 255), 2)
        y_pos += 30
        
        # Đèn đỏ
        cv2.circle(frame, (info_x + 30, y_pos - 5), 8, (0, 0, 255), -1)
        cv2.circle(frame, (info_x + 30, y_pos - 5), 8, (255, 255, 255), 1)
        cv2.putText(frame, f"RED:    {red:>5.1f}s", (info_x + 50, y_pos), font, 0.65, (150, 150, 255), 2)
        y_pos += 30
        
        # Chu kỳ
        cv2.circle(frame, (info_x + 30, y_pos - 5), 8, (200, 200, 200), -1)
        cv2.circle(frame, (info_x + 30, y_pos - 5), 8, (255, 255, 255), 1)
        cv2.putText(frame, f"CYCLE:  {cycle:>5.1f}s", (info_x + 50, y_pos), font, 0.65, (220, 220, 220), 2)
        y_pos += 35
        
        # Status với màu nền
        msg = result['status']['message']
        level = result['status']['level']
        
        # Chuyển message sang tiếng Anh
        msg_mapping = {
            'QUA TAI': 'OVERLOAD',
            'DONG XE': 'HEAVY',
            'THONG THOANG': 'LIGHT',
            'BINH THUONG': 'NORMAL'
        }
        msg_en = msg_mapping.get(msg, msg)
        
        if level == 'critical':
            bg_color = (0, 0, 150)
            text_color = (100, 100, 255)
        elif level == 'warning':
            bg_color = (0, 100, 150)
            text_color = (100, 200, 255)
        else:
            bg_color = (0, 100, 0)
            text_color = (100, 255, 100)
        
        cv2.rectangle(frame, (info_x + 15, y_pos - 20), 
                     (info_x + panel_width - 15, y_pos + 5), bg_color, -1)
        cv2.putText(frame, msg_en[:28], (info_x + 20, y_pos - 2), font, 0.5, text_color, 1)
    
    # ============================================
    # PANEL PHẢI TRÊN - SYSTEM INFO (Compact)
    # ============================================
    sys_x = w - 280
    sys_y = 20
    sys_width = 260
    sys_height = 140
    
    # Nền gradient
    overlay_sys = frame.copy()
    for i in range(0, sys_height, 10):
        alpha_val = 0.75 - (i / sys_height) * 0.2
        gray_val = int(25 + (i / sys_height) * 25)
        cv2.rectangle(overlay_sys, (sys_x, sys_y + i), 
                     (sys_x + sys_width, sys_y + i + 10), 
                     (gray_val, gray_val, gray_val), -1)
    frame = cv2.addWeighted(overlay_sys, 0.8, frame, 0.2, 0)
    
    # Border
    cv2.rectangle(frame, (sys_x, sys_y), (sys_x + sys_width, sys_y + sys_height), (120, 120, 120), 2)
    cv2.rectangle(frame, (sys_x + 2, sys_y + 2), (sys_x + sys_width - 2, sys_y + sys_height - 2), (200, 200, 200), 1)
    
    # Header
    cv2.rectangle(frame, (sys_x + 8, sys_y + 8), (sys_x + sys_width - 8, sys_y + 35), (60, 60, 60), -1)
    cv2.putText(frame, "SYSTEM INFO", (sys_x + 15, sys_y + 27), font, 0.6, (100, 200, 255), 2)
    
    sys_info_y = sys_y + 60
    
    # FPS với màu động
    fps_color = (0, 255, 0) if fps > 20 else (0, 165, 255) if fps > 10 else (0, 0, 255)
    cv2.putText(frame, f"FPS:", (sys_x + 15, sys_info_y), font, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, f"{fps:.1f}", (sys_x + 80, sys_info_y), font, 0.65, fps_color, 2)
    sys_info_y += 28
    
    # Frame number
    cv2.putText(frame, f"Frame:", (sys_x + 15, sys_info_y), font, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"{frame_num}", (sys_x + 90, sys_info_y), font, 0.5, (150, 200, 255), 1)
    sys_info_y += 25
    
    # Time
    current_time = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, f"Time:", (sys_x + 15, sys_info_y), font, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, current_time, (sys_x + 75, sys_info_y), font, 0.5, (255, 200, 100), 1)
    
    # Time mode badge (ở góc dưới panel)
    if result:
        time_mode = result['input']['time_mode_name']
        badge_y = sys_y + sys_height - 25
        
        # Chuyển sang tiếng Anh để tránh lỗi font
        mode_mapping = {
            'Gio thap diem': 'LOW TRAFFIC',
            'Gio binh thuong': 'NORMAL',
            'Gio cao diem': 'RUSH HOUR'
        }
        time_mode_en = mode_mapping.get(time_mode, time_mode)
        
        # Màu badge theo time mode
        if 'cao' in time_mode or 'RUSH' in time_mode_en:
            badge_color = (0, 0, 200)
            text_color = (150, 150, 255)
        elif 'thap' in time_mode or 'LOW' in time_mode_en:
            badge_color = (100, 100, 0)
            text_color = (200, 200, 150)
        else:
            badge_color = (0, 100, 100)
            text_color = (150, 255, 255)
        
        cv2.rectangle(frame, (sys_x + 10, badge_y - 18), (sys_x + sys_width - 10, badge_y + 5), badge_color, -1)
        cv2.putText(frame, time_mode_en, (sys_x + 15, badge_y - 2), font, 0.45, text_color, 1)
    
    # ============================================
    # PAUSE INDICATOR
    # ============================================
    if paused:
        # Hiển thị chữ PAUSED lớn ở giữa màn hình
        pause_text = "PAUSED"
        text_size = cv2.getTextSize(pause_text, font, 2, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h // 2
        
        # Nền cho text
        cv2.rectangle(frame, (text_x - 20, text_y - 50), 
                     (text_x + text_size[0] + 20, text_y + 20), (0, 0, 0), -1)
        cv2.rectangle(frame, (text_x - 20, text_y - 50), 
                     (text_x + text_size[0] + 20, text_y + 20), (0, 0, 255), 3)
        
        cv2.putText(frame, pause_text, (text_x, text_y), font, 2, (0, 0, 255), 3)
        cv2.putText(frame, "Press 'p' to resume", (text_x - 50, text_y + 50), 
                   font, 0.7, (255, 255, 255), 2)
    
    # ============================================
    # CONTROLS HINT (dưới cùng)
    # ============================================
    hint_y = h - 15
    cv2.putText(frame, "Controls: Q-Quit | P-Pause | S-Screenshot | R-Reset", 
               (15, hint_y), font, 0.5, (150, 150, 150), 1)
    
    return frame


def main():
    """Hàm main"""
    
    print("\n" + "=" * 70)
    print("TRAFFIC LIGHT AI SYSTEM - YOLOv5 + QCVN 41:2019")
    print("=" * 70)
    
    # Load model
    print("Loading model...", end=" ", flush=True)
    try:
        model = torch.hub.load(YOLO_DIR, 'custom', path=MODEL_PATH, source='local')
        model.conf = 0.5
        print("OK")
    except Exception as e:
        print(f"\nERROR: {e}")
        return
    
    # Open video
    print("Opening video...", end=" ", flush=True)
    traffic_brain = TrafficLightSystem()
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("ERROR: Cannot open video!")
        return
    
    # Lấy thông tin video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"OK ({total_frames} frames, {video_fps:.0f}fps)")
    
    print("\nControls: Q-Quit | P-Pause | S-Screenshot | R-Reset")
    print("=" * 70 + "\n")
    
    # Biến theo dõi
    frame_count = 0
    cached_result = None
    update_interval = 5
    
    paused = False
    prev_time = cv2.getTickCount()
    fps = 0
    
    # Thống kê
    total_vehicles = 0
    max_vehicles = 0
    
    # Biến để kiểm soát print (chỉ print mỗi 30 frames = 1 giây)
    print_interval = 30
    
    # === BIẾN CHO CHU KỲ ĐÈN THỰC TẾ ===
    cycle_start_time = cv2.getTickCount()  # Thời điểm bắt đầu chu kỳ
    current_phase = 'green'  # Trạng thái hiện tại: 'green', 'yellow', 'red'
    phase_durations = {'green': 30, 'yellow': 3, 'red': 20}  # Thời gian mặc định
    last_vehicle_count = 0  # Số xe lần cập nhật trước
    update_threshold = 5  # Chỉ cập nhật khi thay đổi >= 5 xe
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.resize(frame, (1020, 600))
                results = model(frame)
                df = results.pandas().xyxy[0]
                vehicle_count = len(df)
                
                total_vehicles += vehicle_count
                if vehicle_count > max_vehicles:
                    max_vehicles = vehicle_count
                
                rendered_frame = results.render()[0]
                frame_clean = np.array(rendered_frame)
                
                # Cập nhật thời gian đèn (CHỈ KHI THAY ĐỔI ĐÁNG KỂ)
                if frame_count % update_interval == 0:
                    # Kiểm tra thay đổi số xe
                    vehicle_change = abs(vehicle_count - last_vehicle_count)
                    
                    # CHỈ tính toán lại và reset chu kỳ khi thay đổi >= threshold
                    if vehicle_change >= update_threshold or cached_result is None:
                        cached_result = traffic_brain.calculate(num_vehicles=vehicle_count)
                        
                        # Cập nhật thời gian chu kỳ từ kết quả
                        if cached_result:
                            phase_durations = {
                                'green': cached_result['output']['green'],
                                'yellow': cached_result['output']['yellow'],
                                'red': cached_result['output']['red']
                            }
                            # RESET chu kỳ CHỈ khi thay đổi lớn
                            cycle_start_time = cv2.getTickCount()
                            current_phase = 'green'
                            last_vehicle_count = vehicle_count  # Lưu lại số xe
                    
                    # Chỉ print mỗi print_interval frames
                    if frame_count % print_interval == 0 and cached_result:
                        status_icon = "[OK]" if cached_result['status']['level'] == 'normal' else \
                                     "[!!]" if cached_result['status']['level'] == 'warning' else "[XX]"
                        print(f"{status_icon} Frame {frame_count:4d} | Vehicles:{vehicle_count:3d} | "
                              f"G:{cached_result['output']['green']:4.1f}s "
                              f"R:{cached_result['output']['red']:4.1f}s | "
                              f"Phase: {current_phase.upper()}", end="\r", flush=True)
                
                # === TÍNH TOÁN CHU KỲ ĐÈN THỰC TẾ ===
                current_time = cv2.getTickCount()
                elapsed_time = (current_time - cycle_start_time) / cv2.getTickFrequency()
                
                # Tổng thời gian chu kỳ
                total_cycle = phase_durations['green'] + phase_durations['yellow'] + phase_durations['red']
                
                # Tính phase hiện tại dựa trên thời gian đã trôi qua
                cycle_time = elapsed_time % total_cycle  # Thời gian trong chu kỳ hiện tại
                
                if cycle_time < phase_durations['green']:
                    current_phase = 'green'
                elif cycle_time < phase_durations['green'] + phase_durations['yellow']:
                    current_phase = 'yellow'
                else:
                    current_phase = 'red'
                
                frame_count += 1
            else:
                frame_clean = frame_clean.copy()
            
            # Tính FPS
            curr_time = cv2.getTickCount()
            time_diff = (curr_time - prev_time) / cv2.getTickFrequency()
            if time_diff > 0:
                fps = 1.0 / time_diff
            prev_time = curr_time
            
            # Vẽ thông tin với đèn đúng phase
            final_frame = draw_enhanced_info(
                frame_clean, cached_result, vehicle_count if not paused else 0,
                fps, frame_count, paused, current_phase  # Thêm current_phase
            )
            
            cv2.imshow("He thong den giao thong AI - YOLOv5", final_frame)
            
            # Xử lý phím
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n⏹️  Đang dừng...")
                break
            elif key == ord('p'):
                paused = not paused
                print(f"\n{'⏸️  PAUSED' if paused else '▶️  RESUMED':<50}")
            elif key == ord('s'):
                screenshot_name = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(screenshot_name, final_frame)
                print(f"\n📸 Screenshot: {screenshot_name:<50}")
            elif key == ord('r'):
                total_vehicles = 0
                max_vehicles = 0
                cycle_start_time = cv2.getTickCount()  # Reset chu kỳ
                print("\n🔄 Reset stats" + " " * 50)
    
    except KeyboardInterrupt:
        print("\n⚠️  Stopped by user")
    
    finally:
        # Thống kê cuối (gọn hơn)
        print("\n\n" + "=" * 70)
        print("📊 SUMMARY")
        print("=" * 70)
        print(f"Frames:        {frame_count}")
        print(f"Total vehicles: {total_vehicles}")
        print(f"Max vehicles:   {max_vehicles}")
        if frame_count > 0:
            print(f"Avg vehicles:   {total_vehicles/frame_count:.1f}")
        print("=" * 70)
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Done\n")


if __name__ == "__main__":
    main()
