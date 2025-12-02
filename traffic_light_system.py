"""
traffic_light_system.py

HỆ THỐNG ĐÈN GIAO THÔNG THÔNG MINH - THÀNH PHẨM
Tự động lấy giờ từ hệ thống, tính toán và trả về thời gian đèn

Cách dùng:
    from traffic_light_system import TrafficLightSystem
    
    system = TrafficLightSystem()
    result = system.calculate(num_vehicles=25)
    print(result)
"""

from datetime import datetime
from typing import Dict, Optional
import json

class TrafficLightSystem:
    """
    Hệ thống đèn giao thông hoàn chỉnh
    Tuân thủ QCVN 41:2019/BGTVT
    """
    
    def __init__(self):
        """Khởi tạo hệ thống với cấu hình QCVN Việt Nam"""
        self.config = {
            # Theo QCVN 41:2019/BGTVT
            "g_min": 15.0,              # Xanh tối thiểu (giây)
            "g_max": 90.0,              # Xanh tối đa (giây)
            "y_time": 3.0,              # Vàng (giây)
            "cycle_min": 30.0,          # Chu kỳ tối thiểu
            "cycle_max": 120.0,         # Chu kỳ tối đa
            
            # Hệ số thời gian theo giờ (giây/xe)
            "time_per_vehicle": {
                "low": 1.8,             # 22h-6h
                "normal": 2.2,          # 6h-7h, 9h-16h, 19h-22h
                "rush": 2.8             # 7h-9h, 16h-19h
            },
            
            "safety_margin": 1.15       # Hệ số an toàn +15%
        }
    
    def get_time_mode(self, hour: Optional[int] = None) -> str:
        """
        Xác định chế độ giờ (tự động lấy từ hệ thống)
        
        Args:
            hour: Giờ trong ngày (0-23). Nếu None, lấy giờ hiện tại
        
        Returns:
            "low" | "normal" | "rush"
        """
        if hour is None:
            hour = datetime.now().hour
        
        # Giờ cao điểm
        if (7 <= hour < 9) or (16 <= hour < 19):
            return "rush"
        
        # Giờ thấp điểm
        elif (22 <= hour) or (hour < 6):
            return "low"
        
        # Giờ bình thường
        else:
            return "normal"
    
    def get_time_mode_name(self, mode: str) -> str:
        """Chuyển mode thành tên tiếng Việt"""
        names = {
            "low": "Giờ thấp điểm",
            "normal": "Giờ bình thường",
            "rush": "Giờ cao điểm"
        }
        return names.get(mode, "Không xác định")
    
    def calculate(self, num_vehicles: int, hour: Optional[int] = None) -> Dict:
        """
        CHỨC NĂNG CHÍNH: Tính thời gian đèn
        
        Args:
            num_vehicles: Số xe đếm được (từ YOLOv5)
            hour: Giờ (tùy chọn). Nếu không truyền, tự động lấy giờ hiện tại
        
        Returns:
            {
                'input': {...},         # Thông tin đầu vào
                'output': {...},        # Kết quả thời gian đèn
                'status': {...},        # Trạng thái hệ thống
                'timestamp': '...'      # Thời gian tính toán
            }
        """
        # Lấy thời gian hiện tại
        now = datetime.now()
        current_hour = hour if hour is not None else now.hour
        time_mode = self.get_time_mode(current_hour)
        
        # Kiểm tra đầu vào
        if num_vehicles < 0:
            num_vehicles = 0
        
        # Lấy hệ số thời gian
        t_veh = self.config["time_per_vehicle"][time_mode]
        
        # Tính thời gian xanh
        green = num_vehicles * t_veh * self.config["safety_margin"]
        green = max(self.config["g_min"], min(green, self.config["g_max"]))
        
        # Thời gian vàng (cố định)
        yellow = self.config["y_time"]
        
        # Tính chu kỳ
        cycle = 2 * (green + yellow)
        
        # Điều chỉnh chu kỳ nếu vượt giới hạn
        if cycle < self.config["cycle_min"]:
            green = (self.config["cycle_min"] / 2) - yellow
            cycle = self.config["cycle_min"]
        elif cycle > self.config["cycle_max"]:
            green = (self.config["cycle_max"] / 2) - yellow
            cycle = self.config["cycle_max"]
        
        # Tính thời gian đỏ
        red = cycle - green - yellow
        
        # Đánh giá trạng thái
        status = self._evaluate_status(num_vehicles, green, cycle, time_mode)
        
        # Kết quả
        result = {
            "input": {
                "num_vehicles": num_vehicles,
                "hour": current_hour,
                "time_mode": time_mode,
                "time_mode_name": self.get_time_mode_name(time_mode)
            },
            "output": {
                "green": round(green, 1),
                "yellow": round(yellow, 1),
                "red": round(red, 1),
                "cycle": round(cycle, 1)
            },
            "status": status,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return result
    
    def _evaluate_status(self, num_vehicles: int, green: float, 
                        cycle: float, time_mode: str) -> Dict:
        """
        Đánh giá trạng thái giao lộ
        
        Returns:
            {
                'level': 'normal' | 'warning' | 'critical',
                'message': '...',
                'recommendations': [...]
            }
        """
        recommendations = []
        
        # Kiểm tra quá tải
        if green >= self.config["g_max"]:
            level = "critical"
            message = "🚨 GIAO LỘ QUÁ TẢI"
            recommendations.append("Cần mở rộng giao lộ hoặc phân luồng ngay")
            recommendations.append("Cân nhắc điều động CSGT hỗ trợ")
        
        elif green >= self.config["g_max"] * 0.8:
            level = "warning"
            message = "⚠️ Giao lộ gần quá tải"
            recommendations.append("Theo dõi sát tình hình")
            recommendations.append("Chuẩn bị phương án dự phòng")
        
        elif cycle >= self.config["cycle_max"]:
            level = "warning"
            message = "⚠️ Chu kỳ đã đạt tối đa"
            recommendations.append("Thời gian chờ có thể gây bức xúc")
        
        else:
            level = "normal"
            message = "✅ Hoạt động bình thường"
            
            # Gợi ý cho giờ thấp điểm
            if time_mode == "low" and num_vehicles < 5:
                recommendations.append("💡 Có thể chuyển sang đèn vàng nhấp nháy")
        
        # Thông tin hệ số
        if time_mode == "rush":
            recommendations.append(f"🚗 Đã áp dụng hệ số giờ cao điểm (2.8s/xe)")
        elif time_mode == "low":
            recommendations.append(f"🌙 Đã áp dụng hệ số giờ thấp điểm (1.8s/xe)")
        
        return {
            "level": level,
            "message": message,
            "recommendations": recommendations
        }
    
    def print_result(self, result: Dict):
        """In kết quả ra màn hình (định dạng đẹp)"""
        print("=" * 70)
        print("🚦 HỆ THỐNG ĐÈN GIAO THÔNG THÔNG MINH")
        print("=" * 70)
        
        # Thông tin đầu vào
        inp = result["input"]
        print(f"\n📥 THÔNG TIN ĐẦU VÀO")
        print(f"   • Thời gian:     {result['timestamp']}")
        print(f"   • Số xe:         {inp['num_vehicles']} xe")
        print(f"   • Giờ:           {inp['hour']}h ({inp['time_mode_name']})")
        
        # Kết quả
        out = result["output"]
        print(f"\n📤 KẾT QUẢ THỜI GIAN ĐÈN")
        print(f"   🟢 Đèn XANH:     {out['green']:>6} giây")
        print(f"   🟡 Đèn VÀNG:     {out['yellow']:>6} giây")
        print(f"   🔴 Đèn ĐỎ:       {out['red']:>6} giây")
        print(f"   ⭕ Chu kỳ:       {out['cycle']:>6} giây")
        
        # Trạng thái
        status = result["status"]
        print(f"\n📊 TRẠNG THÁI")
        print(f"   {status['message']}")
        
        if status['recommendations']:
            print(f"\n💡 KHUYẾN NGHỊ")
            for rec in status['recommendations']:
                print(f"   • {rec}")
        
        print("=" * 70)
    
    def to_json(self, result: Dict) -> str:
        """Chuyển kết quả thành JSON string"""
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    def save_to_file(self, result: Dict, filename: str = "traffic_result.json"):
        """Lưu kết quả vào file JSON"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ Đã lưu kết quả vào {filename}")


# ============================================
# CÁCH SỬ DỤNG
# ============================================

def main():
    """Hàm main - ví dụ sử dụng"""
    
    # 1. Khởi tạo hệ thống (chỉ cần 1 lần)
    system = TrafficLightSystem()
    
    print("\n" + "=" * 70)
    print("DEMO: HỆ THỐNG ĐÈN GIAO THÔNG TỰ ĐỘNG")
    print("=" * 70)
    
    # 2. Tình huống 1: Lấy số xe từ YOLOv5 (giả lập)
    print("\n[Tình huống 1] Số xe từ camera YOLOv5")
    num_vehicles_from_yolo = 25  # Giả sử YOLOv5 đếm được 25 xe
    
    result = system.calculate(num_vehicles_from_yolo)
    system.print_result(result)
    
    # 3. Tình huống 2: Test với giờ cụ thể
    print("\n[Tình huống 2] Test giờ cao điểm (7h sáng)")
    result = system.calculate(num_vehicles=30, hour=7)
    system.print_result(result)
    
    # 4. Tình huống 3: Test giờ khuya
    print("\n[Tình huống 3] Test giờ khuya (2h sáng)")
    result = system.calculate(num_vehicles=5, hour=2)
    system.print_result(result)
    
    # 5. Xuất JSON
    print("\n" + "=" * 70)
    print("JSON OUTPUT")
    print("=" * 70)
    print(system.to_json(result))
    
    # 6. Lưu file
    # system.save_to_file(result, "traffic_result.json")


# ============================================
# CHẠY TRỰC TIẾP
# ============================================

if __name__ == "__main__":
    """
    CÁC CÁCH SỬ DỤNG:
    
    1. Sử dụng cơ bản (tự động lấy giờ hiện tại):
       ----------------------------------------
       system = TrafficLightSystem()
       result = system.calculate(num_vehicles=25)
       system.print_result(result)
    
    2. Lấy chỉ kết quả thời gian đèn:
       --------------------------------
       result = system.calculate(25)
       green = result['output']['green']
       yellow = result['output']['yellow']
       red = result['output']['red']
    
    3. Tích hợp với YOLOv5:
       ---------------------
       # Giả sử YOLOv5 trả về số xe
       num_cars = yolo_model.count_vehicles(image)
       
       # Tính thời gian đèn
       system = TrafficLightSystem()
       result = system.calculate(num_cars)
       
       # Điều khiển đèn
       set_green_light(result['output']['green'])
       set_yellow_light(result['output']['yellow'])
       set_red_light(result['output']['red'])
    
    4. API endpoint (Flask/FastAPI):
       ------------------------------
       @app.post("/calculate")
       def calculate_traffic(num_vehicles: int):
           system = TrafficLightSystem()
           result = system.calculate(num_vehicles)
           return result
    """
    
    # Chạy demo
    main()
    
    print("\n" + "=" * 70)
    print("✅ HỆ THỐNG SẴN SÀNG SỬ DỤNG!")
    print("=" * 70)
    print("""
    📌 HƯỚNG DẪN NHANH:
    
    from traffic_light_system import TrafficLightSystem
    
    system = TrafficLightSystem()
    result = system.calculate(num_vehicles=25)
    
    # Lấy kết quả
    green = result['output']['green']
    yellow = result['output']['yellow']
    red = result['output']['red']
    """)