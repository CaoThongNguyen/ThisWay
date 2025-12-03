"""
traffic_light_system.py

HỆ THỐNG ĐÈN GIAO THÔNG HỢP LÝ
Logic: Xe nhiều → Xanh cao | Xe ít → Xanh thấp
Đèn đỏ điều chỉnh hợp lý để cân bằng chu kỳ
"""

from datetime import datetime
from typing import Dict, Optional
import json

class TrafficLightSystem:
    """Hệ thống đèn giao thông hợp lý"""
    
    def __init__(self):
        """Khởi tạo hệ thống"""
        self.config = {
            # Giới hạn thời gian đèn
            "g_min": 15.0,              # Xanh tối thiểu (giây)
            "g_max": 90.0,              # Xanh tối đa (giây)
            "r_min": 20.0,              # Đỏ tối thiểu (giây) - QUAN TRỌNG
            "r_max": 60.0,              # Đỏ tối đa (giây)
            "y_time": 3.0,              # Vàng cố định (giây)
            
            # Hệ số thời gian theo giờ (giây/xe)
            "time_per_vehicle": {
                "low": 1.8,             # 22h-6h
                "normal": 2.2,          # 6h-7h, 9h-16h, 19h-22h
                "rush": 2.8             # 7h-9h, 16h-19h
            },
            
            "safety_margin": 1.15,      # Hệ số an toàn +15%
            
            # Tỷ lệ đèn đỏ so với đèn xanh (hợp lý)
            "red_ratio": 0.7            # Đỏ = 70% Xanh (có thể điều chỉnh)
        }
    
    def get_time_mode(self, hour: Optional[int] = None) -> str:
        """Xác định chế độ giờ"""
        if hour is None:
            hour = datetime.now().hour
        
        if (7 <= hour < 9) or (16 <= hour < 19):
            return "rush"
        elif (22 <= hour) or (hour < 6):
            return "low"
        else:
            return "normal"
    
    def get_time_mode_name(self, mode: str) -> str:
        """Tên tiếng Việt"""
        names = {
            "low": "Gio thap diem",
            "normal": "Gio binh thuong",
            "rush": "Gio cao diem"
        }
        return names.get(mode, "Khong xac dinh")
    
    def calculate(self, num_vehicles: int, hour: Optional[int] = None) -> Dict:
        """
        Tính thời gian đèn HỢP LÝ
        
        LOGIC:
        1. Đèn XANH = Số_xe × Thời_gian_1_xe (tăng theo số xe)
        2. Đèn VÀNG = 3 giây (cố định)
        3. Đèn ĐỎ = Tỷ lệ hợp lý với đèn xanh (70% xanh)
           - VD: Xanh 30s → Đỏ 21s
           - VD: Xanh 60s → Đỏ 42s
        """
        now = datetime.now()
        current_hour = hour if hour is not None else now.hour
        time_mode = self.get_time_mode(current_hour)
        
        if num_vehicles < 0:
            num_vehicles = 0
        
        # Lấy hệ số thời gian theo giờ
        t_veh = self.config["time_per_vehicle"][time_mode]
        
        # === ĐÈN XANH (tăng theo số xe) ===
        green = num_vehicles * t_veh * self.config["safety_margin"]
        green = max(self.config["g_min"], min(green, self.config["g_max"]))
        
        # === ĐÈN VÀNG (cố định) ===
        yellow = self.config["y_time"]
        
        # === ĐÈN ĐỎ (tỷ lệ hợp lý với xanh) ===
        # Đỏ = 70% Xanh (có thể thay đổi tỷ lệ này)
        red = green * self.config["red_ratio"]
        
        # Giới hạn đỏ hợp lý (QUAN TRỌNG)
        # Đỏ tối thiểu: 20s (không bao giờ thấp hơn)
        # Đỏ tối đa: 60s (tránh chờ quá lâu)
        red = max(self.config["r_min"], min(red, self.config["r_max"]))
        
        # === CHU KỲ ===
        cycle = green + yellow + red
        
        # Đánh giá trạng thái
        status = self._evaluate_status(num_vehicles, green, red, cycle, time_mode)
        
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
    
    def _evaluate_status(self, num_vehicles: int, green: float, red: float,
                        cycle: float, time_mode: str) -> Dict:
        """Đánh giá trạng thái"""
        recommendations = []
        
        if green >= self.config["g_max"]:
            level = "critical"
            message = "QUA TAI"
            recommendations.append("Luong xe qua dong")
        
        elif green >= self.config["g_max"] * 0.8:
            level = "warning"
            message = "DONG XE"
            recommendations.append("Gan dat nguong qua tai")
        
        elif num_vehicles < 5:
            level = "normal"
            message = "THONG THOANG"
        
        else:
            level = "normal"
            message = "BINH THUONG"
        
        return {
            "level": level,
            "message": message,
            "recommendations": recommendations
        }
    
    def print_result(self, result: Dict):
        """In kết quả"""
        print("=" * 60)
        print("DEN GIAO THONG HOP LY")
        print("=" * 60)
        
        inp = result["input"]
        print(f"\nSo xe:      {inp['num_vehicles']} xe")
        print(f"Gio:        {inp['hour']}h ({inp['time_mode_name']})")
        
        out = result["output"]
        print(f"\n=== THOI GIAN DEN ===")
        print(f"Xanh:   {out['green']:>5.1f}s")
        print(f"Vang:   {out['yellow']:>5.1f}s")
        print(f"Do:     {out['red']:>5.1f}s  (= {out['red']/out['green']*100:.0f}% Xanh)")
        print(f"Chu ky: {out['cycle']:>5.1f}s")
        
        status = result["status"]
        print(f"\nTrang thai: {status['message']}")
        
        if status['recommendations']:
            for rec in status['recommendations']:
                print(f"  - {rec}")
        
        print("=" * 60)


def main():
    """Demo"""
    system = TrafficLightSystem()
    
    print("\n=== TEST LOGIC HOP LY ===")
    print("Xanh: Min 15s, Max 90s")
    print("Do:   Min 20s, Max 60s")
    print("Do = 70% Xanh (nhung khong thap hon 20s)\n")
    
    test_cases = [
        (3, "Rat it xe"),
        (5, "It xe"),
        (8, "Trung binh thap"),
        (12, "Trung binh"),
        (18, "Kha dong"),
        (25, "Dong xe"),
        (32, "Rat dong"),
        (40, "Qua tai"),
        (50, "Qua tai nang")
    ]
    
    for num_vehicles, desc in test_cases:
        result = system.calculate(num_vehicles)
        out = result['output']
        ratio = out['red'] / out['green'] * 100 if out['green'] > 0 else 0
        print(f"{desc:18s} ({num_vehicles:2d} xe) → "
              f"Xanh: {out['green']:5.1f}s | "
              f"Do: {out['red']:5.1f}s ({ratio:.0f}%) | "
              f"Chu ky: {out['cycle']:5.1f}s")
    
    print("\n" + "="*60)
    print("CHI TIET (5 xe - xe it):")
    result = system.calculate(5)
    system.print_result(result)
    
    print("\n" + "="*60)
    print("CHI TIET (25 xe - dong xe):")
    result = system.calculate(25)
    system.print_result(result)


if __name__ == "__main__":
    main()
