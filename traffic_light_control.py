def calculate_total_vehicles(data):
    return sum(data.values())


def adjust_traffic_lights(data, main_roads=None, cycle_time=100, bonus_time=5):

    total = calculate_total_vehicles(data)

    if total == 0:
        print("Giữ tất cả đèn đỏ hoặc chế độ chờ.")
        return {k: 0 for k in data}

    # Tính thời gian cơ bản theo tỉ lệ xe
    light_times = {}
    for direction, count in data.items():
        light_times[direction] = (count / total) * cycle_time

    # Cộng thêm thời gian ưu tiên cho đường chính
    if main_roads:
        for road in main_roads:
            if road in light_times:
                light_times[road] += bonus_time

    # Chuẩn hóa lại tổng thời gian để không vượt quá cycle_time
    total_time = sum(light_times.values())
    if total_time > cycle_time:
        scale = cycle_time / total_time
        for direction in light_times:
            light_times[direction] = int(light_times[direction] * scale)
    else:
        for direction in light_times:
            light_times[direction] = int(light_times[direction])

    return light_times


def main():
    # Dữ liệu quét từ 4 hướng
    traffic_data = {
        "north": 35,
        "south": 30,
        "east": 40,
        "west": 25
    }

    # Đường chính (ưu tiên)
    main_roads = ["east", "west"]

    print(f"=== DỮ LIỆU QUÉT (NGÃ TƯ, {len(traffic_data)} HƯỚNG) ===")
    for direction, count in traffic_data.items():
        print(f"{direction.capitalize()}: {count} xe")

    total = calculate_total_vehicles(traffic_data)
    print(f"\nTổng số phương tiện: {total}")

    print("\n=== ĐIỀU CHỈNH ĐÈN GIAO THÔNG (CÓ ƯU TIÊN) ===")
    light_times = adjust_traffic_lights(traffic_data, main_roads=main_roads)
    for direction, time in light_times.items():
        label = " (ưu tiên)" if direction in main_roads else ""
        print(f"Đèn {direction.capitalize()}{label}: xanh {time} giây")

    print("\n=> Hướng có nhiều xe hoặc là đường chính được ưu tiên lâu hơn.")

if __name__ == "__main__":
    main()
