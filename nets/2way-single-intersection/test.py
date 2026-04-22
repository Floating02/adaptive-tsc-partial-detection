#!/usr/bin/env python3
# ------------------------------------------------------------
#  check_expected_vehicles.py
#  使用方式:
#     python check_expected_vehicles.py flows_9000s.rou.xml
# ------------------------------------------------------------
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

def main(file_path: str):
    tree = ET.parse(file_path)
    root = tree.getroot()

    total_rate      = 0.0          # 所有 flow 的 λ 之和 (veh/s)
    total_expected  = 0.0          # 期望车辆总数
    by_direction    = defaultdict(float)  # 入口方向 -> 期望车辆数
    by_route        = defaultdict(float)  # route_id  -> 期望车辆数

    for flow in root.findall("flow"):
        p       = float(flow.get("probability", 0))
        begin   = int(flow.get("begin", 0))
        end     = int(flow.get("end", 0))
        dur     = end - begin
        exp_num = p * dur

        total_rate     += p
        total_expected += exp_num

        # ----- 根据 route 前缀粗分方向 (N/E/S/W) -----
        route_id = flow.get("route")
        direction = route_id.split("_")[1][0].upper()  # e.g. route_ns -> n
        by_direction[direction] += exp_num
        by_route[route_id]      += exp_num

    print(f"\n=== file: {Path(file_path).name} ===")
    print(f"λ_total   = {total_rate:.5f} veh/s")
    print(f"Expected total vehicles = {total_expected:.0f}\n")

    print("---- by direction ----")
    for d, num in sorted(by_direction.items()):
        print(f"{d}: {num:.0f} veh")

    print("\n---- by route ----")
    for r, num in sorted(by_route.items()):
        print(f"{r:<8} {num:>7.0f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python check_expected_vehicles.py <flow_file.rou.xml>")
        sys.exit(1)
    main(sys.argv[1])
