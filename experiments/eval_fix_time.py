import os, sys
import xml.etree.ElementTree as ET

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("请设置环境变量 SUMO_HOME")

import traci

def parse_tripinfo(path="tripinfo.xml"):
    tree = ET.parse(path)
    trips = tree.findall("tripinfo")
    if not trips:
        return 0, 0, 0
    durations    = [float(t.get("duration"))    for t in trips]
    waitings     = [float(t.get("waitingTime")) for t in trips]
    arrived      = len(trips)
    return (sum(durations) / arrived,
            sum(waitings)  / arrived,
            arrived)

def run_fixed_timing(net_file, route_file, sim_steps=3600):
    sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
    sumo_cmd = [
        sumo_binary,
        "--net-file",       net_file,
        "--route-files",    route_file,
        "--tripinfo-output","tripinfo.xml",
        "--step-length",    "1",
        "--no-step-log",
        "--begin",          "0",
        "--end",            str(sim_steps),
    ]

    traci.start(sumo_cmd)
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
    traci.close()

    att, awt, arrived = parse_tripinfo("tripinfo.xml")
    generated = arrived  # 可从 summary.xml 读取，或直接用 arrived

    print("=== 固定配时仿真结果 ===")
    print(f"完成行程车辆数 : {arrived} 辆")
    print(f"平均出行时间   : {att:.2f} 秒")   # 核心指标
    print(f"平均等待时间   : {awt:.2f} 秒")
    return att, awt, arrived

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_fixed_timing("../nets/2way-single-intersection/single-intersection.net.xml",
                     "../nets/2way-single-intersection/single-intersection_medium.rou.xml")
