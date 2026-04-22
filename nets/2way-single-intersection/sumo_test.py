#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import traci

# 确保SUMO_HOME环境变量已设置
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("请设置环境变量 SUMO_HOME")

def run():
    # 连接到SUMO
    sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
    sumo_cmd = [sumo_binary, "-c", "sumo_rl/nets/2way-single-intersection/single-intersection-poisson.sumocfg"]
    
    # 启动SUMO并建立连接
    traci.start(sumo_cmd)
    
    # 获取仿真中的车辆ID列表
    step = 0
    while step < 1000:  # 运行1000个仿真步
        traci.simulationStep()  # 执行一步仿真
        
        vehicle_ids = traci.vehicle.getIDList()
        if vehicle_ids:
            print(f"步骤 {step}，车辆数量: {len(vehicle_ids)}")
            
            # 获取第一辆车的信息示例
            veh_id = vehicle_ids[0]
            x, y = traci.vehicle.getPosition(veh_id)
            speed = traci.vehicle.getSpeed(veh_id)
            
            print(f"  车辆 {veh_id}: 位置=({x:.2f}, {y:.2f}), 速度={speed:.2f}m/s")
            
            # 控制车辆示例 - 更改某辆车的速度
            traci.vehicle.setSpeed(veh_id, speed + 1)  # 增加1m/s的速度
        
        step += 1
    
    # 关闭连接
    traci.close()

if __name__ == "__main__":
    run()
