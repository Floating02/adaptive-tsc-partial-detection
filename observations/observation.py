"""用于部分检测车辆的交通信号控制状态表示

本模块实现了一个用于部分检测车辆的观察类，用于交通信号控制系统。
状态表示包括：检测车辆数量、最近检测车辆距离、当前相位时间、黄灯指示器、当前时间等信息。
"""

import os
import sys
import numpy as np
from gymnasium import spaces

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("请声明环境变量'SUMO_HOME'")

from sumo_rl.environment.observations import ObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal


class TableIObservationFunction(ObservationFunction):
    """用于部分检测车辆的观察类
    
    状态表示包括（所有特征均已归一化）：
    - 检测车辆数量（带符号）：每个路径上检测到的车辆数量，除以车道最大容量；
      绿灯车道为正值，红灯车道为负值，范围 -1 到 1
    - 最近车辆距离（带符号）：每个路径上最近检测车辆的距离，除以车道长度；
      绿灯车道为正值，红灯车道为负值，范围 -1 到 1
    - 当前相位时间：从当前相位开始到现在的持续时间（秒），除以最大相位时长，范围 0-1
    - 黄灯指示器：黄灯相位指示器（0或1）
    - 当前时间：一天中的当前时间（午夜后的小时数），归一化到0-1（除以24）
    
    属性:
        ts (TrafficSignal): 交通信号对象
        detection_rate (float): 检测率，即车辆被检测到的概率，取值范围[0,1]
        detected_vehicles (dict): 记录已被检测到的车辆ID
        max_car_capacity (int): 每个车道的最大车辆容量估算值
        max_phase_duration (int): 最大相位持续时间（秒）
    """
    
    def __init__(self, ts: TrafficSignal, detection_rate: float = 0.7, max_car_capacity: int = 10, max_phase_duration: int = 120, seed: int = None):
        """初始化部分检测车辆的观察函数
        
        Args:
            ts (TrafficSignal): 交通信号对象
            detection_rate (float, optional): 检测率，默认为0.7
            max_car_capacity (int, optional): 每个车道的最大车辆容量估算值，默认为10
            max_phase_duration (int, optional): 最大相位持续时间（秒），默认为120
            seed (int, optional): 随机种子，用于可复现的检测行为
        """
        super().__init__(ts)
        
        # 检测率必须在[0,1]范围内
        assert 0 <= detection_rate <= 1, "检测率必须在0到1之间"
        self.detection_rate = detection_rate
        
        # 存储已检测到的车辆ID
        self.detected_vehicles = {}
        
        # 归一化参数
        self.max_car_capacity = max_car_capacity
        self.max_phase_duration = max_phase_duration
        
        # 初始化独立的随机数生成器，确保可复现性
        self.rng = np.random.RandomState(seed)
    
    def __call__(self) -> np.ndarray:
        """返回基于部分检测车辆的状态表示观察
        
        Returns:
            np.ndarray: 观察向量
         """
        self._update_detected_vehicles()
        
        current_state = self.ts.sumo.trafficlight.getRedYellowGreenState(self.ts.id)
        controlled_links = self.ts.sumo.trafficlight.getControlledLinks(self.ts.id)
        
        signed_car_counts = []
        signed_distances = []
        
        for lane in self.ts.lanes:
            veh_list = self.ts.sumo.lane.getLastStepVehicleIDs(lane)
            detected_veh_list = [veh for veh in veh_list if veh in self.detected_vehicles and self.detected_vehicles[veh]]
            
            detected_count = len(detected_veh_list)
            normalized_count = min(detected_count / self.max_car_capacity, 1.0)
            
            lane_length = self.ts.lanes_length[lane]
            if detected_veh_list:
                positions = [lane_length - self.ts.sumo.vehicle.getLanePosition(veh) for veh in detected_veh_list]
                min_distance = min(positions)
            else:
                min_distance = lane_length
            normalized_distance = min_distance / lane_length
            
            is_green = self._is_lane_green(lane, current_state, controlled_links)
            sign = 1.0 if is_green else -1.0
            
            signed_car_counts.append(sign * normalized_count)
            signed_distances.append(sign * normalized_distance)
        
        normalized_phase_time = [min(float(self.ts.time_since_last_phase_change) / self.ts.max_green_time, 1.0)]
        
        amber_phase = [1.0 if self.ts.is_yellow else 0.0]
        
        current_time = [float(self.ts.env.sim_step % (24 * 3600)) / (24 * 3600)]
        
        observation = np.array(
            signed_car_counts + 
            signed_distances + 
            normalized_phase_time + 
            amber_phase + 
            current_time, 
            dtype=np.float32
        )
        return observation
    
    def reset(self):
        """重置检测车辆字典（预留接口）
        
        注意：当前 SumoEnvironment 实现会在每个 episode 开始时重新创建
        TrafficSignal 和 ObservationFunction 实例，因此该方法不会被调用。
        该方法保留为未来优化预留接口。
        """
        self.detected_vehicles.clear()
    
    def _is_lane_green(self, lane: str, current_state: str, controlled_links: list) -> bool:
        """判断车道在当前信号状态下是否为绿灯
        
        Args:
            lane: 车道ID
            current_state: 当前信号灯状态字符串（如"GGgrrrGGgrrr"）
            controlled_links: 受控连接列表，由getControlledLinks返回
            
        Returns:
            bool: 如果车道任意连接方向为绿灯则返回True，否则返回False
        """
        for link_idx, link_group in enumerate(controlled_links):
            for link_tuple in link_group:
                if link_tuple[0] == lane:
                    if current_state[link_idx] in ('G', 'g'):
                        return True
                    break
        return False
    
    def _update_detected_vehicles(self):
        """更新被检测到的车辆列表
        
        为新出现的车辆决定是否能被检测到，并从列表中移除已离开的车辆。
        使用集合操作优化清理逻辑，确保字典只包含当前存在的车辆。
        """
        # 获取当前所有车辆及其所在车道
        current_vehicles = set()
        for lane in self.ts.lanes:
            veh_list = self.ts.sumo.lane.getLastStepVehicleIDs(lane)
            current_vehicles.update(veh_list)
        
        # 一次性移除所有不在当前系统中的车辆（集合差集操作）
        vehicles_to_remove = set(self.detected_vehicles.keys()) - current_vehicles
        for veh in vehicles_to_remove:
            del self.detected_vehicles[veh]
        
        # 检查新出现的车辆并决定它们是否可被检测
        new_vehicles = current_vehicles - set(self.detected_vehicles.keys())
        for veh in new_vehicles:
            # 对新车辆进行伯努利试验，决定是否可被检测
            self.detected_vehicles[veh] = self.rng.random() < self.detection_rate
    
    def observation_space(self) -> spaces.Box:
        """定义观察空间
        
        观察空间的维度由以下部分组成：
        - 各车道检测车辆数量（带符号，绿灯为正、红灯为负，-1到1）
        - 各车道最近检测车辆距离（带符号，绿灯为正、红灯为负，-1到1）
        - 当前相位时间 (0-1)
        - 黄灯相位指示器 (0或1)
        - 当前时间 (0-1)
        
        Returns:
            spaces.Box: 观察空间
        """
        n_lanes = len(self.ts.lanes)
        
        return spaces.Box(
            low=np.array(
                [-1] * n_lanes +       
                [-1] * n_lanes +       
                [0] +                  
                [0] +                  
                [0],                   
                dtype=np.float32
            ),
            high=np.array(
                [1] * n_lanes +        
                [1] * n_lanes +        
                [1] +                  
                [1] +                  
                [1],                   
                dtype=np.float32
            ),
        )

