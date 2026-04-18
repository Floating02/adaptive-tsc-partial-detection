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
    - 检测车辆数量：每个路径上检测到的车辆数量，除以车道最大容量，范围 0-1
    - 最近车辆距离：每个路径上最近检测车辆的距离，除以车道长度，范围 0-1
    - 当前相位时间：从当前相位开始到现在的持续时间（秒），除以最大相位时长，范围 0-1
    - 黄灯指示器：黄灯相位指示器（one-hot编码）
    - 当前相位：当前绿灯相位的one-hot编码
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
        # 更新被检测到的车辆
        self._update_detected_vehicles()
        
        # 一次遍历车道，同时计算车辆数和最近距离
        normalized_car_counts = []
        normalized_distances = []
        
        for lane in self.ts.lanes:
            # 获取车道上所有车辆
            veh_list = self.ts.sumo.lane.getLastStepVehicleIDs(lane)
            # 筛选被检测到的车辆
            detected_veh_list = [veh for veh in veh_list if veh in self.detected_vehicles and self.detected_vehicles[veh]]
            
            # 计算检测车辆数并归一化
            detected_count = len(detected_veh_list)
            normalized_car_counts.append(min(detected_count / self.max_car_capacity, 1.0))
            
            # 计算最近车辆距离并归一化
            lane_length = self.ts.lanes_length[lane]
            if detected_veh_list:  # 如果有被检测到的车辆
                # 获取各车辆位置
                positions = [lane_length - self.ts.sumo.vehicle.getLanePosition(veh) for veh in detected_veh_list]
                # 最近的车辆距离
                min_distance = min(positions)
            else:  # 如果没有被检测到的车辆，设为车道长度
                min_distance = lane_length
            
            normalized_distances.append(min_distance / lane_length)
        
        # 当前相位持续时间，归一化到0-1范围
        normalized_phase_time = [min(float(self.ts.time_since_last_phase_change) / self.max_phase_duration, 1.0)]
        
        # 黄灯相位指示器(one-hot编码)
        amber_phase = [1.0 if self.ts.is_yellow else 0.0, 0.0 if self.ts.is_yellow else 1.0]
        
        # 处理固定时间交通信号灯模式 (fixed_ts=True)
        # 确保 n_phases 至少为 1，避免观察空间维度动态变化
        if hasattr(self.ts, 'green_phases'):
            n_phases = max(1, len(self.ts.green_phases))
            # 当前绿灯相位的one-hot编码
            current_phase_onehot = [0.0] * n_phases
            if not self.ts.is_yellow:  # 不在黄灯相位时
                current_phase_onehot[self.ts.green_phase] = 1.0
        elif hasattr(self.ts, 'num_green_phases'):
            # fixed_ts模式下
            n_phases = max(1, self.ts.num_green_phases)
            # 当前绿灯相位的one-hot编码
            current_phase_onehot = [0.0] * n_phases
            if not self.ts.is_yellow:  # 不在黄灯相位时
                current_phase_onehot[self.ts.green_phase] = 1.0
        else:
            # 如果都没有，默认为1个相位，确保维度固定
            n_phases = 1
            current_phase_onehot = [0.0] * n_phases
        
        # 当前时间（一天中的小时数，归一化到0-1）
        current_time = [float(self.ts.env.sim_step % (24 * 3600)) / (24 * 3600)]
        
        # 合并所有观察值
        observation = np.array(
            normalized_car_counts + 
            normalized_distances + 
            normalized_phase_time + 
            amber_phase + 
            current_phase_onehot + 
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
        
        观察空间的维度由以下部分组成（所有特征均已归一化）：
        - 各车道检测车辆数量 (0-1)
        - 各车道最近检测车辆距离 (0-1)
        - 当前相位时间 (0-1)
        - 黄灯相位指示器(one-hot编码, 2维)
        - 当前绿灯相位(one-hot编码)
        - 当前时间 (0-1)
        
        Returns:
            spaces.Box: 观察空间
        """
        # 计算观察空间的维度
        n_lanes = len(self.ts.lanes)
        
        # 处理固定时间交通信号灯模式 (fixed_ts=True)
        # 确保 n_phases 至少为 1，与 __call__ 方法保持一致
        if hasattr(self.ts, 'green_phases'):
            n_phases = max(1, len(self.ts.green_phases))
        elif hasattr(self.ts, 'num_green_phases'):
            # fixed_ts模式下使用num_green_phases
            n_phases = max(1, self.ts.num_green_phases)
        else:
            # 如果都没有，默认为1
            n_phases = 1
        
        # 设置上下界
        return spaces.Box(
            low=np.array(
                [0] * n_lanes +        # 归一化车辆数（0-1）
                [0] * n_lanes +        # 归一化距离（0-1）
                [0] +                  # 归一化相位时间（0-1）
                [0, 0] +               # 黄灯指示器(one-hot, 2维)
                [0] * n_phases +       # 当前相位(one-hot)
                [0],                   # 当前时间（0-1）
                dtype=np.float32
            ),
            high=np.array(
                [1] * n_lanes +        # 归一化车辆数（0-1）
                [1] * n_lanes +        # 归一化距离（0-1）
                [1] +                  # 归一化相位时间（0-1）
                [1, 1] +               # 黄灯指示器(one-hot, 2维)
                [1] * n_phases +       # 当前相位(one-hot)
                [1],                   # 当前时间（0-1）
                dtype=np.float32
            ),
        )

