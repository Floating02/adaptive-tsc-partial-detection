"""Custom reward functions for SUMO-RL."""

from sumo_rl.environment.traffic_signal import TrafficSignal


def speed_based_reward(traffic_signal):
    """
    Implements a simplified reward function that directly uses average speed as reward
    
    where:
    - The reward is directly proportional to the average vehicle speed
    - Higher speed = higher reward
    
    This reward encourages higher vehicle speeds:
    - Maximum reward (1.0) when vehicles move at maximum allowed speed
    - Minimum reward (0.0) when vehicles are completely stopped
    """
    # 获取交通信号控制的路口当前的平均车速
    # get_average_speed()返回的是基于最大允许速度归一化的值（范围0-1）
    avg_speed = traffic_signal.get_average_speed()
    
    # 直接使用平均速度作为奖励
    # 车辆达到最大速度时奖励为1（最优）
    # 车辆完全停止时奖励为0（最差）
    return avg_speed


def mixed_reward(traffic_signal):
    """
    Implements a mixed reward function that considers both speed and queue length
    
    where:
    - Positive component: average speed (encourages higher speeds)
    - Negative component: queue length (discourages long queues)
    """
    # 获取平均速度
    avg_speed = traffic_signal.get_average_speed()
    
    # 获取总排队车辆数并归一化
    total_queue = traffic_signal.get_total_queued()
    # 假设最大排队长度为50，进行归一化
    normalized_queue = min(total_queue / 50, 1.0)
    
    # 混合奖励：速度权重0.7，队列权重0.3
    return 0.7 * avg_speed - 0.3 * normalized_queue


# 注册自定义奖励函数
print("正在注册自定义奖励函数...")
TrafficSignal.register_reward_fn(speed_based_reward)
TrafficSignal.register_reward_fn(mixed_reward)
print("自定义奖励函数注册成功!")
