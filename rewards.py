"""Custom reward functions for SUMO-RL."""

from sumo_rl.environment.traffic_signal import TrafficSignal


def average_speed_reward(ts):
    # get_average_speed()返回的是基于最大允许速度归一化的值（范围0-1）
    avg_speed = ts.get_average_speed()

    return avg_speed


def mixed_reward(ts):
    speed_reward = ts.get_average_speed()
    queue_len = ts.get_total_queued()
    max_cap = sum(
        ts.lanes_length[l] / max(1.0, ts.sumo.lane.getLastStepLength(l) + ts.MIN_GAP)
        for l in ts.lanes
    )
    norm_queue = min(1.0, queue_len / max(1.0, max_cap))
    pressure = ts.get_pressure()
    neg_pressure = abs(pressure) if pressure < 0 else 0.0
    norm_pressure = min(1.0, neg_pressure / max(1.0, max_cap))
    
    return 0.4 * speed_reward - 0.3 * norm_queue - 0.3 * norm_pressure


# 注册自定义奖励函数
def register_custom_rewards():
    """注册自定义奖励函数，避免重复注册"""
    for fn in [average_speed_reward, mixed_reward]:
        if fn.__name__ not in TrafficSignal.reward_fns:
            TrafficSignal.register_reward_fn(fn)

# 仅在首次导入时注册
register_custom_rewards()
