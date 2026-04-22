"""强化学习Episode调试脚本

在训练或推理过程中添加详细的打印语句，输出每个时间步的关键信息：
- 当前观测值 (obs)
- 智能体选择的动作 (action)
- 获得的奖励值 (reward)

使用方法:
    python experiments/run_episode_debug.py --mode train --detection_rate 0.7
    python experiments/run_episode_debug.py --mode inference --detection_rate 0.7 --model_path path/to/model
"""

import os
import sys
import argparse
import numpy as np
import torch

os.environ["LIBSUMO_AS_TRACI"] = "1"

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("请声明环境变量'SUMO_HOME'")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sumo_rl import SumoEnvironment
from observations.observation import TableIObservationFunction
from rewards import average_speed_reward, mixed_reward
from stable_baselines3 import DQN


import sumo_rl
import os

pkg_dir = os.path.dirname(sumo_rl.__file__)
DEFAULT_NET = os.path.join(pkg_dir, "nets", "2way-single-intersection", "single-intersection.net.xml")
DEFAULT_ROUTE = os.path.join(pkg_dir, "nets", "2way-single-intersection", "single-intersection-poisson.rou.xml")


def run_debug_episode(net_file=None, route_file=None, detection_rate=0.7, reward_fn_name='average-speed',
                      num_seconds=300, seed=42, use_gui=False, model_path=None,
                      max_steps=60, eval_begin_time=28800):
    """运行单个episode并打印每个时间步的详细信息

    Args:
        net_file: SUMO网络文件路径
        route_file: SUMO路由文件路径
        detection_rate: 车辆检测率
        reward_fn_name: 奖励函数名称
        num_seconds: 模拟总时长
        seed: 随机种子
        use_gui: 是否使用GUI
        model_path: 模型路径（若为None则使用随机策略）
        max_steps: 最大打印步数（防止输出过多）
        eval_begin_time: 评估开始时间
    """
    if net_file is None:
        net_file = DEFAULT_NET
    if route_file is None:
        route_file = DEFAULT_ROUTE
    REWARD_FUNCTIONS = {
        'average-speed': average_speed_reward,
        'mixed': mixed_reward,
    }
    reward_function = REWARD_FUNCTIONS[reward_fn_name]

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=use_gui,
        begin_time=eval_begin_time,
        num_seconds=num_seconds,
        delta_time=5,
        yellow_time=3,
        min_green=5,
        max_green=50,
        single_agent=True,
        reward_fn=reward_function,
        observation_class=lambda ts: TableIObservationFunction(
            ts, detection_rate=detection_rate, seed=seed
        ),
        sumo_seed=seed,
    )

    print("=" * 80)
    print("强化学习Episode调试信息")
    print("=" * 80)
    print(f"检测率 (Detection Rate): {detection_rate}")
    print(f"奖励函数 (Reward Function): {reward_fn_name}")
    print(f"随机种子 (Seed): {seed}")
    print(f"模拟时长 (Duration): {num_seconds} 秒")
    print(f"评估开始时间 (Begin Time): {eval_begin_time} ({eval_begin_time//3600:02d}:{(eval_begin_time%3600)//60:02d})")
    print(f"模型路径 (Model Path): {model_path if model_path else '随机策略 (Random Policy)'}")
    print("=" * 80)
    print()

    if model_path and os.path.exists(model_path + '.zip'):
        print(f"[INFO] 加载模型: {model_path}")
        model = DQN.load(model_path)
        deterministic = True
    else:
        print("[INFO] 未找到模型，使用随机策略 (Random Policy)")
        model = None
        deterministic = False

    obs, info = env.reset()
    obs_shape = obs.shape[0] if hasattr(obs, 'shape') else len(obs)
    print(f"观测维度: {obs_shape}")
    print("\n" + "=" * 80)
    print(f"{'时间步':^8} | {'动作':^6} | {'奖励 (Reward)':^15} | waiting_time | queue | speed")
    print("-" * 80)

    step_count = 0
    total_reward = 0.0

    done = False
    truncated = False

    while not (done or truncated) and step_count < max_steps:
        step_count += 1

        if model is not None:
            action, _ = model.predict(obs, deterministic=deterministic)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated

        total_reward += reward if isinstance(reward, (int, float)) else reward[0]
        reward_val = reward if isinstance(reward, (int, float)) else reward[0]

        waiting = info.get('system_mean_waiting_time', 0) if info else 0
        queue = info.get('system_total_stopped', 0) if info else 0
        speed = info.get('system_mean_speed', 0) if info else 0

        print(f"{step_count:^8} | {action:^6} | {reward_val:^15.6f} | {waiting:^12.2f} | {queue:^5} | {speed:^5.2f}")

        obs_str = np.array2string(obs, precision=4, separator=', ')
        print(f"       obs[{obs_shape}]: {obs_str}")

        print("-" * 80)
    print(f"\nEpisode结束:")
    print(f"  总时间步数 (Total Steps): {step_count}")
    print(f"  总奖励值 (Total Reward): {total_reward:.6f}")
    print(f"  平均奖励 (Avg Reward): {total_reward/step_count if step_count > 0 else 0:.6f}")
    print()

    env.close()

    return step_count, total_reward


def parse_args():
    parser = argparse.ArgumentParser(description="强化学习Episode调试脚本")

    parser.add_argument("--mode", type=str, default="inference", choices=["train", "inference"],
                        help="运行模式: train(训练) 或 inference(推理)")
    parser.add_argument("--detection_rate", type=float, default=0.7,
                        help="车辆检测率 (0.0-1.0)")
    parser.add_argument("--reward_fn", type=str, default="average-speed",
                        choices=["average-speed", "mixed"],
                        help="奖励函数")
    parser.add_argument("--num_seconds", type=int, default=300,
                        help="模拟时长（秒）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--max_steps", type=int, default=60,
                        help="最大打印步数")
    parser.add_argument("--net", type=str,
                        default="nets/2way-single-intersection/single-intersection.net.xml",
                        help="SUMO网络文件路径")
    parser.add_argument("--route", type=str,
                        default="nets/2way-single-intersection/single-intersection-poisson.rou.xml",
                        help="SUMO路由文件路径")
    parser.add_argument("--model_path", type=str, default=None,
                        help="模型路径 (.zip文件，不带扩展名)")
    parser.add_argument("--begin_time", type=int, default=28800,
                        help="评估开始时间（秒）")
    parser.add_argument("--gui", action="store_true",
                        help="启用SUMO GUI")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_debug_episode(
        net_file=args.net,
        route_file=args.route,
        detection_rate=args.detection_rate,
        reward_fn_name=args.reward_fn,
        num_seconds=args.num_seconds,
        seed=args.seed,
        use_gui=args.gui,
        model_path=args.model_path,
        max_steps=args.max_steps,
        eval_begin_time=args.begin_time,
    )