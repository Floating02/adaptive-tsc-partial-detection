"""部分检测场景下的DQN交通信号控制实验

这个脚本实现了一个基于部分检测的状态表示方法，该方法包括：
- 每个路径的检测车辆数量
- 每个路径上最近检测车辆的距离
- 当前相位时间
- 黄灯相位指示器
- 当前时间
- 通过正负号区分红绿灯状态

运行此脚本后，可以使用TensorBoard查看训练过程中的奖励曲线和其他指标：
1. 安装TensorBoard: pip install tensorboard
2. 在命令行中运行: tensorboard --logdir=./logs
3. 打开浏览器访问: http://localhost:6006
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.nn import functional as F
import multiprocessing
import logging
import json
from typing import Callable
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
import threading
from datetime import datetime

# 设置SUMO环境变量
# 注意：在Windows多进程环境下，Libsumo可能与SubprocVecEnv存在兼容性问题
# 如果遇到问题，可以尝试禁用Libsumo（注释下面这一行）
os.environ["LIBSUMO_AS_TRACI"] = "1"

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("请声明环境变量'SUMO_HOME'")

# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sumo_rl import SumoEnvironment
from observations.observation import TableIObservationFunction
from rewards import average_speed_reward, mixed_reward


def setup_logging(log_dir: str = "logs", experiment_id: str = None) -> logging.Logger:
    """设置日志记录系统
    
    Args:
        log_dir: 日志目录
        experiment_id: 实验ID，用于命名日志文件
    
    Returns:
        配置好的logger对象
    """
    if experiment_id is None:
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"experiment_{experiment_id}.log")
    
    # 配置日志
    logger = logging.getLogger(f"experiment_{experiment_id}")
    logger.setLevel(logging.DEBUG)
    
    # 文件处理器 - 记录所有级别
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # 控制台处理器 - 只记录INFO及以上
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def generate_experiment_id() -> str:
    """生成唯一的实验ID"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_experiment_config(config: dict, config_dir: str = "configs", experiment_id: str = None):
    """保存实验配置到JSON文件
    
    Args:
        config: 配置字典
        config_dir: 配置目录
        experiment_id: 实验ID
    """
    if experiment_id is None:
        experiment_id = generate_experiment_id()
    
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, f"config_{experiment_id}.json")
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    return config_file


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="部分检测场景下的DQN交通信号控制实验")
    
    # 检测率参数
    parser.add_argument("--detection_rate", type=float, default=0.2,
                       help="车辆被检测到的概率 (0到1之间)")
    
    # 奖励函数参数
    parser.add_argument("--reward_fn", type=str, default='mixed',
                       choices=['average-speed', 'mixed'],
                       help="奖励函数类型: 'average-speed' (平均速度) 或 'mixed' (混合奖励)")
    
    # 环境参数
    parser.add_argument("--gui", action="store_true", 
                       help="是否使用SUMO GUI")
    parser.add_argument("--net", type=str, 
                       default="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
                       help="SUMO网络文件路径")
    parser.add_argument("--route", type=str, 
                       default="sumo_rl/nets/2way-single-intersection/single-intersection-poisson.rou.xml",
                       help="SUMO路由文件路径，默认使用基于24小时泊松分布的车流模型")
    
    # 输出控制参数
    parser.add_argument("--out_csv_name", type=str, default="outputs/dqn_table_i",
                       help="输出CSV文件名前缀")
    parser.add_argument("--save_csv", action="store_true", default=False,
                       help="是否保存详细的CSV输出文件")
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="实验名称，用于组织输出文件")
    
    # 训练参数
    parser.add_argument("--total_timesteps", type=int, default=6_000_000,
                       help="总训练步数")
    parser.add_argument("--n_envs", type=int, default=8,
                       help="并行环境数量")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="初始学习率")
    
    # 模型保存参数
    parser.add_argument("--save_freq", type=int, default=500_000,
                       help="模型保存频率（步数）")

    return parser.parse_args()

# 奖励函数映射表，用于通过名称获取奖励函数
REWARD_FUNCTIONS = {
    'average-speed': average_speed_reward,
    'mixed': mixed_reward,
}


def make_env(net_file, route_file, out_csv_name, detection_rate, reward_fn='mixed', 
             use_gui=False, seed=42, save_csv=False, env_index=0):
    """创建单个环境的工厂函数
    
    Args:
        net_file: SUMO网络文件
        route_file: SUMO路由文件
        out_csv_name: 输出CSV文件名前缀
        detection_rate: 车辆检测率
        reward_fn: 奖励函数，可以是字符串名称（'average-speed' 或 'mixed'）或函数对象
                   默认为 'mixed'（混合奖励函数）
        use_gui: 是否使用GUI
        seed: 基础随机种子
        save_csv: 是否保存CSV输出
        env_index: 环境索引，用于区分不同的并行环境
    
    Returns:
        环境工厂函数
    """
    # 如果 reward_fn 是字符串，从映射表中获取对应的函数
    if isinstance(reward_fn, str):
        if reward_fn not in REWARD_FUNCTIONS:
            raise ValueError(f"未知的奖励函数: {reward_fn}. 可用的奖励函数: {list(REWARD_FUNCTIONS.keys())}")
        reward_function = REWARD_FUNCTIONS[reward_fn]
    elif callable(reward_fn):
        reward_function = reward_fn
    else:
        raise ValueError("reward_fn 必须是字符串名称或可调用函数对象")
    
    def _init():
        """在子进程中初始化环境
        
        注意：这个函数在 SubprocVecEnv 的子进程中执行，因此需要在每个子进程中
        独立设置随机种子，确保实验的可复现性。
        """
        import random as python_random
        
        # 为当前环境计算唯一的种子
        # 使用基础种子 + 环境索引，确保每个环境有不同的但可复现的种子
        env_seed = seed + env_index
        
        # 在子进程中设置所有随机源种子
        np.random.seed(env_seed)                    # NumPy 种子
        torch.manual_seed(env_seed)                 # PyTorch 种子
        python_random.seed(env_seed)                # Python random 种子
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(env_seed)    # CUDA 种子
        
        # 固定模拟持续时间为9000秒
        sim_duration = 9000
        
        # 使用独立的随机数生成器生成开始时间，避免影响全局状态
        rng = np.random.RandomState(env_seed + 1000)  # 使用偏移量避免与其他随机数冲突
        start_time = int(rng.randint(0, 86400 - sim_duration))
        
        # 根据save_csv参数决定是否输出CSV
        env_kwargs = {
            'net_file': net_file,
            'route_file': route_file,
            'use_gui': use_gui,
            'begin_time': start_time,
            'num_seconds': sim_duration,
            'delta_time': 5,
            'yellow_time': 3,
            'min_green': 5,
            'max_green': 50,
            'enforce_max_green': True,
            'single_agent': True,
            'reward_fn': reward_function,  # 使用配置的奖励函数
            'observation_class': lambda ts: TableIObservationFunction(ts, detection_rate=detection_rate, seed=env_seed + 2000),
            'sumo_seed': env_seed,  # 使用环境特定的种子
        }
        
        if save_csv:
            env_kwargs['out_csv_name'] = f"{out_csv_name}_dr{detection_rate}_seed{seed}_env{env_index}"
        
        env = SumoEnvironment(**env_kwargs)
        return env
    return _init


def generate_training_summary(model, env, experiment_id: str, logger: logging.Logger) -> dict:
    """生成训练总结
    
    Args:
        model: 训练好的模型
        env: 环境
        experiment_id: 实验ID
        logger: 日志记录器
    
    Returns:
        包含总结信息的字典
    """
    summary = {
        'experiment_id': experiment_id,
        'completion_time': datetime.now().isoformat(),
        'model_info': {
            'algorithm': 'DQN',
            'policy': 'MlpPolicy',
        },
        'environment_info': {
            'observation_space': str(env.observation_space),
            'action_space': str(env.action_space),
        }
    }
    
    logger.info("=" * 60)
    logger.info("训练总结")
    logger.info("=" * 60)
    logger.info(f"实验ID: {experiment_id}")
    logger.info(f"完成时间: {summary['completion_time']}")
    logger.info("=" * 60)
    
    return summary


def save_training_summary(summary: dict, summary_dir: str = "summaries"):
    """保存训练总结到文件
    
    Args:
        summary: 总结字典
        summary_dir: 总结目录
    """
    os.makedirs(summary_dir, exist_ok=True)
    summary_file = os.path.join(summary_dir, f"summary_{summary['experiment_id']}.json")
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    return summary_file


def visualize_training_curves(log_dir: str, experiment_id: str, output_dir: str = "plots"):
    """从TensorBoard日志生成训练曲线图
    
    Args:
        log_dir: TensorBoard日志目录
        experiment_id: 实验ID
        output_dir: 输出目录
    """
    try:
        import matplotlib.pyplot as plt
        from tensorboard.backend.event_processing import event_accumulator
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 查找最新的TensorBoard事件文件
        tb_log_dir = os.path.join(log_dir, "v1")
        if not os.path.exists(tb_log_dir):
            return None
        
        # 获取最新的事件文件
        event_files = []
        for root, dirs, files in os.walk(tb_log_dir):
            for file in files:
                if file.startswith("events.out.tfevents"):
                    event_files.append(os.path.join(root, file))
        
        if not event_files:
            return None
        
        # 按修改时间排序，取最新的
        event_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_event_file = event_files[0]
        
        # 读取TensorBoard事件
        ea = event_accumulator.EventAccumulator(
            latest_event_file,
            size_guidance={
                event_accumulator.SCALARS: 0,
            }
        )
        ea.Reload()
        
        # 生成图表
        if 'rollout/ep_rew_mean' in ea.Tags()['scalars']:
            scalar = ea.Scalars('rollout/ep_rew_mean')
            steps = [s.step for s in scalar]
            values = [s.value for s in scalar]
            
            plt.figure(figsize=(12, 6))
            plt.plot(steps, values)
            plt.title('Episode Reward Mean')
            plt.xlabel('Steps')
            plt.ylabel('Reward')
            plt.grid(True)
            plt.tight_layout()
            
            plot_file = os.path.join(output_dir, f"reward_curve_{experiment_id}.png")
            plt.savefig(plot_file)
            plt.close()
            
            return plot_file
    
    except ImportError:
        pass
    except Exception as e:
        pass
    
    return None

def run_experiment(args):
    """运行部分检测场景下的DQN交通信号控制实验"""
    
    # 生成实验ID
    experiment_id = generate_experiment_id()
    if args.experiment_name:
        experiment_id = f"{args.experiment_name}_{experiment_id}"
    
    # 设置日志系统
    logger = setup_logging(log_dir="logs", experiment_id=experiment_id)
    logger.info("=" * 60)
    logger.info("开始部分检测场景下的DQN交通信号控制实验")
    logger.info("=" * 60)
    logger.info(f"实验ID: {experiment_id}")
    logger.info(f"检测率: {args.detection_rate}")
    logger.info(f"奖励函数: {args.reward_fn}")
    logger.info(f"并行环境数量: {args.n_envs}")
    logger.info(f"总训练步数: {args.total_timesteps}")
    logger.info(f"学习率: {args.learning_rate}")
    logger.info(f"是否保存CSV: {args.save_csv}")
    logger.info("=" * 60)
    
    # 保存实验配置
    config = {
        'experiment_id': experiment_id,
        'detection_rate': args.detection_rate,
        'reward_fn': args.reward_fn,
        'n_envs': args.n_envs,
        'total_timesteps': args.total_timesteps,
        'learning_rate': args.learning_rate,
        'net_file': args.net,
        'route_file': args.route,
        'save_csv': args.save_csv,
        'experiment_name': args.experiment_name,
        'save_freq': args.save_freq,
    }
    config_file = save_experiment_config(config, config_dir="configs", experiment_id=experiment_id)
    logger.info(f"实验配置已保存到: {config_file}")
    
    # 设置全局随机种子以确保实验可复现
    global_seed = 42
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(global_seed)
        torch.cuda.manual_seed_all(global_seed)
    
    torch.set_num_threads(6)
    
    # 设置并行计算策略
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
    
    # 创建并行环境
    env_fns = []
    for i in range(args.n_envs):
        env_fn = make_env(
            net_file=args.net,
            route_file=args.route,
            out_csv_name=f"{args.out_csv_name}",
            detection_rate=args.detection_rate,
            reward_fn=args.reward_fn,  # 传递奖励函数参数
            use_gui=False,
            seed=global_seed,
            env_index=i,  # 传递环境索引，确保每个环境有唯一种子
            save_csv=args.save_csv
        )
        env_fns.append(env_fn)
    
    logger.info(f"创建了 {args.n_envs} 个并行环境，使用奖励函数: {args.reward_fn}")
    logger.info(f"全局种子: {global_seed}, 环境种子范围: {global_seed} - {global_seed + args.n_envs - 1}")
    
    # 创建SubprocVecEnv
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env)
    
    # 记录环境数量
    n_env = args.n_envs
    
    # 添加VecNormalize来归一化观察空间，不归一化奖励
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99,
        epsilon=1e-8
    )

    # 使用更大的网络架构
    policy_kwargs = dict(net_arch=[256, 256])
    
    # 创建并训练DQN模型
    logger.info("开始创建DQN模型")
    model = DQN(
        env=env,
        policy="MlpPolicy",
        policy_kwargs=policy_kwargs,
        learning_rate=linear_schedule(args.learning_rate),
        learning_starts=10000,
        train_freq=4 * n_env,
        gradient_steps=-1,
        target_update_interval=5000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.3,
        buffer_size=200000,
        batch_size=256,
        gamma=0.99,
        tensorboard_log="./logs/v1",
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=global_seed,
    )
    
    logger.info(f"开始训练，总步数: {args.total_timesteps}")
    model.learn(total_timesteps=args.total_timesteps)
    logger.info("训练完成")
    
    # 保存模型和归一化统计信息
    model_path = f"models/dqn_table_i_dr{args.detection_rate}_{args.reward_fn}_{experiment_id}"
    model.save(model_path)
    logger.info(f"模型已保存到: {model_path}")
    
    # 保存归一化统计信息
    norm_path = f"{model_path}_vec_normalize.pkl"
    env.save(norm_path)
    logger.info(f"归一化统计信息已保存到: {norm_path}")
    
    # 生成并保存训练总结
    logger.info("生成训练总结")
    summary = generate_training_summary(model, env, experiment_id, logger)
    summary_file = save_training_summary(summary, summary_dir="summaries")
    logger.info(f"训练总结已保存到: {summary_file}")
    
    # 生成可视化图表
    logger.info("生成训练曲线图")
    plot_file = visualize_training_curves("./logs", experiment_id, output_dir="plots")
    if plot_file:
        logger.info(f"训练曲线图已保存到: {plot_file}")
    else:
        logger.warning("未能生成训练曲线图（可能缺少matplotlib或tensorboard）")
    
    # 关闭环境
    env.close()
    
    logger.info("=" * 60)
    logger.info("实验完成！")
    logger.info("=" * 60)
    
    return model, env


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 确保输出目录存在
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    os.makedirs("summaries", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # 运行实验
    model, env = run_experiment(args) 