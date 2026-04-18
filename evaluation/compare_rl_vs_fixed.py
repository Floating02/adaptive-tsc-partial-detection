"""评估和对比RL与固定信号控制的性能

该脚本用于评估训练好的RL模型与固定信号控制的性能对比，
包括平均等待时间、平均速度、队列长度等指标。

使用方法:
python evaluation/compare_rl_vs_fixed.py --model_path models/dqn_table_i_dr0.7 --detection_rate 0.7 --n_runs 5
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# 设置SUMO环境变量
# 注意：在Windows多进程环境下，Libsumo可能与SubprocVecEnv存在兼容性问题
# 如果遇到问题，可以尝试禁用Libsumo（注释下面这一行）
os.environ["LIBSUMO_AS_TRACI"] = "1"

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sumo_rl import SumoEnvironment
from observations.observation import TableIObservationFunction
from rewards import average_speed_reward, mixed_reward

# 奖励函数映射表（与 train.py 保持一致）
REWARD_FUNCTIONS = {
    'average-speed': average_speed_reward,
    'mixed': mixed_reward,
}

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Evaluate and compare RL vs fixed signal control')
    
    # 模型参数
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pre-trained model, e.g., models/dqn_table_i_dr0.7")
    parser.add_argument("--detection_rate", type=float, default=0.7,
                        help="Probability of vehicles being detected (between 0 and 1)")
    parser.add_argument("--reward_fn", type=str, default='mixed',
                        choices=['average-speed', 'mixed'],
                        help="Reward function used during training: 'average-speed' or 'mixed'")
    
    # 环境参数
    parser.add_argument("--gui", action="store_true", 
                        help="Use SUMO GUI for visualization")
    parser.add_argument("--net", type=str, 
                        default="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
                        help="Path to SUMO network file")
    parser.add_argument("--route", type=str, 
                        default="sumo_rl/nets/2way-single-intersection/single-intersection-poisson-none.rou.xml",
                        help="Path to SUMO route file")
    parser.add_argument("--output_dir", type=str, default="evaluation/results",
                        help="Directory for evaluation outputs")
    parser.add_argument("--n_runs", type=int, default=5,
                        help="Number of evaluation runs")
    parser.add_argument("--eval_duration", type=int, default=9000,
                        help="Duration of each evaluation simulation (seconds)")
    
    return parser.parse_args()

def create_rl_env(net_file, route_file, use_gui, detection_rate, eval_duration, reward_fn='mixed', seed=None):
    """创建RL评估环境
    
    Args:
        net_file (str): Path to SUMO network file
        route_file (str): Path to SUMO route file
        use_gui (bool): Whether to use SUMO GUI
        detection_rate (float): Probability of vehicles being detected
        eval_duration (int): Evaluation duration (seconds)
        reward_fn (str): Reward function name ('average-speed' or 'mixed')
        seed (int, optional): Random seed
        
    Returns:
        SumoEnvironment: Created SUMO environment
    """
    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)
    
    # 获取奖励函数
    if isinstance(reward_fn, str):
        if reward_fn not in REWARD_FUNCTIONS:
            raise ValueError(f"Unknown reward function: {reward_fn}. Available: {list(REWARD_FUNCTIONS.keys())}")
        reward_function = REWARD_FUNCTIONS[reward_fn]
    elif callable(reward_fn):
        reward_function = reward_fn
    else:
        raise ValueError("reward_fn must be a string name or callable function")
    
    # 创建环境
    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        out_csv_name=None,  # Don't output per-step CSV, only save final results
        use_gui=use_gui,
        begin_time=0,
        num_seconds=eval_duration,
        delta_time=5,
        yellow_time=3,
        min_green=5,
        max_green=50,
        enforce_max_green=True,
        single_agent=True,
        reward_fn=reward_function,  # 使用配置的奖励函数
        observation_class=lambda ts: TableIObservationFunction(ts, detection_rate=detection_rate, seed=seed + 4000 if seed is not None else None),
        sumo_seed=seed,
        add_system_info=True,  # Add system-level info for evaluation
    )
    
    return env

def create_fixed_env(net_file, route_file, use_gui, eval_duration, seed=None, detection_rate=0.7):
    """创建固定信号控制环境
    
    Args:
        net_file (str): Path to SUMO network file
        route_file (str): Path to SUMO route file
        use_gui (bool): Whether to use SUMO GUI
        eval_duration (int): Evaluation duration (seconds)
        seed (int, optional): Random seed
        detection_rate (float, optional): 检测率，用于保持与RL评估的观测空间一致性
        
    Returns:
        SumoEnvironment: Created SUMO environment
    """
    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)
    
    # 创建环境，设置fixed_ts=True
    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        out_csv_name=None,  # Don't output per-step CSV, only save final results
        use_gui=use_gui,
        begin_time=0,
        num_seconds=eval_duration,
        delta_time=5,
        yellow_time=3,
        min_green=5,
        max_green=50,
        enforce_max_green=True,
        single_agent=True,
        reward_fn="average-speed",
        observation_class=lambda ts: TableIObservationFunction(ts, detection_rate=detection_rate, seed=seed + 6000 if seed is not None else None),
        sumo_seed=seed,
        add_system_info=True,  # Add system-level info for evaluation
        fixed_ts=True,  # 使用固定信号控制
    )
    
    return env

def wrap_env(env):
    """包装环境用于评估
    
    Args:
        env: Original environment
        
    Returns:
        VecNormalize: Wrapped environment
    """
    # Wrap single environment with DummyVecEnv
    env = DummyVecEnv([lambda: env])
    
    # Add VecNormalize wrapper, but disable reward normalization
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99,
        epsilon=1e-8,
        training=False  # Set to False to avoid updating statistics during evaluation
    )
    
    return env

def evaluate_rl_model(model, env, n_episodes=5):
    """评估RL模型性能
    
    Args:
        model: 预训练的DQN模型
        env: 评估环境
        n_episodes (int): 评估轮数
        
    Returns:
        dict: 包含各种性能指标的字典
    """
    # 记录评估指标
    metrics_cache = {
        'rewards': [],
        'waiting_times': [],
        'queue_lengths': [],
        'speeds': [],
        'throughputs': []
    }
    
    try:
        for episode in range(n_episodes):
            print(f"RL模型评估轮次 {episode+1}/{n_episodes}")
            
            # 重置环境
            obs = env.reset()
            episode_reward = 0
            done = False
            info = None
            
            # 运行一个episode
            while not done.any():
                try:
                    # 预测动作
                    action, _ = model.predict(obs, deterministic=True)
                    # 执行动作
                    obs, reward, done, info = env.step(action)
                    episode_reward += reward[0]
                except Exception as e:
                    print(f"执行动作时出错: {e}")
                    break
            
            # 收集指标
            metrics_cache['rewards'].append(episode_reward)
            
            if info is not None and len(info) > 0:
                info_dict = info[0]
                
                # 使用get方法安全地获取指标
                metrics_cache['waiting_times'].append(
                    info_dict.get('system_mean_waiting_time', 0)
                )
                
                metrics_cache['queue_lengths'].append(
                    info_dict.get('system_total_stopped', 0)
                )
                
                metrics_cache['speeds'].append(
                    info_dict.get('system_mean_speed', 0)
                )
                
                metrics_cache['throughputs'].append(
                    info_dict.get('system_total_departed', 0)
                )
        
        # 计算平均指标
        results = {}
        for metric, values in metrics_cache.items():
            if values:  # 只处理非空列表
                results[f'average_{metric[:-1]}'] = np.mean(values)
                results[f'{metric[:-1]}_std'] = np.std(values)  # 添加标准差
        
        return results
    
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_fixed_signal(env, n_episodes=5):
    """评估固定信号控制性能
    
    Args:
        env: 评估环境（已用DummyVecEnv包装）
        n_episodes (int): 评估轮数
        
    Returns:
        dict: 包含各种性能指标的字典
    """
    # 记录评估指标
    metrics_cache = {
        'rewards': [],
        'waiting_times': [],
        'queue_lengths': [],
        'speeds': [],
        'throughputs': []
    }
    
    try:
        for episode in range(n_episodes):
            print(f"固定信号控制评估轮次 {episode+1}/{n_episodes}")
            
            # 重置环境
            obs = env.reset()
            episode_reward = 0
            done = False
            info = None
            
            # 运行一个episode
            while not done.any():
                try:
                    # 固定信号控制不需要动作，传入None
                    obs, reward, done, info = env.step(None)
                    episode_reward += reward[0]
                except Exception as e:
                    print(f"执行步骤时出错: {e}")
                    break
            
            # 收集指标
            metrics_cache['rewards'].append(episode_reward)
            
            if info is not None and len(info) > 0:
                info_dict = info[0]
                
                # 使用get方法安全地获取指标
                metrics_cache['waiting_times'].append(
                    info_dict.get('system_mean_waiting_time', 0)
                )
                
                metrics_cache['queue_lengths'].append(
                    info_dict.get('system_total_stopped', 0)
                )
                
                metrics_cache['speeds'].append(
                    info_dict.get('system_mean_speed', 0)
                )
                
                metrics_cache['throughputs'].append(
                    info_dict.get('system_total_departed', 0)
                )
        
        # 计算平均指标
        results = {}
        for metric, values in metrics_cache.items():
            if values:  # 只处理非空列表
                results[f'average_{metric[:-1]}'] = np.mean(values)
                results[f'{metric[:-1]}_std'] = np.std(values)  # 添加标准差
        
        return results
    
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_comparison(rl_results, fixed_results, output_dir):
    """可视化对比结果
    
    Args:
        rl_results (dict): RL模型评估结果
        fixed_results (dict): 固定信号控制评估结果
        output_dir (str): 输出目录
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置Seaborn样式
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'font.size': 12,
            'figure.figsize': (12, 8),
            'savefig.dpi': 300
        })
        
        # 主要指标可视化
        metrics = {
            'average_waiting_time': '平均等待时间 (秒)',
            'average_queue_length': '平均队列长度 (辆)',
            'average_speed': '平均速度 (m/s)',
            'average_throughput': '平均通过量 (辆/小时)'
        }
        
        rl_values = []
        fixed_values = []
        labels = []
        rl_errors = []
        fixed_errors = []
        
        for metric, label in metrics.items():
            if metric in rl_results and metric in fixed_results:
                rl_values.append(rl_results[metric])
                fixed_values.append(fixed_results[metric])
                labels.append(label)
                rl_errors.append(rl_results.get(f'{metric[:-7]}_std', 0))
                fixed_errors.append(fixed_results.get(f'{metric[:-7]}_std', 0))
        
        if rl_values and fixed_values:
            # 创建对比柱状图
            x = np.arange(len(labels))
            width = 0.35
            
            fig, ax = plt.subplots()
            rl_bars = ax.bar(x - width/2, rl_values, width, yerr=rl_errors, capsize=5, label='RL控制', color='skyblue')
            fixed_bars = ax.bar(x + width/2, fixed_values, width, yerr=fixed_errors, capsize=5, label='固定信号控制', color='lightgreen')
            
            # 添加数值标签
            for bar in rl_bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}',
                        ha='center', va='bottom')
            
            for bar in fixed_bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}',
                        ha='center', va='bottom')
            
            ax.set_ylabel('性能指标')
            ax.set_title('RL控制 vs 固定信号控制性能对比')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'comparison_metrics.png'))
            plt.close()
        
        # 计算性能提升百分比
        improvement = {}
        for metric in metrics.keys():
            if metric in rl_results and metric in fixed_results:
                if 'waiting' in metric or 'queue' in metric:
                    # 对于等待时间和队列长度，越小越好
                    improvement[metric] = ((fixed_results[metric] - rl_results[metric]) / fixed_results[metric]) * 100
                else:
                    # 对于速度和通过量，越大越好
                    improvement[metric] = ((rl_results[metric] - fixed_results[metric]) / fixed_results[metric]) * 100
        
        # 可视化性能提升
        if improvement:
            plt.figure()
            metrics_labels = [metrics[m] for m in improvement.keys()]
            improvement_values = list(improvement.values())
            
            bars = plt.bar(metrics_labels, improvement_values, color='green', alpha=0.7)
            
            # 添加数值标签
            for bar, value in zip(bars, improvement_values):
                height = value
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{value:.1f}%',
                        ha='center', va='bottom')
            
            plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            plt.title('RL控制相比固定信号控制的性能提升')
            plt.ylabel('性能提升百分比 (%)')
            plt.xticks(rotation=15)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'performance_improvement.png'))
            plt.close()
        
        # 保存结果到CSV
        comparison_df = pd.DataFrame({
            '指标': [metrics[m] for m in metrics.keys()],
            'RL控制': [rl_results.get(m, 'N/A') for m in metrics.keys()],
            '固定信号控制': [fixed_results.get(m, 'N/A') for m in metrics.keys()],
            '性能提升': [f"{improvement.get(m, 'N/A'):.1f}%" if isinstance(improvement.get(m), (int, float)) else 'N/A' for m in metrics.keys()]
        })
        
        csv_path = os.path.join(output_dir, f'comparison_results_{time.strftime("%Y%m%d-%H%M%S")}.csv')
        comparison_df.to_csv(csv_path, index=False)
        print(f"对比结果已保存到: {csv_path}")
        
    except Exception as e:
        print(f"可视化结果时出错: {e}")
        import traceback
        traceback.print_exc()

def run_evaluation(args):
    """运行模型评估和对比
    
    Args:
        args: 命令行参数
    """
    try:
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 设置全局随机种子
        global_seed = 42
        np.random.seed(global_seed)
        
        print(f"加载模型: {args.model_path}")
        
        # 检查模型文件
        if not (os.path.exists(args.model_path) or os.path.exists(args.model_path + ".zip")):
            print(f"未找到模型文件: {args.model_path}")
            return None
        
        # 加载模型
        try:
            model = DQN.load(args.model_path)
            print("模型加载成功!")
            
            # 加载归一化统计信息
            norm_path = f"{args.model_path}_vec_normalize.pkl"
            if not os.path.exists(norm_path):
                norm_path = f"{args.model_path}.pkl"
                if not os.path.exists(norm_path):
                    print("未找到归一化统计信息文件，将创建新的归一化环境")
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return None
        
        # 创建RL评估环境
        rl_raw_env = create_rl_env(
            net_file=args.net,
            route_file=args.route,
            use_gui=args.gui,
            detection_rate=args.detection_rate,
            eval_duration=args.eval_duration,
            reward_fn=args.reward_fn,  # 传递奖励函数参数
            seed=global_seed
        )
        
        # 包装环境
        rl_env = wrap_env(rl_raw_env)
        
        # 加载归一化统计信息
        if os.path.exists(norm_path):
            print(f"加载归一化统计信息: {norm_path}")
            rl_env = VecNormalize.load(norm_path, rl_env)
            rl_env.training = False
            rl_env.norm_reward = False
        
        # 创建固定信号控制环境
        fixed_raw_env = create_fixed_env(
            net_file=args.net,
            route_file=args.route,
            use_gui=args.gui,
            eval_duration=args.eval_duration,
            seed=global_seed,
            detection_rate=args.detection_rate
        )
        
        # 包装固定信号控制环境（与RL评估一致）
        fixed_env = wrap_env(fixed_raw_env)
        
        # 运行RL模型评估
        print(f"开始评估RL模型，共{args.n_runs}轮...")
        rl_results = evaluate_rl_model(model, rl_env, n_episodes=args.n_runs)
        
        # 运行固定信号控制评估
        print(f"开始评估固定信号控制，共{args.n_runs}轮...")
        fixed_results = evaluate_fixed_signal(fixed_env, n_episodes=args.n_runs)
        
        if rl_results is None or fixed_results is None:
            print("评估失败，未生成结果")
            return None
        
        # 生成可视化结果
        visualize_comparison(rl_results, fixed_results, args.output_dir)
        print(f"可视化结果已保存到: {args.output_dir}")
        
        # 关闭环境
        rl_env.close()
        fixed_env.close()
        
        # 打印评估结果
        print("\n评估结果:")
        print("RL模型:")
        for metric, value in rl_results.items():
            if not metric.endswith('_std'):
                print(f"  - {metric}: {value:.4f} ± {rl_results.get(f'{metric}_std', 0):.4f}")
        
        print("\n固定信号控制:")
        for metric, value in fixed_results.items():
            if not metric.endswith('_std'):
                print(f"  - {metric}: {value:.4f} ± {fixed_results.get(f'{metric}_std', 0):.4f}")
        
        # 计算性能提升
        print("\n性能提升:")
        metrics = ['average_waiting_time', 'average_queue_length', 'average_speed', 'average_throughput']
        for metric in metrics:
            if metric in rl_results and metric in fixed_results:
                if 'waiting' in metric or 'queue' in metric:
                    # 对于等待时间和队列长度，越小越好
                    improvement = ((fixed_results[metric] - rl_results[metric]) / fixed_results[metric]) * 100
                else:
                    # 对于速度和通过量，越大越好
                    improvement = ((rl_results[metric] - fixed_results[metric]) / fixed_results[metric]) * 100
                print(f"  - {metric}: {improvement:.1f}%")
        
        return rl_results, fixed_results
        
    except Exception as e:
        print(f"评估过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Add debug info
    print("开始评估和对比脚本，参数如下:")
    print(f"  - 模型路径: {args.model_path}")
    print(f"  - 检测率: {args.detection_rate}")
    print(f"  - 奖励函数: {args.reward_fn}")
    print(f"  - 评估次数: {args.n_runs}")
    print(f"  - 评估持续时间: {args.eval_duration}秒")
    print(f"  - 输出目录: {args.output_dir}")
    
    # Check if model file exists before trying to run evaluation
    # Try both with and without .zip extension
    model_exists = os.path.exists(args.model_path) or os.path.exists(args.model_path + ".zip")
    
    if not model_exists:
        print(f"错误: 未找到模型文件 {args.model_path} 或 {args.model_path}.zip")
        sys.exit(1)
    
    try:
        # Run evaluation
        results = run_evaluation(args)
        
        if results is None:
            print("评估未能完成，未生成结果。")
            sys.exit(1)
            
        print("\n评估完成!")
    except Exception as e:
        import traceback
        print(f"评估过程中出错: {e}")
        traceback.print_exc()
        
        # Try to close any remaining SUMO processes
        try:
            import traci
            traci.close()
        except:
            pass
