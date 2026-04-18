"""比较不同检测率的DQN交通信号控制模型性能

这个脚本用于比较不同检测率下训练的DQN交通信号控制模型的性能差异。
它将加载多个预训练模型（具有不同的检测率），在相同的交通场景下评估它们，
并生成比较性能指标的图表。

运行方式：
python evaluation/compare_detection_rates.py --detection_rates "0.3,0.5,0.7,0.9" --gui
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
    sys.exit("请声明环境变量'SUMO_HOME'")

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sumo_rl import SumoEnvironment
from observations.observation import TableIObservationFunction
from rewards import mixed_reward


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='比较不同检测率的DQN交通信号控制模型性能')
    
    # 模型参数
    parser.add_argument("--model_prefix", type=str, default="models/dqn_table_i_dr",
                        help="模型文件路径前缀，例如 models/dqn_table_i_dr")
    parser.add_argument("--reward_fn", type=str, default='mixed',
                        choices=['average-speed', 'mixed'],
                        help="训练时使用的奖励函数")
    
    # 环境参数
    parser.add_argument("--gui", action="store_true", 
                        help="是否使用SUMO GUI进行可视化")
    parser.add_argument("--net", type=str, 
                        default="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
                        help="SUMO网络文件路径")
    parser.add_argument("--route", type=str, 
                        default="sumo_rl/nets/2way-single-intersection/single-intersection-poisson.rou.xml",
                        help="SUMO路由文件路径")
    parser.add_argument("--output_dir", type=str, default="outputs/comparison",
                        help="比较结果输出目录")
    parser.add_argument("--n_runs", type=int, default=3,
                        help="每个模型的评估运行次数")
    parser.add_argument("--eval_duration", type=int, default=3600,
                        help="每次评估的模拟持续时间(秒)")
    parser.add_argument("--detection_rates", type=str, default="0.3,0.5,0.7,0.9",
                        help="用逗号分隔的检测率列表，例如 '0.3,0.5,0.7,0.9'")
    
    return parser.parse_args()


def create_eval_env(net_file, route_file, detection_rate, eval_duration, use_gui=False, seed=None):
    """创建评估环境
    
    Args:
        net_file (str): SUMO网络文件路径
        route_file (str): SUMO路由文件路径
        detection_rate (float): 检测率
        eval_duration (int): 评估持续时间（秒）
        use_gui (bool): 是否使用GUI
        seed (int, optional): 随机种子
    
    Returns:
        SumoEnvironment: 创建的环境
    """
    if seed is not None:
        np.random.seed(seed)
    
    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        out_csv_name=None,
        use_gui=use_gui,
        begin_time=0,
        num_seconds=eval_duration,
        delta_time=5,
        yellow_time=3,
        min_green=5,
        max_green=50,
        enforce_max_green=True,
        single_agent=True,
        reward_fn=mixed_reward,
        observation_class=lambda ts: TableIObservationFunction(ts, detection_rate=detection_rate, seed=seed + 3000 if seed is not None else None),
        sumo_seed=seed,
        add_system_info=True,
    )
    
    return env


def wrap_env(env):
    """包装环境用于评估
    
    Args:
        env: 原始环境
    
    Returns:
        VecNormalize: 包装后的环境
    """
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99,
        epsilon=1e-8,
        training=False
    )
    return env


def evaluate_model(model, env, n_episodes=3):
    """评估模型性能
    
    Args:
        model: 预训练的DQN模型
        env: 评估环境
        n_episodes (int): 评估轮数
    
    Returns:
        dict: 包含各种性能指标的字典
    """
    metrics_cache = {
        'rewards': [],
        'waiting_times': [],
        'queue_lengths': [],
        'speeds': [],
        'throughputs': []
    }
    
    try:
        for episode in range(n_episodes):
            obs = env.reset()
            episode_reward = 0
            done = False
            info = None
            
            while not done.any():
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]
            
            metrics_cache['rewards'].append(episode_reward)
            
            if info is not None and len(info) > 0:
                info_dict = info[0]
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
            if values:
                results[f'average_{metric[:-1]}'] = np.mean(values)
                results[f'{metric[:-1]}_std'] = np.std(values)
        
        return results
    
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_evaluation_for_detection_rate(detection_rate, args):
    """对特定检测率的模型进行评估
    
    Args:
        detection_rate (float): 检测率
        args: 命令行参数
    
    Returns:
        dict: 评估结果字典
    """
    # 构建模型路径
    model_path = f"{args.model_prefix}{detection_rate}"
    
    # 检查模型文件是否存在（尝试多种可能的命名格式）
    possible_paths = [
        f"{model_path}.zip",
        f"{model_path}_mixed.zip",
        f"{model_path}_average-speed.zip",
    ]
    
    model_found = False
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path.replace(".zip", "")
            model_found = True
            print(f"找到模型: {path}")
            break
    
    if not model_found:
        print(f"错误: 未找到检测率为 {detection_rate} 的模型 (尝试的路径: {possible_paths})")
        return None
    
    print(f"\n评估检测率 {detection_rate} 的模型...")
    
    try:
        # 加载模型
        model = DQN.load(model_path)
        
        # 创建评估环境
        raw_env = create_eval_env(
            net_file=args.net,
            route_file=args.route,
            detection_rate=detection_rate,
            eval_duration=args.eval_duration,
            use_gui=args.gui,
            seed=42
        )
        
        # 包装环境
        env = wrap_env(raw_env)
        
        # 尝试加载归一化统计信息
        norm_path = f"{model_path}_vec_normalize.pkl"
        if os.path.exists(norm_path):
            print(f"加载归一化统计信息: {norm_path}")
            env = VecNormalize.load(norm_path, env)
            env.training = False
            env.norm_reward = False
        
        # 运行评估
        results = evaluate_model(model, env, n_episodes=args.n_runs)
        
        # 关闭环境
        env.close()
        
        if results:
            print(f"检测率 {detection_rate} 评估完成:")
            for metric, value in results.items():
                if not metric.endswith('_std'):
                    std_value = results.get(f'{metric}_std', 0)
                    print(f"  - {metric}: {value:.4f} ± {std_value:.4f}")
        
        return results
    
    except Exception as e:
        print(f"评估检测率 {detection_rate} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def combine_results(all_results, detection_rates):
    """合并不同检测率的评估结果
    
    Args:
        all_results (list): 评估结果列表
        detection_rates (list): 对应的检测率列表
    
    Returns:
        pd.DataFrame: 合并后的结果数据框
    """
    combined_data = []
    
    for dr, results in zip(detection_rates, all_results):
        if results is not None:
            row = {'detection_rate': dr}
            row.update(results)
            combined_data.append(row)
    
    if combined_data:
        return pd.DataFrame(combined_data)
    else:
        return None


def visualize_comparison(df, output_dir):
    """可视化不同检测率下的性能比较
    
    Args:
        df (pd.DataFrame): 合并后的结果数据框
        output_dir (str): 输出目录
    """
    if df is None or len(df) == 0:
        print("没有可比较的结果")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置图表样式
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # 转换检测率为字符串，便于绘图
    df['detection_rate_str'] = df['detection_rate'].astype(str)
    
    # 绘制主要指标的条形图
    metrics = ['average_waiting_time', 'average_queue_length', 'average_speed']
    titles = ['平均等待时间 (秒)', '平均队列长度 (辆)', '平均速度 (m/s)']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        if metric in df.columns:
            sns.barplot(x='detection_rate_str', y=metric, data=df, ax=axes[i], palette='viridis')
            axes[i].set_title(title)
            axes[i].set_xlabel('检测率')
            
            # 在条形上方显示数值
            for j, p in enumerate(axes[i].patches):
                axes[i].annotate(f'{p.get_height():.2f}', 
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detection_rate_comparison_main.png'), dpi=300)
    plt.close()
    print(f"已保存主要指标对比图: {os.path.join(output_dir, 'detection_rate_comparison_main.png')}")
    
    # 绘制平均奖励的对比图
    if 'average_reward' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='detection_rate_str', y='average_reward', data=df, palette='viridis')
        plt.title('不同检测率下的平均奖励')
        plt.xlabel('检测率')
        plt.ylabel('平均奖励')
        
        for i, p in enumerate(plt.gca().patches):
            plt.gca().annotate(f'{p.get_height():.2f}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'detection_rate_comparison_reward.png'), dpi=300)
        plt.close()
    
    # 可选指标的绘制
    optional_metrics = ['average_throughput']
    available_metrics = [m for m in optional_metrics if m in df.columns]
    
    if available_metrics:
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(6 * len(available_metrics), 6))
        
        if len(available_metrics) == 1:
            axes = [axes]
        
        titles_dict = {
            'average_throughput': '平均吞吐量 (辆/小时)',
        }
        
        for i, metric in enumerate(available_metrics):
            sns.barplot(x='detection_rate_str', y=metric, data=df, ax=axes[i], palette='viridis')
            axes[i].set_title(titles_dict.get(metric, metric))
            axes[i].set_xlabel('检测率')
            
            for j, p in enumerate(axes[i].patches):
                axes[i].annotate(f'{p.get_height():.2f}', 
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'detection_rate_comparison_extra.png'), dpi=300)
        plt.close()
    
    # 创建雷达图比较不同检测率的多个指标
    radar_metrics = ['average_waiting_time', 'average_queue_length', 'average_speed']
    available_radar_metrics = [m for m in radar_metrics if m in df.columns]
    
    if len(available_radar_metrics) >= 3 and len(df) > 1:
        # 归一化数据到[0,1]区间
        df_norm = df.copy()
        
        for metric in available_radar_metrics:
            max_val = df[metric].max()
            min_val = df[metric].min()
            if max_val > min_val:
                if metric in ['average_waiting_time', 'average_queue_length']:
                    # 越小越好的指标，取反
                    df_norm[metric] = 1 - (df[metric] - min_val) / (max_val - min_val)
                else:
                    # 越大越好的指标
                    df_norm[metric] = (df[metric] - min_val) / (max_val - min_val)
            else:
                df_norm[metric] = 1.0
        
        plt.figure(figsize=(10, 10))
        
        angles = np.linspace(0, 2*np.pi, len(available_radar_metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        labels = {
            'average_waiting_time': '平均等待时间',
            'average_queue_length': '平均队列长度',
            'average_speed': '平均速度',
        }
        
        categories = [labels.get(m, m) for m in available_radar_metrics]
        categories += categories[:1]
        
        ax = plt.subplot(111, polar=True)
        
        for _, row in df_norm.iterrows():
            dr = row['detection_rate']
            values = [row[m] for m in available_radar_metrics]
            values += values[:1]
            ax.plot(angles, values, linewidth=2, label=f"检测率 {dr}")
            ax.fill(angles, values, alpha=0.1)
        
        plt.xticks(angles[:-1], categories[:-1])
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ['0.2', '0.4', '0.6', '0.8', '1.0'], color='gray', size=10)
        plt.ylim(0, 1)
        
        plt.title('不同检测率下的性能雷达图', size=15, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'detection_rate_radar_chart.png'), dpi=300)
        plt.close()
        print(f"已保存雷达图: {os.path.join(output_dir, 'detection_rate_radar_chart.png')}")


def run_comparison(args):
    """运行不同检测率的模型比较
    
    Args:
        args: 命令行参数
    """
    print("=" * 60)
    print("开始比较不同检测率的模型性能")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 解析检测率列表
    detection_rates = [float(dr) for dr in args.detection_rates.split(',')]
    print(f"将比较以下检测率: {detection_rates}")
    print(f"模型前缀: {args.model_prefix}")
    print(f"每个模型评估 {args.n_runs} 次")
    print(f"评估时长: {args.eval_duration} 秒")
    print("=" * 60)
    
    # 对每个检测率运行评估
    all_results = []
    for dr in detection_rates:
        results = run_evaluation_for_detection_rate(dr, args)
        all_results.append(results)
    
    # 合并评估结果
    combined_results = combine_results(all_results, detection_rates)
    
    # 如果有结果，保存并可视化
    if combined_results is not None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        combined_csv_path = os.path.join(args.output_dir, f"comparison_{timestamp}.csv")
        combined_results.to_csv(combined_csv_path, index=False)
        print(f"\n合并结果已保存到: {combined_csv_path}")
        
        # 打印汇总表格
        print("\n" + "=" * 60)
        print("性能对比汇总")
        print("=" * 60)
        print(combined_results.to_string(index=False))
        print("=" * 60)
        
        # 可视化比较结果
        visualize_comparison(combined_results, args.output_dir)
        print(f"\n比较可视化已保存到: {args.output_dir}")
    else:
        print("\n警告: 没有找到有效的评估结果")


if __name__ == "__main__":
    args = parse_args()
    run_comparison(args)
