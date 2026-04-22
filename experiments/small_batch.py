"""小批量实验：快速验证不同检测率与奖励函数组合的性能

本脚本设计用于快速验证性实验，相比完整训练大幅缩减了训练步数和评估时长，
适合在有限计算资源下快速探索不同配置的效果趋势。

实验设计：
- 检测率: [0.3, 0.5, 0.7, 0.9]
- 奖励函数: [average-speed, mixed]
- 随机种子: [42, 123, 456] (多seed取均值±标准差)
- 训练步数: 300,000 (完整训练为 6,000,000)
- 并行环境: 2 (完整训练为 8)
- 评估时长: 3600秒 (完整评估为 9000秒)
- 评估轮次: 5 (完整评估为 5)

总计 4 × 2 × 3 = 24 组实验

使用方法:
    python experiments/small_batch.py
    python experiments/small_batch.py --detection_rates "0.5,0.7" --reward_fns "mixed"
    python experiments/small_batch.py --seeds "42,123,456,789"
    python experiments/small_batch.py --total_timesteps 200000 --n_envs 4
"""

import os
import sys
import argparse
import numpy as np
import torch
import json
import time
from datetime import datetime
from itertools import product
from pathlib import Path

from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize, DummyVecEnv

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

REWARD_FUNCTIONS = {
    'average-speed': average_speed_reward,
    'mixed': mixed_reward,
}

EVALUATION_TIME_POINTS = [
    28800,   # 08:00 早高峰
    43200,   # 12:00 平峰
    61200,   # 17:00 晚高峰
]


def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def make_env(net_file, route_file, detection_rate, reward_fn, seed=42, env_index=0,
             sim_duration=3600, use_gui=False):
    if isinstance(reward_fn, str):
        reward_function = REWARD_FUNCTIONS[reward_fn]
    else:
        reward_function = reward_fn

    def _init():
        import random as python_random
        env_seed = seed + env_index
        np.random.seed(env_seed)
        torch.manual_seed(env_seed)
        python_random.seed(env_seed)

        rng = np.random.RandomState(env_seed + 1000)
        start_time = int(rng.randint(0, 86400 - sim_duration))

        env = SumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            use_gui=use_gui,
            begin_time=start_time,
            num_seconds=sim_duration,
            delta_time=5,
            yellow_time=3,
            min_green=5,
            max_green=50,
            enforce_max_green=True,
            single_agent=True,
            reward_fn=reward_function,
            observation_class=lambda ts: TableIObservationFunction(
                ts, detection_rate=detection_rate, seed=env_seed + 2000
            ),
            sumo_seed=env_seed,
        )
        return env
    return _init


def train_single_config(detection_rate, reward_fn_name, total_timesteps, n_envs,
                        net_file, route_file, output_dir, seed=42):
    experiment_id = f"dr{detection_rate}_{reward_fn_name}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\n{'='*60}")
    print(f"开始训练: detection_rate={detection_rate}, reward_fn={reward_fn_name}, seed={seed}")
    print(f"实验ID: {experiment_id}")
    print(f"训练步数: {total_timesteps}, 并行环境: {n_envs}")
    print(f"{'='*60}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env_fns = [
        make_env(net_file, route_file, detection_rate, reward_fn_name, seed=seed, env_index=i)
        for i in range(n_envs)
    ]

    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.99, epsilon=1e-8)

    policy_kwargs = dict(net_arch=[256, 256])

    model = DQN(
        env=env,
        policy="MlpPolicy",
        policy_kwargs=policy_kwargs,
        learning_rate=linear_schedule(1e-4),
        learning_starts=5000,
        train_freq=4 * n_envs,
        gradient_steps=-1,
        target_update_interval=2000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.3,
        buffer_size=100000,
        batch_size=256,
        gamma=0.99,
        tensorboard_log=f"{output_dir}/logs",
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=seed,
    )

    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, tb_log_name=experiment_id)
    train_duration = time.time() - start_time

    model_path = f"{output_dir}/models/dqn_table_i_dr{detection_rate}_{reward_fn_name}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model.save(model_path)
    env.save(f"{model_path}_vec_normalize.pkl")
    env.close()

    print(f"训练完成，耗时 {train_duration:.1f} 秒")
    print(f"模型已保存到: {model_path}")

    return model_path, train_duration


def evaluate_model(model_path, detection_rate, reward_fn_name, net_file, route_file,
                   eval_duration=3600, n_eval_episodes=5, seed=42):
    print(f"\n评估模型: detection_rate={detection_rate}, reward_fn={reward_fn_name}, seed={seed}")

    model = DQN.load(model_path)
    reward_function = REWARD_FUNCTIONS[reward_fn_name]
    norm_path = f"{model_path}_vec_normalize.pkl"

    all_metrics = {
        'rewards': [], 'waiting_times': [], 'queue_lengths': [],
        'speeds': [], 'throughputs': []
    }

    for time_idx, begin_time in enumerate(EVALUATION_TIME_POINTS):
        print(f"  时间点 {time_idx + 1}/{len(EVALUATION_TIME_POINTS)}: begin_time={begin_time} ({begin_time//3600:02d}:{(begin_time%3600)//60:02d})")
        
        for ep in range(n_eval_episodes):
            eval_env = SumoEnvironment(
                net_file=net_file,
                route_file=route_file,
                use_gui=False,
                begin_time=begin_time,
                num_seconds=eval_duration,
                delta_time=5,
                yellow_time=3,
                min_green=5,
                max_green=50,
                enforce_max_green=True,
                single_agent=True,
                reward_fn=reward_function,
                observation_class=lambda ts, t_idx=time_idx, e_idx=ep: TableIObservationFunction(
                    ts, detection_rate=detection_rate, seed=seed + 4000 + t_idx * 100 + e_idx
                ),
                sumo_seed=seed + time_idx * 100 + ep,
                add_system_info=True,
            )

            eval_env = DummyVecEnv([lambda: eval_env])
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                                    clip_obs=10.0, gamma=0.99, epsilon=1e-8, training=False)

            if os.path.exists(norm_path):
                eval_env = VecNormalize.load(norm_path, eval_env)
                eval_env.training = False
                eval_env.norm_reward = False

            obs = eval_env.reset()
            episode_reward = 0
            done = np.array([False])
            info = None

            while not done.any():
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward[0]

            all_metrics['rewards'].append(episode_reward)
            if info is not None and len(info) > 0:
                info_dict = info[0]
                all_metrics['waiting_times'].append(info_dict.get('system_mean_waiting_time', 0))
                all_metrics['queue_lengths'].append(info_dict.get('system_total_stopped', 0))
                all_metrics['speeds'].append(info_dict.get('system_mean_speed', 0))
                all_metrics['throughputs'].append(info_dict.get('system_total_departed', 0))

            eval_env.close()

    results = {}
    for metric, values in all_metrics.items():
        if values:
            results[f'mean_{metric[:-1]}'] = float(np.mean(values))
            results[f'std_{metric[:-1]}'] = float(np.std(values))

    print(f"  平均奖励: {results.get('mean_reward', 0):.4f}")
    print(f"  平均等待时间: {results.get('mean_waiting_time', 0):.2f}")
    print(f"  平均队列长度: {results.get('mean_queue_length', 0):.2f}")
    print(f"  平均速度: {results.get('mean_speed', 0):.4f}")

    return results


def evaluate_fixed_baseline(net_file, route_file, eval_duration=3600, n_eval_episodes=2, seed=42):
    print(f"\n评估固定信号控制基线 (seed={seed})...")

    all_metrics = {
        'rewards': [], 'waiting_times': [], 'queue_lengths': [],
        'speeds': [], 'throughputs': []
    }

    for time_idx, begin_time in enumerate(EVALUATION_TIME_POINTS):
        print(f"  时间点 {time_idx + 1}/{len(EVALUATION_TIME_POINTS)}: begin_time={begin_time} ({begin_time//3600:02d}:{(begin_time%3600)//60:02d})")
        
        for ep in range(n_eval_episodes):
            eval_env = SumoEnvironment(
                net_file=net_file,
                route_file=route_file,
                use_gui=False,
                begin_time=begin_time,
                num_seconds=eval_duration,
                delta_time=5,
                yellow_time=3,
                min_green=5,
                max_green=50,
                enforce_max_green=True,
                single_agent=True,
                reward_fn="average-speed",
                observation_class=lambda ts, t_idx=time_idx, e_idx=ep: TableIObservationFunction(
                    ts, detection_rate=1.0, seed=seed + 6000 + t_idx * 100 + e_idx
                ),
                sumo_seed=seed + time_idx * 100 + ep,
                add_system_info=True,
                fixed_ts=True,
            )

            obs, _ = eval_env.reset()
            episode_reward = 0
            done = False
            info = None

            while not done:
                obs, reward, terminated, truncated, info = eval_env.step(None)
                done = terminated or truncated
                episode_reward += reward

            all_metrics['rewards'].append(episode_reward)
            if info is not None:
                all_metrics['waiting_times'].append(info.get('system_mean_waiting_time', 0))
                all_metrics['queue_lengths'].append(info.get('system_total_stopped', 0))
                all_metrics['speeds'].append(info.get('system_mean_speed', 0))
                all_metrics['throughputs'].append(info.get('system_total_departed', 0))

            eval_env.close()

    results = {}
    for metric, values in all_metrics.items():
        if values:
            results[f'mean_{metric[:-1]}'] = float(np.mean(values))
            results[f'std_{metric[:-1]}'] = float(np.std(values))

    print(f"  固定基线 - 平均等待时间: {results.get('mean_waiting_time', 0):.2f}")
    print(f"  固定基线 - 平均速度: {results.get('mean_speed', 0):.4f}")

    return results


def aggregate_across_seeds(all_results, seeds):
    """将多seed的结果按 (detection_rate, reward_fn) 聚合，计算跨seed的均值和标准差"""
    from collections import defaultdict

    grouped = defaultdict(list)
    for r in all_results:
        key = (r['detection_rate'], r['reward_fn'])
        grouped[key].append(r)

    aggregated = []
    for (dr, rf), runs in grouped.items():
        metric_keys = list(runs[0]['eval_results'].keys()) if runs[0]['eval_results'] else []
        agg_eval = {}
        for mk in metric_keys:
            if mk.startswith('mean_'):
                values = [run['eval_results'][mk] for run in runs if mk in run['eval_results']]
                if values:
                    base = mk.replace('mean_', '')
                    agg_eval[f'mean_{base}'] = float(np.mean(values))
                    agg_eval[f'std_{base}'] = float(np.std(values))
                    agg_eval[f'min_{base}'] = float(np.min(values))
                    agg_eval[f'max_{base}'] = float(np.max(values))
                    agg_eval[f'n_seeds'] = len(values)

        total_train_duration = sum(run.get('train_duration', 0) for run in runs)
        model_paths = [run['model_path'] for run in runs]

        aggregated.append({
            'detection_rate': dr,
            'reward_fn': rf,
            'n_seeds': len(runs),
            'train_duration_total_sec': total_train_duration,
            'model_paths': model_paths,
            'eval_results': agg_eval,
            'per_seed_results': runs,
        })

    return aggregated


def generate_report(all_results, fixed_baselines, output_dir, seeds):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    per_seed_rows = []
    for r in all_results:
        row = {
            'detection_rate': r['detection_rate'],
            'reward_fn': r['reward_fn'],
            'seed': r['seed'],
            'train_duration_sec': r.get('train_duration', 0),
        }
        for k, v in r['eval_results'].items():
            row[k] = v
        per_seed_rows.append(row)

    df_per_seed = pd.DataFrame(per_seed_rows)

    csv_path = os.path.join(output_dir, f"small_batch_per_seed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df_per_seed.to_csv(csv_path, index=False)
    print(f"\n逐seed结果已保存到: {csv_path}")

    aggregated = aggregate_across_seeds(all_results, seeds)

    agg_rows = []
    for r in aggregated:
        row = {
            'detection_rate': r['detection_rate'],
            'reward_fn': r['reward_fn'],
            'n_seeds': r['n_seeds'],
        }
        for k, v in r['eval_results'].items():
            row[k] = v
        agg_rows.append(row)

    df_agg = pd.DataFrame(agg_rows)

    agg_csv_path = os.path.join(output_dir, f"small_batch_aggregated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df_agg.to_csv(agg_csv_path, index=False)
    print(f"聚合结果已保存到: {agg_csv_path}")

    fixed_baseline = None
    if fixed_baselines:
        fixed_baseline = {}
        for mk in fixed_baselines[0].keys():
            if mk.startswith('mean_'):
                values = [fb[mk] for fb in fixed_baselines if mk in fb]
                if values:
                    base = mk.replace('mean_', '')
                    fixed_baseline[f'mean_{base}'] = float(np.mean(values))
                    fixed_baseline[f'std_{base}'] = float(np.std(values))

    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 11, 'figure.figsize': (14, 5), 'savefig.dpi': 200})

    key_metrics = [
        ('mean_reward', '平均奖励'),
        ('mean_waiting_time', '平均等待时间 (秒)'),
        ('mean_queue_length', '平均队列长度'),
        ('mean_speed', '平均速度'),
    ]

    fig, axes = plt.subplots(1, len(key_metrics), figsize=(5 * len(key_metrics), 5))
    if len(key_metrics) == 1:
        axes = [axes]

    for ax, (metric, title) in zip(axes, key_metrics):
        if metric not in df_agg.columns:
            continue

        std_col = metric.replace('mean_', 'std_')
        err = df_agg[std_col] if std_col in df_agg.columns else None

        pivot_mean = df_agg.pivot(index='detection_rate', columns='reward_fn', values=metric)
        if err is not None:
            pivot_std = df_agg.pivot(index='detection_rate', columns='reward_fn', values=std_col)
        else:
            pivot_std = None

        x = np.arange(len(pivot_mean.index))
        width = 0.35
        n_bars = len(pivot_mean.columns)

        for j, col in enumerate(pivot_mean.columns):
            offset = (j - n_bars / 2 + 0.5) * width
            yerr = pivot_std[col].values if pivot_std is not None else None
            bars = ax.bar(x + offset, pivot_mean[col].values, width,
                          yerr=yerr, capsize=3, label=col, alpha=0.85)

        if fixed_baseline and metric in fixed_baseline:
            baseline_val = fixed_baseline[metric]
            baseline_std = fixed_baseline.get(metric.replace('mean_', 'std_'), 0)
            ax.axhline(y=baseline_val, color='red', linestyle='--', linewidth=1.5,
                       label=f'固定信号基线 ({baseline_val:.2f}±{baseline_std:.2f})')

        ax.set_title(title)
        ax.set_xlabel('检测率')
        ax.set_ylabel(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in pivot_mean.index])
        ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"small_batch_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"对比图已保存到: {plot_path}")

    if len(df_agg) >= 2:
        radar_metrics = ['mean_waiting_time', 'mean_queue_length', 'mean_speed']
        available = [m for m in radar_metrics if m in df_agg.columns]

        if len(available) >= 2:
            df_norm = df_agg.copy()
            for metric in available:
                max_val = df_agg[metric].max()
                min_val = df_agg[metric].min()
                if max_val > min_val:
                    if metric in ['mean_waiting_time', 'mean_queue_length']:
                        df_norm[metric] = 1 - (df_agg[metric] - min_val) / (max_val - min_val)
                    else:
                        df_norm[metric] = (df_agg[metric] - min_val) / (max_val - min_val)
                else:
                    df_norm[metric] = 1.0

            angles = np.linspace(0, 2 * np.pi, len(available), endpoint=False).tolist()
            angles += angles[:1]

            labels_map = {
                'mean_waiting_time': '等待时间',
                'mean_queue_length': '队列长度',
                'mean_speed': '速度',
            }
            categories = [labels_map.get(m, m) for m in available]
            categories += categories[:1]

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

            for _, row in df_norm.iterrows():
                label = f"DR={row['detection_rate']}, {row['reward_fn']}"
                values = [row[m] for m in available]
                values += values[:1]
                ax.plot(angles, values, linewidth=1.5, label=label)
                ax.fill(angles, values, alpha=0.05)

            plt.xticks(angles[:-1], categories[:-1])
            plt.ylim(0, 1)
            plt.title('小批量实验性能雷达图 (跨seed均值)', size=14, y=1.08)
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

            radar_path = os.path.join(output_dir, f"small_batch_radar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(radar_path, bbox_inches='tight')
            plt.close()
            print(f"雷达图已保存到: {radar_path}")

    if len(df_per_seed) >= 4:
        fig, axes = plt.subplots(1, len(key_metrics), figsize=(5 * len(key_metrics), 5))
        if len(key_metrics) == 1:
            axes = [axes]

        for ax, (metric, title) in zip(axes, key_metrics):
            if metric not in df_per_seed.columns:
                continue

            sns.boxplot(data=df_per_seed, x='detection_rate', y=metric,
                        hue='reward_fn', ax=ax, palette='Set2')
            sns.stripplot(data=df_per_seed, x='detection_rate', y=metric,
                          hue='reward_fn', ax=ax, dodge=True, color='black',
                          alpha=0.5, size=4)

            if fixed_baseline and metric in fixed_baseline:
                baseline_val = fixed_baseline[metric]
                ax.axhline(y=baseline_val, color='red', linestyle='--', linewidth=1.5,
                           label=f'固定信号基线')

            ax.set_title(f'{title} (跨seed分布)')
            ax.set_xlabel('检测率')
            handles, labels = ax.get_legend_handles_labels()
            n_unique = len(df_per_seed['reward_fn'].unique())
            ax.legend(handles[:n_unique], labels[:n_unique], fontsize=8)

        plt.tight_layout()
        box_path = os.path.join(output_dir, f"small_batch_boxplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(box_path)
        plt.close()
        print(f"箱线图已保存到: {box_path}")

    summary = {
        'experiment_type': 'small_batch_multi_seed',
        'timestamp': datetime.now().isoformat(),
        'seeds': seeds,
        'num_configs_per_seed': len(all_results) // len(seeds) if seeds else 0,
        'fixed_baseline': fixed_baseline,
        'aggregated_results': [{
            'detection_rate': r['detection_rate'],
            'reward_fn': r['reward_fn'],
            'n_seeds': r['n_seeds'],
            'eval_results': r['eval_results'],
        } for r in aggregated],
    }
    summary_path = os.path.join(output_dir, f"small_batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"实验摘要已保存到: {summary_path}")

    return df_agg, df_per_seed


def parse_args():
    parser = argparse.ArgumentParser(description="小批量实验：快速验证不同检测率与奖励函数组合")

    parser.add_argument("--detection_rates", type=str, default="0.3,0.5,0.7,0.9",
                        help="逗号分隔的检测率列表")
    parser.add_argument("--reward_fns", type=str, default="average-speed,mixed",
                        help="逗号分隔的奖励函数列表")
    parser.add_argument("--seeds", type=str, default="42,123,456",
                        help="逗号分隔的随机种子列表，多seed结果取均值±标准差")
    parser.add_argument("--total_timesteps", type=int, default=300_000,
                        help="每组实验的训练步数 (默认300K)")
    parser.add_argument("--n_envs", type=int, default=2,
                        help="并行环境数量 (默认2)")
    parser.add_argument("--eval_duration", type=int, default=3600,
                        help="评估模拟时长秒数 (默认3600)")
    parser.add_argument("--n_eval_episodes", type=int, default=5,
                        help="评估轮次 (默认5)")
    parser.add_argument("--net", type=str,
                        default="nets/2way-single-intersection/single-intersection.net.xml",
                        help="SUMO网络文件路径")
    parser.add_argument("--route", type=str,
                        default="nets/2way-single-intersection/single-intersection-poisson.rou.xml",
                        help="SUMO路由文件路径")
    parser.add_argument("--output_dir", type=str, default="experiments/results",
                        help="实验输出目录")
    parser.add_argument("--skip_training", action="store_true",
                        help="跳过训练，仅使用已有模型进行评估")
    parser.add_argument("--skip_eval", action="store_true",
                        help="跳过评估，仅进行训练")

    return parser.parse_args()


def main():
    args = parse_args()

    detection_rates = [float(x.strip()) for x in args.detection_rates.split(',')]
    reward_fns = [x.strip() for x in args.reward_fns.split(',')]
    seeds = [int(x.strip()) for x in args.seeds.split(',')]

    configs = list(product(detection_rates, reward_fns, seeds))

    print("=" * 60)
    print("小批量实验配置 (多seed)")
    print("=" * 60)
    print(f"检测率: {detection_rates}")
    print(f"奖励函数: {reward_fns}")
    print(f"随机种子: {seeds}")
    print(f"实验组合数: {len(detection_rates)} × {len(reward_fns)} × {len(seeds)} = {len(configs)}")
    print(f"每组训练步数: {args.total_timesteps:,}")
    print(f"并行环境数: {args.n_envs}")
    print(f"评估时长: {args.eval_duration} 秒")
    print(f"评估轮次: {args.n_eval_episodes}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/models", exist_ok=True)
    os.makedirs(f"{args.output_dir}/logs", exist_ok=True)

    all_results = []

    for i, (dr, rf, seed) in enumerate(configs):
        print(f"\n{'#'*60}")
        print(f"# 实验 {i+1}/{len(configs)}: detection_rate={dr}, reward_fn={rf}, seed={seed}")
        print(f"{'#'*60}")

        model_path = None
        train_duration = 0

        if not args.skip_training:
            model_path, train_duration = train_single_config(
                detection_rate=dr,
                reward_fn_name=rf,
                total_timesteps=args.total_timesteps,
                n_envs=args.n_envs,
                net_file=args.net,
                route_file=args.route,
                output_dir=args.output_dir,
                seed=seed,
            )
        else:
            import glob
            pattern = f"{args.output_dir}/models/dqn_table_i_dr{dr}_{rf}_seed{seed}_*.zip"
            matches = glob.glob(pattern)
            if matches:
                model_path = matches[0].replace('.zip', '')
                print(f"使用已有模型: {model_path}")
            else:
                print(f"未找到已有模型: {pattern}，跳过此配置")
                continue

        eval_results = {}
        if not args.skip_eval and model_path:
            eval_results = evaluate_model(
                model_path=model_path,
                detection_rate=dr,
                reward_fn_name=rf,
                net_file=args.net,
                route_file=args.route,
                eval_duration=args.eval_duration,
                n_eval_episodes=args.n_eval_episodes,
                seed=seed,
            )

        all_results.append({
            'detection_rate': dr,
            'reward_fn': rf,
            'seed': seed,
            'model_path': model_path,
            'train_duration': train_duration,
            'eval_results': eval_results,
        })

    fixed_baselines = []
    if not args.skip_eval:
        for seed in seeds:
            fb = evaluate_fixed_baseline(
                net_file=args.net,
                route_file=args.route,
                eval_duration=args.eval_duration,
                n_eval_episodes=args.n_eval_episodes,
                seed=seed,
            )
            fixed_baselines.append(fb)

    if all_results and not args.skip_eval:
        df_agg, df_per_seed = generate_report(all_results, fixed_baselines, args.output_dir, seeds)

        print("\n" + "=" * 60)
        print("小批量实验结果汇总 (跨seed聚合)")
        print("=" * 60)
        print(df_agg.to_string(index=False))

        if fixed_baselines:
            fixed_baseline = {}
            for mk in fixed_baselines[0].keys():
                if mk.startswith('mean_'):
                    values = [fb[mk] for fb in fixed_baselines if mk in fb]
                    if values:
                        base = mk.replace('mean_', '')
                        fixed_baseline[f'mean_{base}'] = float(np.mean(values))
                        fixed_baseline[f'std_{base}'] = float(np.std(values))

            print("\n固定信号控制基线 (跨seed均值):")
            for k, v in fixed_baseline.items():
                print(f"  {k}: {v:.4f}")

        print("\n" + "=" * 60)
        print("实验完成！")
        print(f"结果保存在: {args.output_dir}")
        print("=" * 60)
    else:
        print("\n训练阶段完成，模型已保存。")


if __name__ == "__main__":
    main()
