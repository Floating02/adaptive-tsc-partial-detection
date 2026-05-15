"""小批量实验（单seed版）：快速验证不同检测率与奖励函数组合的性能

相比 small_batch.py 的多seed取均值设计，本脚本每个配置仅运行单个seed，
进一步缩减实验时间，适合快速筛选有潜力的配置组合。

实验设计：
- 检测率: [0.3, 0.5, 0.7, 0.9]
- 奖励函数: [average-speed, mixed]
- 随机种子: 42 (单seed)
- 训练步数: 300,000
- 环境: 单环境（非并行）
- 评估时长: 3600秒
- 评估轮次: 5

总计 4 × 2 = 8 组实验

使用方法:
    python experiments/small_batch_single_seed.py
    python experiments/small_batch_single_seed.py --detection_rates "0.5,0.7" --reward_fns "mixed"
    python experiments/small_batch_single_seed.py --total_timesteps 200000
"""

import os
import sys
import argparse
import numpy as np
import torch
import json
import time
from collections import deque
from datetime import datetime
from itertools import product
from pathlib import Path

from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.callbacks import BaseCallback

os.environ["LIBSUMO_AS_TRACI"] = "1"

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("请声明环境变量'SUMO_HOME'")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sumo_rl import SumoEnvironment
from observations.observation import PartialObservationFunction
from rewards import average_speed_reward, mixed_reward

REWARD_FUNCTIONS = {
    'average-speed': average_speed_reward,
    'mixed': mixed_reward,
}

def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


class DetailedTrainingCallback(BaseCallback):
    """训练过程详细日志回调：控制台逐episode输出 + TensorBoard指标记录"""

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self._ep_reward = 0.0
        self._ep_count = 0
        self._reward_history = deque(maxlen=100)
        self._last_loss = 0.0
        self._last_grad_norm = 0.0
        self._last_q_mean = 0.0
        self._metrics_start_idx = 0
        self._ep_losses = []
        self._prev_n_updates = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [0])
        reward = float(rewards[0]) if len(rewards) > 0 else 0.0

        dones = self.locals.get("dones", [False])
        done = bool(dones[0]) if len(dones) > 0 else False

        self._ep_reward += reward

        logger_vals = getattr(self.model, "logger", type("", (), {"name_to_value": {}})()).name_to_value
        current_n_updates = getattr(self.model, "_n_updates", 0)
        if current_n_updates > self._prev_n_updates:
            self._last_loss = logger_vals.get("train/loss", self._last_loss)
            self._ep_losses.append(self._last_loss)
            self._prev_n_updates = current_n_updates
        self._last_grad_norm = logger_vals.get("train/grad_norm", self._last_grad_norm)

        if done:
            self._ep_count += 1
            self._reward_history.append(self._ep_reward)

            episode_metrics = self._get_episode_metrics()
            avg_wt, avg_ql, avg_speed, avg_throughput = self._compute_averages(episode_metrics)

            ma_reward = float(np.mean(self._reward_history))
            epsilon = float(self.model.exploration_rate)
            self._last_q_mean = self._compute_q_mean()

            buf_pos = 0
            buf_cap = 100000
            rb = getattr(self.model, "replay_buffer", None)
            if rb is not None:
                buf_pos = getattr(rb, "pos", 0)
                buf_cap = getattr(rb, "buffer_size", 100000)

            n_updates = len(self._ep_losses)
            if n_updates > 0:
                mean_loss = float(np.mean(self._ep_losses))
                loss_std = float(np.std(self._ep_losses))
            else:
                mean_loss = 0.0
                loss_std = 0.0

            print(f"[Ep {self._ep_count:03d}] R: {self._ep_reward:.1f} (MA: {ma_reward:.1f}) | "
                  f"Loss: {mean_loss:.4f} ± {loss_std:.4f} (n={n_updates} updates) | ε: {epsilon:.3f}")
            print(f"         WT: {avg_wt:.1f}s | QL: {avg_ql:.1f} | "
                  f"Speed: {avg_speed:.1f}m/s | Through: {avg_throughput:.2f}")
            print(f"         Q_mean: {self._last_q_mean:.1f} | GradNorm: {self._last_grad_norm:.2f} | "
                  f"BufSize: {buf_pos}/{buf_cap}")

            self.logger.record("episode/reward", self._ep_reward)
            self.logger.record("episode/reward_ma", ma_reward)
            self.logger.record("episode/loss", mean_loss)
            self.logger.record("episode/waiting_time", avg_wt)
            self.logger.record("episode/q_mean", self._last_q_mean)
            self.logger.dump(self.num_timesteps)

            self._ep_reward = 0.0
            self._ep_losses = []

        return True

    def _get_episode_metrics(self):
        """从 SumoEnvironment.metrics 中截取当前 episode 的所有 step 指标。"""
        try:
            env = self.model.env.envs[0]
            all_metrics = getattr(env, "metrics", [])
            episode_metrics = all_metrics[self._metrics_start_idx:]
            self._metrics_start_idx = len(all_metrics)
            return episode_metrics
        except Exception:
            return []

    def _compute_averages(self, metrics):
        if not metrics:
            return 0.0, 0.0, 0.0, 0.0
        wt = np.mean([m.get("system_mean_waiting_time", 0) for m in metrics])
        ql = np.mean([m.get("system_total_stopped", 0) for m in metrics])
        sp = np.mean([m.get("system_mean_speed", 0) for m in metrics])
        th = np.mean([m.get("system_total_departed", 0) for m in metrics])
        return float(wt), float(ql), float(sp), float(th)

    def _compute_q_mean(self) -> float:
        try:
            obs = self.locals.get("new_obs")
            if obs is None:
                return 0.0
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.model.device)
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            with torch.no_grad():
                q_values = self.model.q_net(obs_tensor)
                return float(q_values.mean().item())
        except Exception:
            return 0.0


def make_env(net_file, route_file, detection_rate, reward_fn, seed=42,
             sim_duration=3600, use_gui=False):
    if isinstance(reward_fn, str):
        reward_function = REWARD_FUNCTIONS[reward_fn]
    else:
        reward_function = reward_fn

    import random as python_random
    np.random.seed(seed)
    torch.manual_seed(seed)
    python_random.seed(seed)

    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=use_gui,
        begin_time=0,
        num_seconds=sim_duration,
        delta_time=5,
        yellow_time=3,
        min_green=5,
        max_green=50,
        enforce_max_green=True,
        single_agent=True,
        reward_fn=reward_function,
        observation_class=lambda ts: PartialObservationFunction(
            ts, detection_rate=detection_rate, seed=seed + 2000
        ),
        sumo_seed=seed,
    )
    return env


def train_single_config(detection_rate, reward_fn_name, total_timesteps,
                        net_file, route_file, output_dir, seed=42):
    experiment_id = f"dr{detection_rate}_{reward_fn_name}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\n{'='*60}")
    print(f"开始训练: detection_rate={detection_rate}, reward_fn={reward_fn_name}, seed={seed}")
    print(f"实验ID: {experiment_id}")
    print(f"训练步数: {total_timesteps}")
    print(f"{'='*60}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env = make_env(net_file, route_file, detection_rate, reward_fn_name, seed=seed)

    policy_kwargs = dict(net_arch=[256, 256])

    model = DQN(
        env=env,
        policy="MlpPolicy",
        policy_kwargs=policy_kwargs,
        learning_rate=linear_schedule(1e-4),
        learning_starts=5000,
        train_freq=4,
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

    callback = DetailedTrainingCallback(verbose=1)

    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, tb_log_name=experiment_id, callback=callback)
    train_duration = time.time() - start_time

    model_path = f"{output_dir}/models/dqn_table_i_dr{detection_rate}_{reward_fn_name}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model.save(model_path)
    env.close()

    print(f"训练完成，耗时 {train_duration:.1f} 秒")
    print(f"模型已保存到: {model_path}")

    return model_path, train_duration


def evaluate_model(model_path, detection_rate, reward_fn_name, net_file, route_file,
                   eval_duration=3600, n_eval_episodes=5, seed=42):
    print(f"\n评估模型: detection_rate={detection_rate}, reward_fn={reward_fn_name}, seed={seed}")

    model = DQN.load(model_path)
    reward_function = REWARD_FUNCTIONS[reward_fn_name]

    all_metrics = {
        'rewards': [], 'waiting_times': [], 'queue_lengths': [],
        'speeds': [], 'throughputs': []
    }

    for ep in range(n_eval_episodes):
        eval_env = SumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            use_gui=False,
            begin_time=0,
            num_seconds=eval_duration,
            delta_time=5,
            yellow_time=3,
            min_green=5,
            max_green=50,
            enforce_max_green=True,
            single_agent=True,
            reward_fn=reward_function,
            observation_class=lambda ts, e_idx=ep: PartialObservationFunction(
                ts, detection_rate=detection_rate, seed=seed + 4000 + e_idx
            ),
            sumo_seed=seed + ep,
            add_system_info=True,
            add_per_agent_info=False,
        )

        obs, _ = eval_env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward

        ep_metrics = eval_env.metrics
        avg_wt = float(np.mean([m.get('system_mean_waiting_time', 0) for m in ep_metrics])) if ep_metrics else 0.0
        avg_ql = float(np.mean([m.get('system_total_stopped', 0) for m in ep_metrics])) if ep_metrics else 0.0
        avg_speed = float(np.mean([m.get('system_mean_speed', 0) for m in ep_metrics])) if ep_metrics else 0.0
        avg_throughput = float(np.mean([m.get('system_total_departed', 0) for m in ep_metrics])) if ep_metrics else 0.0

        print(f"  [Eval Ep {ep+1}/{n_eval_episodes}] R: {episode_reward:.1f} | "
              f"WT: {avg_wt:.1f}s | QL: {avg_ql:.1f} | "
              f"Speed: {avg_speed:.1f}m/s | Through: {avg_throughput:.2f}")

        all_metrics['rewards'].append(episode_reward)
        all_metrics['waiting_times'].append(avg_wt)
        all_metrics['queue_lengths'].append(avg_ql)
        all_metrics['speeds'].append(avg_speed)
        all_metrics['throughputs'].append(avg_throughput)

        eval_env.close()

    results = {}
    for metric, values in all_metrics.items():
        if values:
            results[f'mean_{metric[:-1]}'] = float(np.mean(values))
            results[f'std_{metric[:-1]}'] = float(np.std(values))

    print(f"  --- 汇总: 平均奖励={results.get('mean_reward', 0):.1f}, "
          f"等待时间={results.get('mean_waiting_time', 0):.1f}s, "
          f"队列长度={results.get('mean_queue_length', 0):.1f}, "
          f"速度={results.get('mean_speed', 0):.1f}m/s")

    return results


def generate_report(all_results, output_dir, eval_route_labels, seed):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for r in all_results:
        row = {
            'detection_rate': r['detection_rate'],
            'reward_fn': r['reward_fn'],
            'train_duration_sec': r.get('train_duration', 0),
        }
        for k, v in r['eval_results'].items():
            row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    csv_path = os.path.join(output_dir, f"small_batch_single_seed_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n结果已保存到: {csv_path}")

    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 11, 'figure.figsize': (14, 5), 'savefig.dpi': 200})

    key_metrics = [
        ('mean_waiting_time', '平均等待时间 (秒)'),
        ('mean_queue_length', '平均队列长度'),
        ('mean_speed', '平均速度'),
    ]

    for label in eval_route_labels:
        label_prefix = f"{label}_"
        route_metric_cols = {f"{label_prefix}{m}": t for m, t in key_metrics}
        available_metrics = [(col, title) for col, title in route_metric_cols.items() if col in df.columns]

        if not available_metrics:
            continue

        fig, axes = plt.subplots(1, len(available_metrics), figsize=(5 * len(available_metrics), 5))
        if len(available_metrics) == 1:
            axes = [axes]

        for ax, (metric, title) in zip(axes, available_metrics):
            pivot = df.pivot(index='detection_rate', columns='reward_fn', values=metric)

            x = np.arange(len(pivot.index))
            width = 0.35
            n_bars = len(pivot.columns)

            for j, col in enumerate(pivot.columns):
                offset = (j - n_bars / 2 + 0.5) * width
                ax.bar(x + offset, pivot[col].values, width, label=col, alpha=0.85)

            ax.set_title(f'{title} - {label}')
            ax.set_xlabel('检测率')
            ax.set_ylabel(title)
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in pivot.index])
            ax.legend(fontsize=8)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"small_batch_single_comparison_{label}_{timestamp}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"对比图已保存到: {plot_path}")

        radar_metrics = [f'{label_prefix}mean_waiting_time', f'{label_prefix}mean_queue_length', f'{label_prefix}mean_speed']
        available_radar = [m for m in radar_metrics if m in df.columns]

        if len(available_radar) >= 2:
            df_norm = df.copy()
            for metric in available_radar:
                max_val = df[metric].max()
                min_val = df[metric].min()
                if max_val > min_val:
                    if 'waiting_time' in metric or 'queue_length' in metric:
                        df_norm[metric] = 1 - (df[metric] - min_val) / (max_val - min_val)
                    else:
                        df_norm[metric] = (df[metric] - min_val) / (max_val - min_val)
                else:
                    df_norm[metric] = 1.0

            angles = np.linspace(0, 2 * np.pi, len(available_radar), endpoint=False).tolist()
            angles += angles[:1]

            labels_map = {
                f'{label_prefix}mean_waiting_time': '等待时间',
                f'{label_prefix}mean_queue_length': '队列长度',
                f'{label_prefix}mean_speed': '速度',
            }
            categories = [labels_map.get(m, m) for m in available_radar]
            categories += categories[:1]

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

            for _, row in df_norm.iterrows():
                plot_label = f"DR={row['detection_rate']}, {row['reward_fn']}"
                values = [row[m] for m in available_radar]
                values += values[:1]
                ax.plot(angles, values, linewidth=1.5, label=plot_label)
                ax.fill(angles, values, alpha=0.05)

            plt.xticks(angles[:-1], categories[:-1])
            plt.ylim(0, 1)
            plt.title(f'性能雷达图 - {label}', size=14, y=1.08)
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

            radar_path = os.path.join(output_dir, f"small_batch_single_radar_{label}_{timestamp}.png")
            plt.savefig(radar_path, bbox_inches='tight')
            plt.close()
            print(f"雷达图已保存到: {radar_path}")

    summary = {
        'experiment_type': 'small_batch_single_seed_multi_scenario',
        'timestamp': datetime.now().isoformat(),
        'seed': seed,
        'eval_scenarios': eval_route_labels,
        'num_configs': len(all_results),
        'results': [{
            'detection_rate': r['detection_rate'],
            'reward_fn': r['reward_fn'],
            'eval_results': r['eval_results'],
        } for r in all_results],
    }
    summary_path = os.path.join(output_dir, f"small_batch_single_summary_{timestamp}.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"实验摘要已保存到: {summary_path}")

    return df


def parse_args():
    parser = argparse.ArgumentParser(description="小批量实验（单seed版）：快速验证不同检测率与奖励函数组合")

    parser.add_argument("--detection_rates", type=str, default="0.3,0.5,0.7,0.9",
                        help="逗号分隔的检测率列表")
    parser.add_argument("--reward_fns", type=str, default="average-speed,mixed",
                        help="逗号分隔的奖励函数列表")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认42)")
    parser.add_argument("--total_timesteps", type=int, default=300_000,
                        help="每组实验的训练步数 (默认300K)")
    parser.add_argument("--eval_duration", type=int, default=3600,
                        help="评估模拟时长秒数 (默认3600)")
    parser.add_argument("--n_eval_episodes", type=int, default=5,
                        help="评估轮次 (默认5)")
    parser.add_argument("--net", type=str,
                        default="nets/2way-single-intersection/single-intersection.net.xml",
                        help="SUMO网络文件路径")
    parser.add_argument("--route", type=str,
                        default="nets/2way-single-intersection/single-intersection-poisson.rou.xml",
                        help="SUMO路由文件路径（训练用）")
    parser.add_argument("--eval_routes", type=str,
                        default="nets/2way-single-intersection/single-intersection_medium.rou.xml,nets/2way-single-intersection/single-intersection_peak.rou.xml",
                        help="逗号分隔的评估用SUMO路由文件路径列表")
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

    eval_route_files = [x.strip() for x in args.eval_routes.split(',')]
    eval_route_labels = [Path(rf).stem for rf in eval_route_files]

    configs = list(product(detection_rates, reward_fns))

    print("=" * 60)
    print("小批量实验配置 (单seed, 多场景评估)")
    print("=" * 60)
    print(f"检测率: {detection_rates}")
    print(f"奖励函数: {reward_fns}")
    print(f"随机种子: {args.seed}")
    print(f"实验组合数: {len(detection_rates)} × {len(reward_fns)} = {len(configs)}")
    print(f"每组训练步数: {args.total_timesteps:,}")
    print(f"环境模式: 单环境")
    print(f"评估时长: {args.eval_duration} 秒")
    print(f"评估轮次: {args.n_eval_episodes}")
    print(f"训练车流文件: {args.route}")
    print(f"评估车流文件: {eval_route_files}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/models", exist_ok=True)
    os.makedirs(f"{args.output_dir}/logs", exist_ok=True)

    all_results = []

    for i, (dr, rf) in enumerate(configs):
        print(f"\n{'#'*60}")
        print(f"# 实验 {i+1}/{len(configs)}: detection_rate={dr}, reward_fn={rf}, seed={args.seed}")
        print(f"{'#'*60}")

        model_path = None
        train_duration = 0

        if not args.skip_training:
            model_path, train_duration = train_single_config(
                detection_rate=dr,
                reward_fn_name=rf,
                total_timesteps=args.total_timesteps,
                net_file=args.net,
                route_file=args.route,
                output_dir=args.output_dir,
                seed=args.seed,
            )
        else:
            import glob
            pattern = f"{args.output_dir}/models/dqn_table_i_dr{dr}_{rf}_seed{args.seed}_*.zip"
            matches = glob.glob(pattern)
            if matches:
                model_path = matches[0].replace('.zip', '')
                print(f"使用已有模型: {model_path}")
            else:
                print(f"未找到已有模型: {pattern}，跳过此配置")
                continue

        eval_results = {}
        if not args.skip_eval and model_path:
            for eval_route, label in zip(eval_route_files, eval_route_labels):
                print(f"\n  评估场景: {label} ({eval_route})")
                route_results = evaluate_model(
                    model_path=model_path,
                    detection_rate=dr,
                    reward_fn_name=rf,
                    net_file=args.net,
                    route_file=eval_route,
                    eval_duration=args.eval_duration,
                    n_eval_episodes=args.n_eval_episodes,
                    seed=args.seed,
                )
                for k, v in route_results.items():
                    eval_results[f"{label}_{k}"] = v

        all_results.append({
            'detection_rate': dr,
            'reward_fn': rf,
            'model_path': model_path,
            'train_duration': train_duration,
            'eval_results': eval_results,
        })

    if all_results and not args.skip_eval:
        df = generate_report(all_results, args.output_dir, eval_route_labels, args.seed)

        print("\n" + "=" * 60)
        print("小批量实验结果汇总 (单seed)")
        print("=" * 60)
        print(df.to_string(index=False))

        print("\n" + "=" * 60)
        print("实验完成！")
        print(f"结果保存在: {args.output_dir}")
        print("=" * 60)
    else:
        print("\n训练阶段完成，模型已保存。")


if __name__ == "__main__":
    main()
