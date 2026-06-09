"""
使用方法:
    python experiments/small_batch_default_obs.py
    python experiments/small_batch_default_obs.py --reward_fn mixed
    python experiments/small_batch_default_obs.py --total_timesteps 50000 --reward_fn average-speed
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
from pathlib import Path

from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# GUI模式使用TraCI连接SUMO GUI，非GUI模式使用libsumo加速仿真
if "--use_gui" not in sys.argv:
    os.environ["LIBSUMO_AS_TRACI"] = "1"

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("请声明环境变量'SUMO_HOME'")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sumo_rl import SumoEnvironment
from sumo_rl.environment.traffic_signal import TrafficSignal
from observations.observation import FullDetectionObservationFunction
from rewards import average_speed_reward, mixed_reward

REWARD_FUNCTIONS = {
    "queue": TrafficSignal.reward_fns["queue"],
    "pressure": TrafficSignal.reward_fns["pressure"],
    "average-speed": average_speed_reward,
    "mixed": mixed_reward,
}


def make_env(net_file, route_file, reward_fn="queue", seed=42, sim_duration=3600, use_gui=False):
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
        observation_class=FullDetectionObservationFunction,
        sumo_seed=seed,
    )
    return env


class DetailedTrainingCallback(BaseCallback):
    """训练过程详细日志回调：控制台逐episode输出 + TensorBoard指标记录"""

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self._ep_reward = 0.0
        self._ep_count = 0
        self._reward_history = deque(maxlen=10)
        self._last_loss = 0.0
        self._ep_metrics = []
        self._ep_losses = []
        self._prev_n_updates = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [0])
        reward = float(rewards[0]) if len(rewards) > 0 else 0.0

        dones = self.locals.get("dones", [False])
        done = bool(dones[0]) if len(dones) > 0 else False

        self._ep_reward += reward

        logger_vals = self.model.logger.name_to_value
        current_n_updates = getattr(self.model, "_n_updates", 0)
        if current_n_updates > self._prev_n_updates:
            self._last_loss = logger_vals.get("train/loss", self._last_loss)
            self._ep_losses.append(self._last_loss)
            self._prev_n_updates = current_n_updates

        infos = self.locals.get("infos", [{}])
        if infos and any(infos[0]):
            self._ep_metrics.append(infos[0])

        if done:
            self._ep_count += 1
            self._reward_history.append(self._ep_reward)

            episode_metrics = self._ep_metrics
            avg_wt, avg_ql, avg_speed, throughput = self._compute_averages(episode_metrics)

            ma_reward = float(np.mean(self._reward_history))
            epsilon = float(self.model.exploration_rate)

            rb = self.model.replay_buffer
            buf_cap = rb.buffer_size
            buf_pos = buf_cap if rb.full else rb.pos

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
                  f"Speed: {avg_speed:.1f}m/s | Through: {throughput:.2f} | BufSize: {buf_pos}/{buf_cap}")

            self.logger.record("episode/reward", self._ep_reward)
            self.logger.record("episode/reward_ma", ma_reward)
            self.logger.record("episode/loss", mean_loss)
            self.logger.record("episode/waiting_time", avg_wt)
            self.logger.dump(self.num_timesteps)

            self._ep_reward = 0.0
            self._ep_losses = []
            self._ep_metrics = []

        return True

    def _compute_averages(self, metrics):
        if not metrics:
            return 0.0, 0.0, 0.0, 0.0
        wt = np.mean([m.get("system_mean_waiting_time", 0) for m in metrics])
        ql = np.mean([m.get("system_total_stopped", 0) for m in metrics])
        sp = np.mean([m.get("system_mean_speed", 0) for m in metrics])
        th = metrics[-1].get("system_total_arrived", 0) if metrics else 0.0
        return float(wt), float(ql), float(sp), float(th)


def evaluate_model(model_path, net_file, route_file, reward_fn="queue",
                   eval_duration=3600, n_eval_episodes=5, seed=42, use_gui=False):
    print(f"\n评估模型 on {Path(route_file).stem}")

    model = DQN.load(model_path)
    reward_function = REWARD_FUNCTIONS[reward_fn] if isinstance(reward_fn, str) else reward_fn
    norm_path = f"{model_path}_vec_normalize.pkl"

    all_metrics = {
        'rewards': [], 'waiting_times': [], 'queue_lengths': [],
        'speeds': [], 'throughputs': []
    }

    for ep in range(n_eval_episodes):
        raw_env = SumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            use_gui=use_gui,
            begin_time=0,
            num_seconds=eval_duration,
            delta_time=5,
            yellow_time=3,
            min_green=5,
            max_green=50,
            enforce_max_green=True,
            single_agent=True,
            reward_fn=reward_function,
            observation_class=FullDetectionObservationFunction,
            sumo_seed=seed + ep,
            add_system_info=True,
            add_per_agent_info=False,
        )

        eval_env = DummyVecEnv([lambda: raw_env])
        eval_env = VecNormalize.load(norm_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

        obs = eval_env.reset()
        episode_reward = 0.0
        done = np.array([False])

        # 保存 metrics 列表引用，防止 DummyVecEnv 在 episode 结束时的
        # 自动 reset() 清空 raw_env.metrics（reset 会创建新列表，但原有
        # 列表对象通过此引用保留，不受影响）
        ep_metrics = raw_env.metrics

        while not done.any():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward[0]
        avg_wt = float(np.mean([m.get('system_mean_waiting_time', 0) for m in ep_metrics])) if ep_metrics else 0.0
        avg_ql = float(np.mean([m.get('system_total_stopped', 0) for m in ep_metrics])) if ep_metrics else 0.0
        avg_speed = float(np.mean([m.get('system_mean_speed', 0) for m in ep_metrics])) if ep_metrics else 0.0
        avg_throughput = float(ep_metrics[-1].get('system_total_arrived', 0)) if ep_metrics else 0.0

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
            name = metric[:-1] if metric != 'throughputs' else 'throughput'
            results[f'mean_{name}'] = float(np.mean(values))
            results[f'std_{name}'] = float(np.std(values))

    print(f"  --- 汇总: 平均奖励={results.get('mean_reward', 0):.1f}, "
          f"等待时间={results.get('mean_waiting_time', 0):.1f}s, "
          f"队列长度={results.get('mean_queue_length', 0):.1f}, "
          f"速度={results.get('mean_speed', 0):.1f}m/s")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="实验（完全检测观测+队列奖励）")

    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认42)")
    parser.add_argument("--total_timesteps", type=int, default=100_000,
                        help="训练步数 (默认100K)")
    parser.add_argument("--eval_duration", type=int, default=3600,
                        help="评估模拟时长秒数 (默认3600)")
    parser.add_argument("--n_eval_episodes", type=int, default=5,
                        help="评估轮次 (默认5)")
    parser.add_argument("--net", type=str,
                        default="nets/2way-single-intersection/single-intersection.net.xml",
                        help="SUMO网络文件路径")
    parser.add_argument("--route", type=str,
                        default="nets/2way-single-intersection/single-intersection_medium.rou.xml",
                        help="SUMO路由文件路径（训练用）")
    parser.add_argument("--eval_routes", type=str,
                        default="nets/2way-single-intersection/single-intersection_medium.rou.xml,nets/2way-single-intersection/single-intersection_peak.rou.xml",
                        help="逗号分隔的评估用SUMO路由文件路径列表")
    parser.add_argument("--output_dir", type=str, default="experiments/results_default_obs",
                        help="实验输出目录")
    parser.add_argument("--skip_training", action="store_true",
                        help="跳过训练，仅使用已有模型进行评估")
    parser.add_argument("--skip_eval", action="store_true",
                        help="跳过评估，仅进行训练")
    parser.add_argument("--reward_fn", type=str, default="queue",
                        choices=["queue", "pressure", "average-speed", "mixed"],
                        help="奖励函数 (默认queue)")
    parser.add_argument("--use_gui", action="store_true",
                        help="评估时启用SUMO GUI界面")
    parser.add_argument("--model_path", type=str,
                        help="指定已有模型的路径")

    return parser.parse_args()


def main():
    args = parse_args()

    eval_route_files = [x.strip() for x in args.eval_routes.split(',')]
    eval_route_labels = [Path(rf).stem for rf in eval_route_files]

    print("=" * 60)
    print("实验配置 (完全检测观测 + 可选奖励)")
    print("=" * 60)
    print(f"观测函数: FullDetectionObservationFunction")
    print(f"奖励函数: {args.reward_fn}")
    print(f"随机种子: {args.seed}")
    print(f"训练步数: {args.total_timesteps:,}")
    print(f"环境模式: 单环境")
    print(f"评估时长: {args.eval_duration} 秒")
    print(f"评估轮次: {args.n_eval_episodes}")
    print(f"训练车流文件: {args.route}")
    print(f"评估车流文件: {eval_route_files}")
    print(f"评估GUI: {'启用' if args.use_gui else '关闭'}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/models", exist_ok=True)
    os.makedirs(f"{args.output_dir}/logs", exist_ok=True)

    model_path = None
    train_duration = 0

    if not args.skip_training:
        experiment_id = f"full-detection-obs_{args.reward_fn}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"\n{'='*60}")
        print(f"开始训练: obs=FullDetection, reward={args.reward_fn}, seed={args.seed}")
        print(f"实验ID: {experiment_id}")
        print(f"{'='*60}")

        env = make_env(args.net, args.route, reward_fn=args.reward_fn, seed=args.seed)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            gamma=0.99,
            epsilon=1e-8,
        )

        policy_kwargs = dict(net_arch=[64, 64])

        model = DQN(
            env=env,
            policy="MlpPolicy",
            policy_kwargs=policy_kwargs,
            learning_rate=1e-4,
            learning_starts=2000,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=2000,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            exploration_fraction=0.5,
            buffer_size=10000,
            batch_size=32,
            gamma=0.95,
            tensorboard_log=f"{args.output_dir}/logs",
            verbose=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed=args.seed,
        )

        callback = DetailedTrainingCallback(verbose=1)

        start_time = time.time()
        model.learn(total_timesteps=args.total_timesteps, tb_log_name=experiment_id, callback=callback)
        train_duration = time.time() - start_time

        model_path = f"{args.output_dir}/models/dqn_full-detection-obs_{args.reward_fn}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model.save(model_path)
        norm_path = f"{model_path}_vec_normalize.pkl"
        env.save(norm_path)
        print(f"VecNormalize stats saved to: {norm_path}")
        env.close()

        print(f"训练完成，耗时 {train_duration:.1f} 秒")
        print(f"模型已保存到: {model_path}")
    else:
        model_path = args.model_path
        print(f"使用已有模型: {model_path}")


    if not args.skip_eval and model_path:
        all_eval_results = {}
        for eval_route, label in zip(eval_route_files, eval_route_labels):
            print(f"\n  评估场景: {label} ({eval_route})")
            route_results = evaluate_model(
                model_path=model_path,
                net_file=args.net,
                route_file=eval_route,
                reward_fn=args.reward_fn,
                eval_duration=args.eval_duration,
                n_eval_episodes=args.n_eval_episodes,
                seed=args.seed,
                use_gui=args.use_gui,
            )
            for k, v in route_results.items():
                all_eval_results[f"{label}_{k}"] = v

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        import pandas as pd
        row = {
            'seed': args.seed,
            'observation': 'FullDetectionObservationFunction',
            'reward': args.reward_fn,
            'train_duration_sec': train_duration,
        }
        row.update(all_eval_results)
        df = pd.DataFrame([row])
        csv_path = os.path.join(args.output_dir, f"small_batch_default_obs_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")

        summary = {
            'experiment_type': 'small_batch_full_detection_obs',
            'timestamp': datetime.now().isoformat(),
            'observation': 'FullDetectionObservationFunction',
            'reward': args.reward_fn,
            'seed': args.seed,
            'eval_scenarios': eval_route_labels,
            'eval_results': all_eval_results,
        }
        summary_path = os.path.join(args.output_dir, f"small_batch_default_obs_summary_{timestamp}.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Experiment summary saved to: {summary_path}")

        print("\n" + "=" * 60)
        print(f"实验结果汇总 (完全检测观测 + {args.reward_fn} 奖励)")
        print("=" * 60)
        print(df.to_string(index=False))
    else:
        print("\n训练阶段完成，模型已保存。")

    print("\n" + "=" * 60)
    print("实验完成！")
    print(f"结果保存在: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
