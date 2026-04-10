"""
使用Optuna优化DQN交通信号控制超参数

该脚本使用Optuna框架对DQN算法在SUMO交通仿真环境中的超参数进行优化。
主要优化的参数包括:
- 学习率
- 探索率相关参数
- 神经网络结构
- 批处理大小
- 更新频率

使用方法:
1. 确保已安装所需库: pip install optuna stable-baselines3
2. 运行脚本: python optimization/optuna_optimizer.py
3. 优化结果将保存在optimization/results目录下

Tensorboard可视化:
1. tensorboard --logdir=./optimization/logs
2. 浏览器访问: http://localhost:6006
"""

import os
import sys
import logging
import optuna
import numpy as np
import torch
from typing import Dict, Any
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.callbacks import EvalCallback

# 设置SUMO环境变量
os.environ["LIBSUMO_AS_TRACI"] = "1"

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("请声明环境变量'SUMO_HOME'")

# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from sumo_rl import SumoEnvironment
from sumo_rl_custom.observations.table_i_observation import TableIObservationFunction
from sumo_rl_custom.rewards import mixed_reward

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建结果目录
STUDY_DIR = "optimization/results"
LOGS_DIR = "optimization/logs"
os.makedirs(STUDY_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# 从环境变量中读取配置（如果存在）
def get_env_config():
    """从环境变量中获取配置参数"""
    config = {
        "n_trials": int(os.environ.get("OPTUNA_N_TRIALS", "20")),
        "timeout": int(os.environ.get("OPTUNA_TIMEOUT", str(6*3600))),  # 默认6小时
        "n_envs": int(os.environ.get("OPTUNA_N_ENVS", "2")),
        "timesteps": int(os.environ.get("OPTUNA_TIMESTEPS", "50000")),
        "eval_freq": int(os.environ.get("OPTUNA_EVAL_FREQ", "2500")),
        "detection_rate": float(os.environ.get("OPTUNA_DETECTION_RATE", "0.7")),
        "study_name": os.environ.get("OPTUNA_STUDY_NAME", "dqn_traffic_signal_optimization"),
        "use_pruning": os.environ.get("OPTUNA_USE_PRUNING", "1") == "1",
        "sampler": os.environ.get("OPTUNA_SAMPLER", "tpe"),
        "pruner": os.environ.get("OPTUNA_PRUNER", "median")
    }
    return config

def make_env(net_file, route_file, out_csv_name, detection_rate, seed=42, use_gui=False):
    """创建单个环境的工厂函数"""
    def _init():
        # 固定模拟持续时间为3000秒（为了快速评估，使用比原始脚本更短的时间）
        sim_duration = 3000
        # 使用种子初始化随机数生成器
        if seed is not None:
            np.random.seed(seed)
        # 开始时间固定为0
        start_time = 0
        
        env = SumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            out_csv_name=f"{out_csv_name}_dr{detection_rate}_seed{seed}",
            use_gui=use_gui,
            begin_time=start_time,
            num_seconds=sim_duration,
            delta_time=5,
            yellow_time=3,
            min_green=5,
            max_green=50,
            enforce_max_green=True,
            single_agent=True,
            reward_fn=mixed_reward,
            observation_class=lambda ts: TableIObservationFunction(ts, detection_rate=detection_rate),
            sumo_seed=seed,
        )
        return env
    return _init

class TrialEvalCallback(EvalCallback):
    """针对Optuna进行特殊定制的评估回调"""
    def __init__(
        self,
        eval_env,
        trial,
        n_eval_episodes=5,
        eval_freq=100,
        deterministic=True,
        verbose=0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
        
    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # 调用父类的评估逻辑
            result = super()._on_step()
            self.eval_idx += 1
            
            # 报告当前评估结果给Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            
            # 根据性能决定是否应该提前停止该试验
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True

def sample_dqn_params(trial: optuna.Trial) -> Dict[str, Any]:
    """从试验中采样DQN参数"""
    # 采样超参数
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    buffer_size = trial.suggest_categorical("buffer_size", [10000, 50000, 100000, 200000])
    learning_starts = trial.suggest_int("learning_starts", 1000, 20000)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    tau = trial.suggest_float("tau", 0.5, 1.0)  # 软更新系数，1.0为硬更新
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8])
    target_update_interval = trial.suggest_categorical("target_update_interval", [1000, 5000, 10000])
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.5)
    exploration_initial_eps = trial.suggest_float("exploration_initial_eps", 0.8, 1.0)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.1)
    
    # 采样网络架构参数
    net_arch_size = trial.suggest_categorical("net_arch_size", ["small", "medium", "large"])
    if net_arch_size == "small":
        net_arch = [64, 64]
    elif net_arch_size == "medium":
        net_arch = [128, 128]
    else:
        net_arch = [256, 256]
    
    return {
        "learning_rate": learning_rate,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "batch_size": batch_size,
        "tau": tau,
        "gamma": gamma,
        "train_freq": train_freq,
        "target_update_interval": target_update_interval,
        "exploration_fraction": exploration_fraction,
        "exploration_initial_eps": exploration_initial_eps,
        "exploration_final_eps": exploration_final_eps,
        "policy_kwargs": dict(net_arch=net_arch),
    }

def optimize_agent(trial, config):
    """Optuna优化目标函数"""
    # 全局随机种子以确保实验可复现
    global_seed = 42
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(global_seed)
        torch.backends.cudnn.deterministic = True
    
    # 设置模型参数
    model_params = sample_dqn_params(trial)
    
    # 设置环境参数
    detection_rate = config["detection_rate"]
    net_file = "sumo_rl/nets/2way-single-intersection/single-intersection.net.xml"
    route_file = "sumo_rl/nets/2way-single-intersection/single-intersection-poisson.rou.xml"
    
    # 创建并行环境
    n_envs = config["n_envs"]
    env_fns = []
    for i in range(n_envs):
        env_fn = make_env(
            net_file=net_file,
            route_file=route_file,
            out_csv_name=f"optuna_{trial.number}",
            detection_rate=detection_rate,
            seed=global_seed + i,
            use_gui=False,
        )
        env_fns.append(env_fn)
    
    # 创建向量化环境
    train_env = SubprocVecEnv(env_fns)
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=model_params["gamma"],
        epsilon=1e-8
    )
    
    # 创建评估环境 (单个环境)
    eval_env_fn = make_env(
        net_file=net_file,
        route_file=route_file,
        out_csv_name=f"optuna_{trial.number}_eval",
        detection_rate=detection_rate,
        seed=global_seed + 42,
        use_gui=False,
    )
    # 使用DummyVecEnv而不是直接调用环境函数
    eval_env = SubprocVecEnv([eval_env_fn])
    eval_env = VecMonitor(eval_env)
    # 确保评估环境也使用VecNormalize包装器，但设置training=False
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=model_params["gamma"],
        epsilon=1e-8,
        training=False  # 评估时不更新归一化统计数据
    )
    
    # 创建DQN模型
    model = DQN(
        env=train_env,
        policy="MlpPolicy",
        verbose=0,
        tensorboard_log=f"{LOGS_DIR}/trial_{trial.number}",
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=global_seed,
        **model_params
    )
    
    # 创建评估回调
    eval_callback = TrialEvalCallback(
        eval_env=eval_env,
        trial=trial,
        n_eval_episodes=5,
        eval_freq=config["eval_freq"],
        deterministic=True,
        verbose=0,
    )
    
    try:
        # 开始训练
        n_timesteps = config["timesteps"]
        model.learn(
            total_timesteps=n_timesteps,
            callback=eval_callback,
            tb_log_name=f"DQN_trial_{trial.number}"
        )
        
        # 如果模型训练被提前停止，返回最后的奖励
        if eval_callback.is_pruned:
            return eval_callback.last_mean_reward
        
        # 保存模型
        model.save(f"{STUDY_DIR}/dqn_trial_{trial.number}")
        
        # 如果训练完成后没有奖励记录，进行最后的评估
        if eval_callback.last_mean_reward is None:
            # 执行最后的评估
            mean_reward = 0
            for _ in range(5):  # 评估5个episode
                obs = eval_env.reset()
                done = False
                episode_reward = 0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = eval_env.step(action)
                    episode_reward += reward
                mean_reward += episode_reward / 5
            return mean_reward
        
        return eval_callback.best_mean_reward
        
    except Exception as e:
        # 记录异常
        logger.error(f"训练过程中发生错误: {e}")
        return float("-inf")
    
    finally:
        # 清理环境
        train_env.close()
        eval_env.close()

def get_sampler(sampler_name, seed=42):
    """获取指定的采样器"""
    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    elif sampler_name == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    else:  # 默认使用TPE
        return optuna.samplers.TPESampler(seed=seed)

def get_pruner(pruner_name):
    """获取指定的修剪器"""
    if pruner_name == "hyperband":
        return optuna.pruners.HyperbandPruner()
    elif pruner_name == "none":
        return optuna.pruners.NopPruner()
    else:  # 默认使用中值修剪
        return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)

def main():
    """主函数: 运行Optuna优化"""
    logger.info("开始Optuna超参数优化...")
    
    # 从环境变量获取配置
    config = get_env_config()
    logger.info(f"使用配置: {config}")
    
    # 创建或加载已有的study
    study_name = config["study_name"]
    storage_name = f"sqlite:///{STUDY_DIR}/{study_name}.db"
    
    try:
        # 选择采样器和修剪器
        sampler = get_sampler(config["sampler"])
        pruner = get_pruner(config["pruner"]) if config["use_pruning"] else optuna.pruners.NopPruner()
        
        # 创建或加载study
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            direction="maximize",  # 最大化奖励
            sampler=sampler,
            pruner=pruner
        )
        
        # 定义优化目标函数的包装器，传递配置参数
        def objective(trial):
            return optimize_agent(trial, config)
        
        # 运行优化
        study.optimize(
            objective, 
            n_trials=config["n_trials"], 
            timeout=config["timeout"]
        )
        
        # 打印结果
        logger.info("优化完成!")
        logger.info(f"最佳参数: {study.best_params}")
        logger.info(f"最佳奖励: {study.best_value}")
        
        # 保存结果
        with open(f"{STUDY_DIR}/best_params.txt", "w") as f:
            f.write(f"Best parameters: {study.best_params}\n")
            f.write(f"Best reward: {study.best_value}\n")
        
        # 可视化
        try:
            # 重要度可视化
            importance_fig = optuna.visualization.plot_param_importances(study)
            importance_fig.write_image(f"{STUDY_DIR}/param_importances.png")
            
            # 历史可视化
            history_fig = optuna.visualization.plot_optimization_history(study)
            history_fig.write_image(f"{STUDY_DIR}/optimization_history.png")
            
            # 边际可视化
            slice_fig = optuna.visualization.plot_slice(study)
            slice_fig.write_image(f"{STUDY_DIR}/param_slices.png")
            
            logger.info(f"可视化图表已保存到 {STUDY_DIR} 目录")
        except Exception as e:
            logger.warning(f"生成可视化图表时出错: {e}")
    
    except Exception as e:
        logger.error(f"优化过程中发生错误: {e}")

if __name__ == "__main__":
    main()
