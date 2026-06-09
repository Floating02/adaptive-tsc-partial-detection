# Adaptive TSC with Partial Detection

基于 [SUMO](https://eclipse.dev/sumo/) 交通仿真器与深度 Q 网络 (DQN) 的自适应交通信号控制系统。项目聚焦于**部分车辆检测 (Partial Detection)** 场景——模拟现实世界中传感器覆盖不完整的情况，使强化学习智能体在观测信息不完备的条件下仍能学习有效的信号控制策略。

## 目录

- [核心思想](#核心思想)
- [项目结构](#项目结构)
- [观测函数](#观测函数)
- [奖励函数](#奖励函数)
- [环境要求](#环境要求)
- [安装与配置](#安装与配置)
- [使用指南](#使用指南)
  - [完全检测实验](#完全检测实验)
  - [部分检测实验](#部分检测实验)
  - [固定配时基线](#固定配时基线)
- [交通网络](#交通网络)
- [扩展指南](#扩展指南)
- [许可证](#许可证)

## 核心思想

传统固定周期信号控制无法响应实时交通流变化。强化学习方法通过学习状态到动作的映射来自适应调整信号配时，但现有研究大多假设**完全可观测**——即所有车辆都能被完美检测。

本项目引入**部分检测机制**：通过 `detection_rate` 参数控制每辆车被传感器检测到的概率（伯努利试验），模拟真实场景中传感器覆盖率有限、数据丢失等不完美感知条件。主要研究问题：

- 不同检测率 (0.3, 0.5, 0.7, 0.9) 下 DQN 智能体的性能退化规律
- 部分检测与完全检测 (100%) 的性能差距
- 不同奖励函数对部分检测鲁棒性的影响

## 项目结构

```
adaptive-tsc-partial-detection/
├── observations/
│   ├── __init__.py            # 观测函数模块
│   └── observation.py         # PartialObservationFunction / FullDetectionObservationFunction
├── experiments/
│   ├── default_obs.py         # 完全检测训练与评估实验
│   ├── single_seed.py         # 部分检测多配置对比实验（检测率 × 奖励函数）
│   └── eval_fix_time.py       # 固定配时基线评估
├── nets/                      # SUMO 交通网络文件
│   ├── 2way-single-intersection/  # 主要使用的双向单交叉口
├── rewards.py                 # 自定义奖励函数
├── requirements.txt           # Python 依赖
├── LICENSE                    # MIT 许可证
└── README.md
```

## 观测函数

项目在 [observations/observation.py](observations/observation.py) 中提供两个观测类：

### PartialObservationFunction

模拟部分车辆检测场景。核心机制：

- **检测决策**：每辆新进入车道的车辆以概率 `detection_rate` 被标记为"可检测"（伯努利试验），使用独立的 `numpy.random.RandomState` 保证可复现性
- **车辆跟踪**：一旦某辆车被检测到，在它离开仿真前将持续被跟踪（模拟传感器锁定目标后的持续追踪）
- **已离开车辆清理**：每个仿真步通过集合差集操作高效移除已离开的车辆

### FullDetectionObservationFunction

与 `PartialObservationFunction` 共享相同特征结构，但始终使用所有车辆（100% 检测率），无随机性。用于作为完全可观测的上界基线。

### 状态向量

两个观测类的状态表示相同，所有特征均归一化：

| 特征 | 维度 | 范围 | 说明 |
|------|------|------|------|
| 带符号车辆数 | N_lanes | [-1, 1] | 检测到的车辆数 / 最大容量；绿灯车道为正，红灯为负 |
| 带符号最近车辆距离 | N_lanes | [-1, 1] | 最近检测车辆的距离 / 车道长度；绿灯为正，红灯为负 |
| 当前相位时间 | 1 | [0, 1] | 当前相位已持续时间 / 最大绿灯时长 |
| 黄灯指示器 | 1 | {0, 1} | 当前是否为黄灯相位 |
| 时间编码 | 2 | [-1, 1] | 一天中时刻的正弦/余弦编码，保证时间表示的连续性 |

其中 `N_lanes` = 8（4 条进口道 × 2 条车道）。总观测维度 = 8 + 8 + 1 + 1 + 2 = **20 维**。

> **带符号特征设计** (Signed Features)：绿灯车道特征为正 (+)，红灯车道特征为负 (−)，使智能体能够同时区分车道的拥堵程度和当前的通行权限。

## 奖励函数

### queue（主要训练奖励）

实验中的**默认训练奖励函数**，来自 sumo-rl 内置实现。

```
R = -total_queued
```

即负的总排队车辆数。智能体最小化排队长度等价于最大化此奖励。形式简单、信号明确，是所有部分检测实验的统一训练目标。

### 其他可选奖励

[default_obs.py](experiments/default_obs.py) 完全检测实验支持通过 `--reward_fn` 切换以下奖励函数：

| 标识 | 来源 | 公式 | 说明 |
|------|------|------|------|
| `queue` | sumo-rl 内置 | `-total_queued` | 默认值，负总排队车辆数 |
| `pressure` | sumo-rl 内置 | `-abs(pressure)` | 负交通压力绝对值，平衡流入与流出 |
| `average-speed` | [rewards.py](rewards.py) 自定义 | `avg_speed` | 归一化平均车速 (0-1)，鼓励快速通行 |
| `mixed` | [rewards.py](rewards.py) 自定义 | `0.4×speed - 0.3×norm_queue - 0.3×norm_pressure` | 综合车速、排队和压力三项指标 |

`mixed` 奖励中各项含义：
- **speed**：归一化平均车速 (0-1)
- **norm_queue**：归一化排队长度 = min(1, 总排队数 / 车道最大容量)
- **norm_pressure**：归一化负向交通压力（仅惩罚入流 > 出流的不平衡）

### 注册机制

[rewards.py](rewards.py) 在导入时自动调用 `register_custom_rewards()`，将 `average_speed_reward` 和 `mixed_reward` 注册到 `TrafficSignal.reward_fns` 字典中，使其可像内置奖励一样通过字符串名称引用。

## 环境要求

| 依赖 | 最低版本 | 用途 |
|------|----------|------|
| Python | ≥ 3.9 | 运行环境 |
| SUMO | ≥ 1.22.0 | 交通仿真引擎 |
| PyTorch | ≥ 2.8.0 | 深度学习框架 |
| Stable Baselines3 | ≥ 2.6.0 | DQN 算法实现 |
| Gymnasium | ≥ 1.1.0 | RL 环境接口 |
| NumPy | ≥ 2.1.0 | 数值计算 |
| Pandas | ≥ 2.2.0 | 数据处理 |
| Matplotlib | ≥ 3.10.0 | 可视化 |
| Seaborn | ≥ 0.13.0 | 统计图表 |
| Optuna | ≥ 3.6.0 | 超参数优化（可选） |


## 安装与配置

### 1. 安装 sumo-rl

**Windows：**

```bash
git clone https://github.com/LucasAlegre/sumo-rl
cd sumo-rl
pip install -e .
```

### 2. 安装 Python 依赖

```bash
git clone https://github.com/yourusername/adaptive-tsc-partial-detection.git
cd adaptive-tsc-partial-detection

# 安装依赖
pip install -r requirements.txt
```

## 使用指南

### 完全检测实验

使用 `default_obs.py` 进行完全检测 (100%) 场景下的训练与评估：

```bash
# 默认配置：完全检测 + queue 奖励 + 100K 步训练
python experiments/default_obs.py

# 指定奖励函数和训练步数
python experiments/default_obs.py --reward_fn mixed --total_timesteps 200000

# 跳过训练，仅评估已有模型
python experiments/default_obs.py --skip_training --model_path path/to/model

# 启用 SUMO GUI 观察评估过程
python experiments/default_obs.py --use_gui --skip_training --model_path path/to/model
```

**主要参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--reward_fn` | `queue` | 奖励函数：`queue`, `pressure`, `average-speed`, `mixed` |
| `--total_timesteps` | 100000 | 训练步数 |
| `--seed` | 42 | 随机种子 |
| `--net` | `nets/2way-single-intersection/single-intersection.net.xml` | 路网文件 |
| `--route` | `nets/2way-single-intersection/single-intersection_medium.rou.xml` | 训练车流 |
| `--eval_routes` | `medium,peak` | 逗号分隔的评估车流列表 |
| `--eval_duration` | 3600 | 评估模拟时长（秒） |
| `--n_eval_episodes` | 5 | 评估轮次 |
| `--output_dir` | `experiments/results_default_obs` | 输出目录 |
| `--skip_training` | False | 跳过训练 |
| `--skip_eval` | False | 跳过评估 |
| `--use_gui` | False | 启用 SUMO GUI |

### 部分检测实验

使用 `single_seed.py` 进行多检测率、多奖励函数的组合实验：

```bash
# 默认配置：4 个检测率 × 2 个奖励函数 = 8 组实验
python experiments/single_seed.py

# 自定义检测率和奖励函数
python experiments/single_seed.py \
  --detection_rates "0.3,0.5,0.7,0.9" \
  --reward_fns "average-speed,mixed" \
  --total_timesteps 200000

# 仅评估已有模型
python experiments/single_seed.py \
  --skip_training \
  --detection_rates "0.3,0.5" \
  --reward_fns "mixed"
```

**主要参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--detection_rates` | `0.3,0.5,0.7,0.9` | 逗号分隔的检测率列表 |
| `--reward_fns` | `average-speed,mixed` | 逗号分隔的奖励函数列表 |
| `--total_timesteps` | 100000 | 每组实验的训练步数 |
| `--seed` | 42 | 随机种子 |
| `--eval_routes` | `medium,peak` | 逗号分隔的评估车流列表 |
| `--eval_duration` | 3600 | 评估模拟时长（秒） |
| `--n_eval_episodes` | 5 | 评估轮次 |
| `--output_dir` | `experiments/results` | 输出目录 |

**输出内容：**
- `small_batch_single_seed_{timestamp}.csv` — 所有配置的实验数据
- `small_batch_single_comparison_{label}_{timestamp}.png` — 指标对比柱状图
- `small_batch_single_radar_{label}_{timestamp}.png` — 综合性能雷达图
- `small_batch_single_summary_{timestamp}.json` — 实验配置与结果汇总

### 固定配时基线

使用 `eval_fix_time.py` 运行固定配时仿真，作为 RL 方法的性能基线：

```bash
python experiments/eval_fix_time.py
```

脚本使用 TraCI 接口运行 SUMO，输出平均出行时间、平均等待时间和完成行程车辆数。

## 交通网络

项目包含 SUMO 路网文件，位于 `nets/` 目录：

| 路网 | 路径 | 说明 |
|------|------|------|
| 双向单交叉口 | `nets/2way-single-intersection/` | 主要实验场景，包含多种车流配置文件 |

### 车流配置文件

主要路网 `2way-single-intersection` 包含多种车流强度：

- `single-intersection_medium.rou.xml` — 中等流量（默认训练用）
- `single-intersection_peak.rou.xml` — 高峰流量（评估泛化用）

## DQN 配置

实验脚本使用以下默认 DQN 配置：

| 超参数 | 值 |
|--------|-----|
| 网络架构 | MLP [64, 64] |
| 学习率 | 1e-4 (default_obs) / 5e-5 (single_seed) |
| 经验回放缓冲区 | 10,000 |
| 批大小 | 32 |
| 折扣因子 γ | 0.95 |
| 探索初始 ε | 1.0 |
| 探索最终 ε | 0.05 |
| 探索衰减比例 | 0.5 (default_obs) / 0.3 (single_seed) |
| 目标网络更新间隔 | 2,000 |
| 学习开始步数 | 2,000 |
| 训练频率 | 4 |
| 观测归一化 | VecNormalize (norm_obs=True, norm_reward=True) |

## 扩展指南

### 添加自定义奖励函数

在 [rewards.py](rewards.py) 中定义新函数并注册：

```python
def my_custom_reward(ts):
    speed = ts.get_average_speed()
    queue = ts.get_total_queued()
    return speed - 0.5 * queue / max(1.0, len(ts.lanes))
```

然后在实验脚本的 `REWARD_FUNCTIONS` 字典中添加映射。

### 添加自定义观测函数

继承 `ObservationFunction` 基类，实现 `__call__()` 和 `observation_space()` 方法。参考 [observations/observation.py](observations/observation.py) 中的实现。

### 使用其他 RL 算法

项目基于 Stable Baselines3，可替换为 PPO、SAC 等算法：

```python
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64)
```

## 许可证

本项目基于 [MIT 许可证](LICENSE) 开源。
