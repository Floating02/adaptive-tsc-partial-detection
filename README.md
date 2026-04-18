# Adaptive TSC Partial Detection

基于 SUMO (Simulation of Urban MObility) 仿真器与深度 Q 网络 (DQN) 的自适应交通信号控制系统。本项目在部分车辆检测（Partial Detection）场景下，利用强化学习算法优化单交叉口的信号配时策略，相比传统固定周期控制可显著降低车辆等待时间与队列长度。

## 目录

- [项目概述](#项目概述)
- [核心功能](#核心功能)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [使用指南](#使用指南)
  - [训练模型](#训练模型)
  - [超参数优化](#超参数优化)
  - [评估与对比](#评估与对比)
  - [不同检测率对比](#不同检测率对比)
- [配置说明](#配置说明)
  - [训练参数](#训练参数)
  - [优化参数](#优化参数)
  - [评估参数](#评估参数)
  - [环境变量](#环境变量)
- [自定义扩展](#自定义扩展)
- [常见问题解答](#常见问题解答)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## 项目概述

传统固定周期交通信号控制无法根据实时交通状况动态调整，而基于强化学习的方法能够根据实时交通流量自动优化信号配时。本项目的核心创新点在于引入**部分车辆检测**机制——模拟现实中传感器覆盖率有限的场景，使智能体在观测不完整的情况下仍能做出合理决策。

主要特点：

- **部分检测观测**：通过 `detection_rate` 参数控制车辆被检测到的概率，模拟真实世界中传感器不完美的场景
- **自定义状态表示**：基于 Table-I 的观测函数，包含检测车辆数、最近车辆距离、相位时间、黄灯指示等特征
- **多种奖励函数**：提供基于平均速度和混合奖励两种策略
- **并行训练**：支持多环境并行采样，加速训练过程
- **自动超参数优化**：集成 Optuna 框架进行贝叶斯超参数搜索
- **全面评估体系**：支持 RL 与固定信号控制的对比评估，以及不同检测率下的性能比较

## 核心功能

### 1. 部分检测观测函数 (`TableIObservationFunction`)

状态向量由以下归一化特征组成（所有值范围 0-1）：

| 特征 | 维度 | 说明 |
|------|------|------|
| 检测车辆数量 | N_lanes | 每条车道上被检测到的车辆数 / 最大容量 |
| 最近车辆距离 | N_lanes | 每条车道上最近被检测车辆的距离 / 车道长度 |
| 当前相位时间 | 1 | 当前相位已持续时间 / 最大相位时长 |
| 黄灯指示器 | 2 | One-hot 编码：[是黄灯, 非黄灯] |
| 当前相位 | N_phases | 当前绿灯相位的 One-hot 编码 |
| 当前时间 | 1 | 一天中的时刻归一化（小时数 / 24） |

其中 `N_lanes` 为车道数，`N_phases` 为绿灯相位数。检测率 `detection_rate` 控制每辆车被检测到的概率，使用伯努利试验决定。

### 2. 奖励函数

| 奖励函数 | 标识 | 公式 | 说明 |
|----------|------|------|------|
| `average_speed_reward` | `average-speed` | `avg_speed` | 基于归一化平均车速，鼓励车辆快速通行 |
| `mixed_reward` | `mixed` | `0.4 × speed - 0.3 × norm_queue - 0.3 × norm_pressure` | 综合考虑车速、队列长度和交通压力 |

混合奖励函数中：
- `speed`：归一化平均车速（0-1）
- `norm_queue`：归一化队列长度，即排队车辆数 / 车道最大容量
- `norm_pressure`：归一化负向交通压力，衡量流入与流出的不平衡程度

### 3. DQN 训练流程

基于 Stable Baselines3 的 DQN 实现，核心配置：

- **网络架构**：MLP `[256, 256]`
- **经验回放**：缓冲区大小 200,000，批大小 256
- **探索策略**：ε-贪婪，从 1.0 线性衰减至 0.05（前 30% 训练步数）
- **学习率调度**：线性衰减，初始值 1e-4
- **观测归一化**：VecNormalize 包装器，归一化观测但不归一化奖励
- **并行环境**：默认 8 个 SubprocVecEnv 并行采样

### 4. Optuna 超参数优化

搜索空间覆盖 DQN 的关键超参数：

| 参数 | 搜索范围 |
|------|----------|
| 学习率 | [1e-5, 1e-3]（对数均匀） |
| 缓冲区大小 | {10000, 50000, 100000, 200000} |
| 批大小 | {32, 64, 128, 256} |
| 网络架构 | Small [64,64] / Medium [128,128] / Large [256,256] |
| 探索初始 ε | [0.8, 1.0] |
| 探索最终 ε | [0.01, 0.1] |
| 探索比例 | [0.1, 0.5] |
| 目标网络更新间隔 | {1000, 5000, 10000} |
| 折扣因子 γ | [0.9, 0.9999] |

支持 TPE / CMA-ES / Random 采样器，以及 Median / Hyperband 剪枝策略。

### 5. 评估与对比

- **RL vs 固定信号**：对比 DQN 智能体与固定周期控制的性能差异
- **多检测率对比**：评估不同检测率（如 0.3, 0.5, 0.7, 0.9）下的模型表现
- **评估指标**：平均等待时间、平均队列长度、平均车速、平均通过量
- **可视化输出**：柱状图、性能提升百分比图、雷达图

## 项目结构

```
adaptive-tsc-partial-detection/
├── observations/
│   └── observation.py          # TableIObservationFunction 部分检测观测函数
├── models/
│   └── train.py                # DQN 模型训练入口
├── optimization/
│   └── optuna_optimizer.py     # Optuna 超参数优化
├── evaluation/
│   ├── compare_rl_vs_fixed.py  # RL 与固定信号控制对比评估
│   └── compare_detection_rates.py  # 不同检测率模型性能对比
├── rewards.py                  # 自定义奖励函数（average_speed / mixed）
├── requirements.txt            # Python 依赖
├── LICENSE                     # MIT 许可证
└── README.md
```

训练过程中自动创建的目录：

```
outputs/          # CSV 输出文件
models/           # 保存的模型文件 (.zip) 和归一化统计 (.pkl)
logs/             # 训练日志和 TensorBoard 数据
configs/          # 实验配置 JSON 文件
summaries/        # 训练总结 JSON 文件
plots/            # 训练曲线图
optimization/
  ├── results/    # Optuna 优化结果和可视化
  └── logs/       # Optuna TensorBoard 日志
evaluation/
  └── results/    # 评估结果和对比图表
```

## 环境要求

### 软件依赖

| 依赖 | 最低版本 | 说明 |
|------|----------|------|
| Python | >= 3.9 | 推荐 3.10+ |
| SUMO | >= 1.22.0 | 交通仿真器 |
| PyTorch | >= 2.8.0 | 深度学习框架 |
| Stable Baselines3 | >= 2.6.0 | 强化学习算法库 |
| Optuna | >= 3.6.1 | 超参数优化 |
| Gymnasium | >= 1.1.1 | RL 环境接口 |
| NumPy | >= 2.1.3 | 数值计算 |
| Pandas | >= 2.2.3 | 数据处理 |
| Matplotlib | >= 3.10.1 | 可视化 |
| Seaborn | >= 0.13.2 | 统计可视化 |

### 硬件建议

- **CPU**：4 核以上（并行环境数越多，需求越高）
- **内存**：16 GB 以上
- **GPU**：可选，支持 CUDA 的 NVIDIA GPU 可显著加速训练

## 安装步骤

### 1. 安装 SUMO 仿真器

**Windows：**

```powershell
# 下载并安装 SUMO：https://sumo.dlr.de/docs/Installing/index.html
# 安装后设置环境变量
[System.Environment]::SetEnvironmentVariable("SUMO_HOME", "C:\Program Files (x86)\Eclipse\Sumo", "User")
```

**Linux (Ubuntu/Debian)：**

```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc

# 设置环境变量
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
```

**验证安装：**

```bash
# 确认 SUMO_HOME 已设置
echo $SUMO_HOME   # Linux
echo %SUMO_HOME%  # Windows CMD
$env:SUMO_HOME    # Windows PowerShell
```

### 2. 克隆项目

```bash
git clone https://github.com/yourusername/adaptive-tsc-partial-detection.git
cd adaptive-tsc-partial-detection
```

### 3. 安装 Python 依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Linux:
source venv/bin/activate
# Windows:
.\venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt
```

### 4. 验证安装

```bash
python -c "import sumo_rl; print('SUMO-RL 安装成功')"
python -c "import stable_baselines3; print('Stable Baselines3 安装成功')"
```

## 使用指南

### 训练模型

**基本训练：**

```bash
python models/train.py
```

使用默认参数：检测率 0.2、混合奖励函数、8 个并行环境、600 万训练步数。

**自定义参数训练：**

```bash
python models/train.py --detection_rate 0.7 --reward_fn mixed --total_timesteps 1000000 --n_envs 4 --learning_rate 1e-4
```

**启用 SUMO GUI 可视化训练过程：**

```bash
python models/train.py --gui --n_envs 1
```

> **注意**：启用 GUI 时建议将 `n_envs` 设为 1，因为 GUI 模式不支持多进程并行。

**保存 CSV 输出：**

```bash
python models/train.py --detection_rate 0.5 --save_csv --experiment_name my_experiment
```

**使用 TensorBoard 监控训练：**

```bash
# 安装 TensorBoard（如尚未安装）
pip install tensorboard

# 启动 TensorBoard
tensorboard --logdir=./logs

# 浏览器访问 http://localhost:6006
```

训练完成后，模型文件和归一化统计将自动保存：

```
models/dqn_table_i_dr{detection_rate}_{reward_fn}_{experiment_id}.zip
models/dqn_table_i_dr{detection_rate}_{reward_fn}_{experiment_id}_vec_normalize.pkl
```

### 超参数优化

**基本优化：**

```bash
python optimization/optuna_optimizer.py
```

默认运行 20 次试验，每次训练 50,000 步，使用 TPE 采样器和 Median 剪枝器。

**自定义优化配置：**

```bash
# 通过环境变量配置
$env:OPTUNA_N_TRIALS = "50"              # Windows PowerShell
$env:OPTUNA_TIMEOUT = "7200"             # 超时时间（秒）
$env:OPTUNA_N_ENVS = "4"                 # 并行环境数
$env:OPTUNA_TIMESTEPS = "100000"         # 每次试验训练步数
$env:OPTUNA_DETECTION_RATE = "0.7"       # 检测率
$env:OPTUNA_SAMPLER = "tpe"              # 采样器: tpe / cmaes / random
$env:OPTUNA_PRUNER = "median"            # 剪枝器: median / hyperband / none
$env:OPTUNA_USE_PRUNING = "1"            # 是否启用剪枝

python optimization/optuna_optimizer.py
```

优化结果保存在 `optimization/results/` 目录下，包括：
- `best_params.txt`：最佳超参数和对应奖励
- `param_importances.png`：参数重要性图
- `optimization_history.png`：优化历史图
- `param_slices.png`：参数切片图
- `dqn_traffic_signal_optimization.db`：SQLite 数据库（支持断点续优）

### 评估与对比

**RL 模型 vs 固定信号控制：**

```bash
python evaluation/compare_rl_vs_fixed.py --model_path models/dqn_table_i_dr0.7_mixed_20260418_120000 --detection_rate 0.7 --reward_fn mixed --n_runs 5
```

**启用 GUI 观察评估过程：**

```bash
python evaluation/compare_rl_vs_fixed.py --model_path models/dqn_table_i_dr0.7_mixed_20260418_120000 --detection_rate 0.7 --gui
```

**自定义评估时长和输出目录：**

```bash
python evaluation/compare_rl_vs_fixed.py --model_path models/dqn_table_i_dr0.7_mixed_20260418_120000 --detection_rate 0.7 --eval_duration 3600 --output_dir evaluation/results/experiment1
```

评估输出包括：
- `comparison_metrics.png`：RL 与固定控制的指标对比柱状图
- `performance_improvement.png`：性能提升百分比图
- `comparison_results_{timestamp}.csv`：详细对比数据

### 不同检测率对比

**对比多个检测率的模型性能：**

```bash
python evaluation/compare_detection_rates.py --detection_rates "0.3,0.5,0.7,0.9" --n_runs 3
```

**自定义模型路径前缀和评估参数：**

```bash
python evaluation/compare_detection_rates.py \
  --model_prefix "models/dqn_table_i_dr" \
  --detection_rates "0.3,0.5,0.7,0.9" \
  --eval_duration 3600 \
  --n_runs 5 \
  --output_dir outputs/comparison
```

输出包括：
- `detection_rate_comparison_main.png`：主要指标对比图
- `detection_rate_comparison_reward.png`：奖励对比图
- `detection_rate_radar_chart.png`：雷达图
- `comparison_{timestamp}.csv`：汇总数据

## 配置说明

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--detection_rate` | 0.2 | 车辆被检测到的概率，范围 [0, 1] |
| `--reward_fn` | `mixed` | 奖励函数：`average-speed` 或 `mixed` |
| `--gui` | False | 是否启用 SUMO GUI |
| `--net` | `sumo_rl/nets/2way-single-intersection/single-intersection.net.xml` | SUMO 网络文件路径 |
| `--route` | `sumo_rl/nets/2way-single-intersection/single-intersection-poisson.rou.xml` | SUMO 路由文件路径 |
| `--total_timesteps` | 6,000,000 | 总训练步数 |
| `--n_envs` | 8 | 并行环境数量 |
| `--learning_rate` | 1e-4 | 初始学习率（线性衰减） |
| `--save_freq` | 500,000 | 模型保存频率（步数） |
| `--out_csv_name` | `outputs/dqn_table_i` | 输出 CSV 文件名前缀 |
| `--save_csv` | False | 是否保存详细 CSV 输出 |
| `--experiment_name` | None | 实验名称，用于组织输出文件 |

### 优化参数

通过环境变量配置：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `OPTUNA_N_TRIALS` | 20 | 优化试验次数 |
| `OPTUNA_TIMEOUT` | 21600 | 超时时间（秒），默认 6 小时 |
| `OPTUNA_N_ENVS` | 2 | 每次试验的并行环境数 |
| `OPTUNA_TIMESTEPS` | 50000 | 每次试验的训练步数 |
| `OPTUNA_EVAL_FREQ` | 2500 | 评估频率（步数） |
| `OPTUNA_DETECTION_RATE` | 0.7 | 检测率 |
| `OPTUNA_STUDY_NAME` | `dqn_traffic_signal_optimization` | Study 名称 |
| `OPTUNA_USE_PRUNING` | 1 | 是否启用剪枝（0/1） |
| `OPTUNA_SAMPLER` | `tpe` | 采样器：`tpe` / `cmaes` / `random` |
| `OPTUNA_PRUNER` | `median` | 剪枝器：`median` / `hyperband` / `none` |

### 评估参数

**compare_rl_vs_fixed.py：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | （必填） | 预训练模型路径 |
| `--detection_rate` | 0.7 | 检测率 |
| `--reward_fn` | `mixed` | 训练时使用的奖励函数 |
| `--gui` | False | 是否启用 SUMO GUI |
| `--n_runs` | 5 | 评估运行次数 |
| `--eval_duration` | 9000 | 每次评估的模拟时长（秒） |
| `--output_dir` | `evaluation/results` | 输出目录 |

**compare_detection_rates.py：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_prefix` | `models/dqn_table_i_dr` | 模型文件路径前缀 |
| `--detection_rates` | `0.3,0.5,0.7,0.9` | 逗号分隔的检测率列表 |
| `--n_runs` | 3 | 每个模型的评估次数 |
| `--eval_duration` | 3600 | 评估模拟时长（秒） |
| `--output_dir` | `outputs/comparison` | 输出目录 |

### 环境变量

| 变量 | 必需 | 说明 |
|------|------|------|
| `SUMO_HOME` | 是 | SUMO 安装目录路径 |
| `LIBSUMO_AS_TRACI` | 否 | 设为 `1` 启用 Libsumo 以提升性能（Windows 多进程下可能不兼容） |

## 自定义扩展

### 添加自定义奖励函数

1. 在 [rewards.py](rewards.py) 中定义新的奖励函数：

```python
def my_custom_reward(ts):
    speed = ts.get_average_speed()
    queue = ts.get_total_queued()
    return speed - 0.5 * queue
```

2. 在 `register_custom_rewards()` 中注册：

```python
def register_custom_rewards():
    for fn in [average_speed_reward, mixed_reward, my_custom_reward]:
        if fn.__name__ not in TrafficSignal.reward_fns:
            TrafficSignal.register_reward_fn(fn)
```

3. 在 [models/train.py](models/train.py) 的 `REWARD_FUNCTIONS` 字典中添加映射：

```python
REWARD_FUNCTIONS = {
    'average-speed': average_speed_reward,
    'mixed': mixed_reward,
    'my-custom': my_custom_reward,
}
```

### 添加自定义观测函数

1. 继承 `ObservationFunction` 基类，在 `observations/` 目录下创建新模块：

```python
from sumo_rl.environment.observations import ObservationFunction

class MyObservationFunction(ObservationFunction):
    def __init__(self, ts, **kwargs):
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        # 实现观测逻辑
        return observation_array

    def observation_space(self) -> spaces.Box:
        # 定义观测空间
        return spaces.Box(low=..., high=..., dtype=np.float32)
```

2. 在训练脚本中替换 `observation_class` 参数：

```python
observation_class=lambda ts: MyObservationFunction(ts, **kwargs)
```

### 使用不同的 RL 算法

项目基于 Stable Baselines3，可轻松替换为其他算法：

```python
from stable_baselines3 import PPO, SAC, A2C

model = PPO(
    env=env,
    policy="MlpPolicy",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    ...
)
```

## 常见问题解答

### Q: 运行时报错 `ImportError: 请声明环境变量 'SUMO_HOME'`

**A:** 需要设置 `SUMO_HOME` 环境变量指向 SUMO 的安装目录。

Windows PowerShell：
```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
```

Linux：
```bash
export SUMO_HOME="/usr/share/sumo"
```

如需永久生效，请将上述命令添加到 shell 配置文件中。

### Q: Windows 下多进程训练报错或卡死

**A:** 这通常是 Libsumo 与 `SubprocVecEnv` 的兼容性问题。尝试以下方案：

1. 注释掉代码中的 `os.environ["LIBSUMO_AS_TRACI"] = "1"` 行
2. 减少 `--n_envs` 参数（如设为 2 或 4）
3. 确保安装了 SUMO 的 TraCI 组件

### Q: CUDA out of memory 错误

**A:** 尝试以下方案：

1. 减少 `--n_envs` 并行环境数量
2. 减小 DQN 的 `buffer_size` 或 `batch_size`
3. 在代码中将 `device` 参数改为 `"cpu"`

### Q: 如何恢复中断的训练？

**A:** 目前训练脚本不直接支持断点续训。可以通过以下方式手动实现：

```python
model = DQN.load("models/your_model")
model.set_env(env)
model.learn(total_timesteps=remaining_steps)
```

### Q: 如何更换交通网络？

**A:** 使用 `--net` 和 `--route` 参数指定自定义的 SUMO 网络文件和路由文件：

```bash
python models/train.py --net path/to/your/network.net.xml --route path/to/your/routes.rou.xml
```

确保网络文件和路由文件格式符合 SUMO 规范。

### Q: Optuna 优化如何断点续优？

**A:** Optuna 使用 SQLite 数据库存储试验记录。再次运行相同 `OPTUNA_STUDY_NAME` 的优化脚本时，会自动加载已有记录并继续搜索：

```bash
$env:OPTUNA_STUDY_NAME = "my_study"
python optimization/optuna_optimizer.py
# 中断后再次运行相同命令即可继续
```

### Q: 评估时模型加载失败

**A:** 请确保：

1. 模型文件 `.zip` 存在于指定路径
2. 归一化统计文件 `_vec_normalize.pkl` 与模型文件在同一目录
3. `--detection_rate` 和 `--reward_fn` 参数与训练时一致
4. 模型路径不要包含 `.zip` 后缀（加载函数会自动添加）

## 贡献指南

欢迎对本项目做出贡献！请遵循以下流程：

1. **Fork** 本仓库
2. 创建功能分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m 'Add your feature'`
4. 推送分支：`git push origin feature/your-feature`
5. 创建 **Pull Request**

### 代码规范

- 遵循 PEP 8 Python 代码风格
- 新增功能需包含对应的文档字符串
- 保持与现有代码风格一致
- 提交前确保代码无语法错误

### 建议贡献方向

- 支持更多 RL 算法（PPO、SAC 等）
- 多交叉口协同控制（多智能体 RL）
- 更复杂的交通网络场景
- 新的观测函数或奖励函数设计
- 训练断点续训功能
- 性能优化和代码重构

## 许可证

本项目采用 [MIT 许可证](LICENSE) 开源。

```
MIT License

Copyright (c) 2026 Adaptive TSC Partial Detection

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
