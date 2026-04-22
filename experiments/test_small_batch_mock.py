"""Mock verification for small_batch.py

Quickly runs the entire small_batch pipeline with mocked SUMO dependencies
to verify that output file structure and content match expectations.

Usage:
    python experiments/test_small_batch_mock.py

Output:
    experiments/results_mock_test/  (can be deleted after verification)
"""

import sys
import os
import types
import glob
import json
import numpy as np
import gymnasium
from gymnasium import spaces

# ============================================================
# Step 1: Define mock classes
# ============================================================

class MockSumoEnvironment(gymnasium.Env):
    metadata = {'render_modes': []}

    def __init__(self, net_file=None, route_file=None, use_gui=False,
                 begin_time=0, num_seconds=3600, delta_time=5,
                 yellow_time=2, min_green=5, max_green=50,
                 enforce_max_green=False, single_agent=True,
                 reward_fn=None, observation_class=None,
                 sumo_seed=42, add_system_info=True, add_per_agent_info=True,
                 fixed_ts=False, out_csv_name=None,
                 max_depart_delay=-1, waiting_time_memory=1000,
                 time_to_teleport=-1, sumo_warnings=True,
                 additional_sumo_cmd=None, render_mode=None,
                 virtual_display=(3200, 1800), reward_weights=None,
                 **kwargs):
        super().__init__()
        self.single_agent = single_agent
        self.fixed_ts = fixed_ts
        self.add_system_info = add_system_info
        self.begin_time = begin_time
        self.num_seconds = num_seconds
        self.delta_time = delta_time
        self.sim_step = begin_time
        self._step_count = 0
        self._max_steps = max(1, num_seconds // delta_time)

        self._obs_dim = 30
        self._n_actions = 2

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self._obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self._n_actions)

        seed_val = sumo_seed if isinstance(sumo_seed, int) else 42
        self._rng = np.random.RandomState(seed_val)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self._step_count = 0
        self.sim_step = self.begin_time
        obs = self._rng.rand(self._obs_dim).astype(np.float32)
        info = self._make_info()
        if self.single_agent:
            return obs, info
        return {'tl': obs}

    def step(self, action):
        self._step_count += 1
        self.sim_step += self.delta_time

        obs = self._rng.rand(self._obs_dim).astype(np.float32)
        reward = float(self._rng.rand())
        terminated = False
        truncated = self._step_count >= self._max_steps
        info = self._make_info()

        if self.single_agent:
            return obs, reward, terminated, truncated, info
        return (
            {'tl': obs},
            {'tl': reward},
            {'tl': terminated, '__all__': terminated or truncated},
            info,
        )

    def _make_info(self):
        info = {'step': self.sim_step}
        if self.add_system_info:
            info.update({
                'system_total_running': int(self._rng.randint(10, 50)),
                'system_total_stopped': int(self._rng.randint(0, 20)),
                'system_mean_waiting_time': float(self._rng.rand() * 30),
                'system_mean_speed': float(self._rng.rand() * 15),
                'system_total_departed': int(self._rng.randint(50, 200)),
                'system_total_arrived': int(self._rng.randint(50, 200)),
            })
        return info

    def close(self):
        pass


class MockTableIObservationFunction:
    def __init__(self, ts, detection_rate=0.7, seed=None, **kwargs):
        self.ts = ts
        self.detection_rate = detection_rate
        self.rng = np.random.RandomState(seed)

    def __call__(self):
        return np.zeros(30, dtype=np.float32)

    def observation_space(self):
        return spaces.Box(low=0, high=1, shape=(30,), dtype=np.float32)

    def reset(self):
        pass


def mock_average_speed_reward(ts):
    return 0.5


def mock_mixed_reward(ts):
    return 0.3


# ============================================================
# Step 2: Inject mock modules into sys.modules BEFORE import
# ============================================================

os.environ.setdefault('SUMO_HOME', 'mock_sumo_home')
os.environ['LIBSUMO_AS_TRACI'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

mock_sumo_rl = types.ModuleType('sumo_rl')
mock_sumo_rl.SumoEnvironment = MockSumoEnvironment
sys.modules['sumo_rl'] = mock_sumo_rl

mock_observations_pkg = types.ModuleType('observations')
mock_observation_module = types.ModuleType('observations.observation')
mock_observation_module.TableIObservationFunction = MockTableIObservationFunction
sys.modules['observations'] = mock_observations_pkg
sys.modules['observations.observation'] = mock_observation_module

mock_rewards = types.ModuleType('rewards')
mock_rewards.average_speed_reward = mock_average_speed_reward
mock_rewards.mixed_reward = mock_mixed_reward
mock_rewards.register_custom_rewards = lambda: None
sys.modules['rewards'] = mock_rewards

# ============================================================
# Step 3: Import small_batch and apply patches
# ============================================================

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stable_baselines3.common.vec_env import DummyVecEnv
import experiments.small_batch as sb

sb.SubprocVecEnv = DummyVecEnv

# ============================================================
# Step 4: Run pipeline with minimal parameters
# ============================================================

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_mock_test')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'logs'), exist_ok=True)

DETECTION_RATE = 0.5
REWARD_FN = 'average-speed'
SEED = 42
TOTAL_TIMESTEPS = 6000
N_ENVS = 1
NET_FILE = "mock.net.xml"
ROUTE_FILE = "mock.rou.xml"
EVAL_DURATION = 100
N_EVAL_EPISODES = 1

print("=" * 60)
print("Mock Verification for small_batch.py")
print("=" * 60)

# --- 4a: Train ---
print("\n[1/4] Training...")
model_path, train_duration = sb.train_single_config(
    detection_rate=DETECTION_RATE,
    reward_fn_name=REWARD_FN,
    total_timesteps=TOTAL_TIMESTEPS,
    n_envs=N_ENVS,
    net_file=NET_FILE,
    route_file=ROUTE_FILE,
    output_dir=OUTPUT_DIR,
    seed=SEED,
)
print(f"  Model saved to: {model_path}")
print(f"  Training duration: {train_duration:.1f}s")

# --- 4b: Evaluate ---
print("\n[2/4] Evaluating model...")
eval_results = sb.evaluate_model(
    model_path=model_path,
    detection_rate=DETECTION_RATE,
    reward_fn_name=REWARD_FN,
    net_file=NET_FILE,
    route_file=ROUTE_FILE,
    eval_duration=EVAL_DURATION,
    n_eval_episodes=N_EVAL_EPISODES,
    seed=SEED,
)
print(f"  Eval results: {eval_results}")

# --- 4c: Fixed baseline ---
print("\n[3/4] Evaluating fixed baseline...")
fixed_baseline = sb.evaluate_fixed_baseline(
    net_file=NET_FILE,
    route_file=ROUTE_FILE,
    eval_duration=EVAL_DURATION,
    n_eval_episodes=N_EVAL_EPISODES,
    seed=SEED,
)
print(f"  Fixed baseline: {fixed_baseline}")

# --- 4d: Generate report ---
print("\n[4/4] Generating report...")
all_results = [{
    'detection_rate': DETECTION_RATE,
    'reward_fn': REWARD_FN,
    'seed': SEED,
    'model_path': model_path,
    'train_duration': train_duration,
    'eval_results': eval_results,
}]

df_agg, df_per_seed = sb.generate_report(
    all_results, [fixed_baseline], OUTPUT_DIR, [SEED]
)

# ============================================================
# Step 5: Verify output files and content
# ============================================================

print("\n" + "=" * 60)
print("Verification Results")
print("=" * 60)

errors = []


def check(condition, msg):
    if not condition:
        errors.append(msg)
        print(f"  FAIL: {msg}")
    else:
        print(f"  PASS: {msg}")


# --- 5a: Model files ---
model_files = [
    f for f in glob.glob(os.path.join(OUTPUT_DIR, 'models', 'dqn_table_i_dr*'))
    if not f.endswith('_vec_normalize.pkl')
]
check(len(model_files) >= 1, f"Model file exists (found {len(model_files)})")

norm_files = glob.glob(os.path.join(OUTPUT_DIR, 'models', '*_vec_normalize.pkl'))
check(len(norm_files) >= 1, f"VecNormalize .pkl file exists (found {len(norm_files)})")

# --- 5b: Log directory ---
log_dirs = glob.glob(os.path.join(OUTPUT_DIR, 'logs', 'dr*'))
check(len(log_dirs) >= 1, f"TensorBoard log directory exists (found {len(log_dirs)})")

# --- 5c: CSV files ---
per_seed_csvs = glob.glob(os.path.join(OUTPUT_DIR, 'small_batch_per_seed_*.csv'))
check(len(per_seed_csvs) >= 1, f"Per-seed CSV exists (found {len(per_seed_csvs)})")

if per_seed_csvs:
    import pandas as pd
    df = pd.read_csv(per_seed_csvs[0])
    for col in ['detection_rate', 'reward_fn', 'seed', 'train_duration_sec']:
        check(col in df.columns, f"Per-seed CSV has column '{col}'")
    check(len(df) == 1, f"Per-seed CSV has 1 row (has {len(df)})")
    if 'mean_reward' in df.columns:
        check(isinstance(df['mean_reward'].iloc[0], (int, float)),
              "Per-seed CSV mean_reward is numeric")

agg_csvs = glob.glob(os.path.join(OUTPUT_DIR, 'small_batch_aggregated_*.csv'))
check(len(agg_csvs) >= 1, f"Aggregated CSV exists (found {len(agg_csvs)})")

if agg_csvs:
    df = pd.read_csv(agg_csvs[0])
    for col in ['detection_rate', 'reward_fn', 'n_seeds']:
        check(col in df.columns, f"Aggregated CSV has column '{col}'")

# --- 5d: JSON summary ---
json_files = glob.glob(os.path.join(OUTPUT_DIR, 'small_batch_summary_*.json'))
check(len(json_files) >= 1, f"Summary JSON exists (found {len(json_files)})")

if json_files:
    with open(json_files[0], 'r', encoding='utf-8') as f:
        summary = json.load(f)

    check(summary.get('experiment_type') == 'small_batch_multi_seed',
          "JSON experiment_type == 'small_batch_multi_seed'")
    check('timestamp' in summary, "JSON has 'timestamp'")
    check('seeds' in summary, "JSON has 'seeds'")
    check(summary.get('seeds') == [SEED], f"JSON seeds == [{SEED}]")
    check('fixed_baseline' in summary, "JSON has 'fixed_baseline'")
    check('aggregated_results' in summary, "JSON has 'aggregated_results'")
    check(len(summary.get('aggregated_results', [])) >= 1,
          "JSON aggregated_results has >= 1 entry")

    if summary.get('aggregated_results'):
        agg = summary['aggregated_results'][0]
        check(agg.get('detection_rate') == DETECTION_RATE,
              f"Aggregated detection_rate == {DETECTION_RATE}")
        check(agg.get('reward_fn') == REWARD_FN,
              f"Aggregated reward_fn == '{REWARD_FN}'")
        check('eval_results' in agg, "Aggregated entry has 'eval_results'")

        er = agg.get('eval_results', {})
        for key in ['mean_reward', 'std_reward', 'mean_waiting_time',
                     'mean_queue_length', 'mean_speed']:
            check(key in er, f"eval_results has '{key}'")

# --- 5e: Plot files ---
comparison_plots = glob.glob(os.path.join(OUTPUT_DIR, 'small_batch_comparison_*.png'))
check(len(comparison_plots) >= 1, f"Comparison plot exists (found {len(comparison_plots)})")

radar_plots = glob.glob(os.path.join(OUTPUT_DIR, 'small_batch_radar_*.png'))
check(len(radar_plots) == 0,
      "Radar plot correctly skipped (need >=2 aggregated rows)")

boxplot_plots = glob.glob(os.path.join(OUTPUT_DIR, 'small_batch_boxplot_*.png'))
check(len(boxplot_plots) == 0,
      "Boxplot correctly skipped (need >=4 per-seed rows)")

# --- 5f: Eval results content ---
for key in ['mean_reward', 'std_reward', 'mean_waiting_time',
            'std_waiting_time', 'mean_queue_length', 'std_queue_length',
            'mean_speed', 'std_speed']:
    check(key in eval_results, f"eval_results has '{key}'")

check(isinstance(eval_results.get('mean_reward'), float),
      "mean_reward is float")
check(isinstance(eval_results.get('std_reward'), float),
      "std_reward is float")

# --- 5g: Fixed baseline content ---
for key in ['mean_waiting_time', 'std_waiting_time',
            'mean_speed', 'std_speed']:
    check(key in fixed_baseline, f"fixed_baseline has '{key}'")

# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
if errors:
    print(f"VERIFICATION FAILED: {len(errors)} error(s)")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
print("=" * 60)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("(You can delete this directory after inspection)")
