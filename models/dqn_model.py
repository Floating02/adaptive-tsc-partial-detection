"""DQN model definition and training for traffic signal control."""

import os
import sys
import numpy as np
import torch
from stable_baselines3.dqn import DQN
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from sumo_rl import SumoEnvironment
from ..observations.table_i_observation import TableIObservationFunction
from ..rewards import speed_based_reward, mixed_reward


def create_env(net_file, route_file, use_gui, detection_rate, num_seconds):
    """Create environment for training or evaluation
    
    Args:
        net_file (str): Path to SUMO network file
        route_file (str): Path to SUMO route file
        use_gui (bool): Whether to use SUMO GUI
        detection_rate (float): Probability of vehicles being detected
        num_seconds (int): Duration of simulation (seconds)
        
    Returns:
        SumoEnvironment: Created SUMO environment
    """
    # Create environment
    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        out_csv_name=None,
        use_gui=use_gui,
        begin_time=0,
        num_seconds=num_seconds,
        delta_time=5,
        yellow_time=3,
        min_green=5,
        max_green=50,
        enforce_max_green=True,
        single_agent=True,
        reward_fn="average-speed",
        observation_class=lambda ts: TableIObservationFunction(ts, detection_rate=detection_rate),
        sumo_seed=42,
        add_system_info=True,
    )
    
    return env


def wrap_env(env):
    """Wrap environment for training
    
    Args:
        env: Original environment
        
    Returns:
        VecNormalize: Wrapped environment
    """
    # Wrap single environment with DummyVecEnv
    env = DummyVecEnv([lambda: env])
    
    # Add VecNormalize wrapper
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99,
        epsilon=1e-8,
        training=True
    )
    
    return env


def create_dqn_model(env, model_path):
    """Create DQN model
    
    Args:
        env: Environment to train on
        model_path (str): Path to save the model
        
    Returns:
        DQN: Created DQN model
    """
    # Create DQN model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=10000,
        batch_size=64,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10,
        tensorboard_log="./logs",
        verbose=1,
    )
    
    return model


def train_model(model, total_timesteps, model_path):
    """Train DQN model
    
    Args:
        model: DQN model to train
        total_timesteps (int): Total number of timesteps to train
        model_path (str): Path to save the model
    """
    # Train the model
    model.learn(total_timesteps=total_timesteps, log_interval=100)
    
    # Save the model
    model.save(model_path)
    
    # Save normalization statistics
    if hasattr(model, 'env') and hasattr(model.env, 'save'):
        model.env.save(f"{model_path}_vec_normalize.pkl")


def load_model(model_path, env):
    """Load pre-trained DQN model
    
    Args:
        model_path (str): Path to load the model from
        env: Environment to use with the model
        
    Returns:
        DQN: Loaded DQN model
    """
    # Load the model
    model = DQN.load(model_path, env=env)
    
    # Load normalization statistics
    norm_path = f"{model_path}_vec_normalize.pkl"
    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False
    
    return model, env
