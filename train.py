import argparse
import os
import numpy as np
import torch
import json
# Import multiple RL algorithms
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed
import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym
from tqdm import tqdm
import sys

from multiview_env import MultiViewEnv

# Add wrapper to encourage exploration with random actions
class RandomInitialActionsWrapper(gym.Wrapper):
    """Forces random actions for the first N steps with annealed exploration rate"""
    def __init__(self, env, num_random_steps=10000, exploration_fraction=0.3):
        super(RandomInitialActionsWrapper, self).__init__(env)
        self.num_random_steps = num_random_steps
        self.step_count = 0
        self.exploration_fraction = exploration_fraction
    
    def step(self, action):
        # Calculate exploration probability with annealing
        if self.step_count < self.num_random_steps:
            # Start with 100% random actions, decay more slowly
            exploration_rate = 1.0 - (self.step_count / self.num_random_steps) * 0.5  # Only decay to 50% by the end
            
            # Decide whether to take random action based on exploration rate
            if np.random.random() < exploration_rate:
                action = self.action_space.sample()
        
        # Even after the initial phase, still sometimes take random actions (10% of the time)
        elif np.random.random() < 0.1:
            action = self.action_space.sample()
        
        self.step_count += 1
        return self.env.step(action)
    
    def reset(self, **kwargs):
        self.step_count = 0  # Reset the step counter
        return self.env.reset(**kwargs)
    
class ForceViewSwitchingWrapper(gym.Wrapper):
    """Wrapper that forces the agent to switch views periodically"""
    def __init__(self, env, max_consecutive_same_view=3):
        super(ForceViewSwitchingWrapper, self).__init__(env)
        self.max_consecutive_same_view = max_consecutive_same_view
        self.consecutive_same_view = 0
        self.current_view = None
        
        # Access the max_views attribute by unwrapping until we find it
        self.unwrapped_env = env
        while not hasattr(self.unwrapped_env, 'max_views') and hasattr(self.unwrapped_env, 'env'):
            self.unwrapped_env = self.unwrapped_env.env
        
    def step(self, action):
        # Convert action to int if it's a numpy array
        if isinstance(action, np.ndarray):
            action = int(action.item())
            
        # Force view switching if stuck on same view too long
        if self.current_view is not None and action == self.current_view:
            self.consecutive_same_view += 1
            if self.consecutive_same_view >= self.max_consecutive_same_view:
                # Force a different view
                available_views = list(range(self.unwrapped_env.max_views))
                available_views.remove(action)
                action = np.random.choice(available_views)
                self.consecutive_same_view = 0
        else:
            self.consecutive_same_view = 0
        
        self.current_view = action
        return self.env.step(action)
        
    def reset(self, **kwargs):
        self.consecutive_same_view = 0
        self.current_view = None
        return self.env.reset(**kwargs)

# Metrics tracking callback to monitor training progress
class MetricsCallback(BaseCallback):
    """Custom callback for tracking training metrics"""
    def __init__(self, check_freq, log_dir, verbose=1):
        super(MetricsCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.metrics = {
            'timesteps': [],
            'rewards': [],
            'episode_lengths': [],
            'losses': [],
            'entropy': [],
            'policy_loss': [],
            'value_loss': [],
            'learning_rate': []
        }
        os.makedirs(log_dir, exist_ok=True)
        
    def _on_step(self):
        # Record data every check_freq steps
        if self.n_calls % self.check_freq == 0:
            # Save timestep
            self.metrics['timesteps'].append(self.num_timesteps)
            
            # Get rewards from monitor
            if len(self.model.ep_info_buffer) > 0:
                ep_rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
                ep_lengths = [ep_info["l"] for ep_info in self.model.ep_info_buffer]
                self.metrics['rewards'].append(np.mean(ep_rewards))
                self.metrics['episode_lengths'].append(np.mean(ep_lengths))
            else:
                self.metrics['rewards'].append(np.nan)
                self.metrics['episode_lengths'].append(np.nan)
            
            # Get training info from model logger - now we know the exact keys
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                # Map our metric names to the actual logger keys
                key_mapping = {
                    'losses': 'train/loss',
                    'entropy': 'train/entropy_loss',
                    'policy_loss': 'train/policy_gradient_loss',
                    'value_loss': 'train/value_loss',
                    'learning_rate': 'train/learning_rate'
                }
                
                # Get values for each metric using the correct keys
                for our_key, logger_key in key_mapping.items():
                    if logger_key in self.model.logger.name_to_value:
                        self.metrics[our_key].append(self.model.logger.name_to_value[logger_key])
                    else:
                        self.metrics[our_key].append(np.nan)
            else:
                # If logger not available, append NaN values
                for metric_name in ['losses', 'entropy', 'policy_loss', 'value_loss', 'learning_rate']:
                    self.metrics[metric_name].append(np.nan)
            
            # Save intermediate metrics to CSV
            df = pd.DataFrame(self.metrics)
            df.to_csv(os.path.join(self.log_dir, 'training_metrics.csv'), index=False)
            
            # Plot metrics - now we should have all the data
            if len(self.metrics['timesteps']) > 1:
                plt.figure(figsize=(12, 8))
                
                # Plot rewards
                plt.subplot(2, 2, 1)
                plt.plot(self.metrics['timesteps'], self.metrics['rewards'])
                plt.title('Mean Episode Reward')
                plt.xlabel('Timesteps')
                plt.ylabel('Reward')
                
                # Plot episode lengths
                plt.subplot(2, 2, 2)
                plt.plot(self.metrics['timesteps'], self.metrics['episode_lengths'])
                plt.title('Mean Episode Length')
                plt.xlabel('Timesteps')
                plt.ylabel('Steps')
                
                # Plot loss
                plt.subplot(2, 2, 3)
                plt.plot(self.metrics['timesteps'], self.metrics['losses'])
                plt.title('Training Loss')
                plt.xlabel('Timesteps')
                plt.ylabel('Loss')
                
                # Plot policy and value losses
                plt.subplot(2, 2, 4)
                plt.plot(self.metrics['timesteps'], self.metrics['policy_loss'], label='Policy Loss')
                plt.plot(self.metrics['timesteps'], self.metrics['value_loss'], label='Value Loss')
                plt.title('Policy and Value Losses')
                plt.xlabel('Timesteps')
                plt.ylabel('Loss')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.log_dir, 'training_curves.png'))
                plt.close()
            
        return True

# Enhanced callback to monitor and potentially fix view switching issues
class ViewSwitchingMonitorCallback(BaseCallback):
    """Monitors view selection during training and takes corrective action if needed"""
    def __init__(self, check_freq=1000, verbose=1, env=None):
        super(ViewSwitchingMonitorCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.view_counts = {}
        self.current_period_counts = {}
        self.consecutive_same_view = 0
        self.env = env
        self.intervention_history = []
        
    def _on_step(self):
        """Called at each step of training"""
        # Get current action from the model's buffer if available
        current_action = None
        if hasattr(self.model, 'rollout_buffer') and len(self.model.rollout_buffer.actions) > 0:
            action_idx = min(self.n_calls % len(self.model.rollout_buffer.actions), 
                            len(self.model.rollout_buffer.actions) - 1)
            current_action = self.model.rollout_buffer.actions[action_idx].item()
            
            # Track view counts
            self.view_counts[current_action] = self.view_counts.get(current_action, 0) + 1
            self.current_period_counts[current_action] = self.current_period_counts.get(current_action, 0) + 1
            
            # Track consecutive selections
            if hasattr(self, 'last_action') and current_action == self.last_action:
                self.consecutive_same_view += 1
            else:
                self.consecutive_same_view = 0
                
            self.last_action = current_action
        
        # Analyze view distributions periodically
        if self.n_calls % self.check_freq == 0:
            # Calculate view distribution
            total = sum(self.current_period_counts.values())
            if total > 0:
                distribution = {v: count/total for v, count in self.current_period_counts.items()}
                
                if self.verbose > 0:
                    print(f"\n=== View Selection Analysis at Step {self.n_calls} ===")
                    for view, freq in sorted(distribution.items()):
                        print(f"  View {view}: {freq*100:.1f}%")
                
                # Check for dominance by a single view
                if len(distribution) > 0:
                    most_frequent = max(distribution.items(), key=lambda x: x[1])
                    
                    # If one view is dominating (>80%), recommend action
                    if most_frequent[1] > 0.8:
                        message = f"WARNING: View {most_frequent[0]} is dominating ({most_frequent[1]*100:.1f}%)"
                        print(message)
                        print("RECOMMENDATION: Consider the following:")
                        print("  1. Reduce switch_penalty further (try 0.01)")
                        print("  2. Increase exploration_bonus_weight to 0.5")
                        print("  3. Increase switch_incentive_weight to 0.3")
                        
                        # If environment is available, we can modify parameters
                        if self.env is not None and hasattr(self.env, 'reward_params'):
                            # Record this intervention
                            intervention = {
                                'step': self.n_calls,
                                'dominant_view': most_frequent[0],
                                'dominance': most_frequent[1],
                                'old_params': self.env.reward_params.copy(),
                                'new_params': None
                            }
                            
                            # Only intervene if this is severely problematic (>90% dominance)
                            if most_frequent[1] > 0.9:
                                print("AUTOMATIC INTERVENTION: Adjusting reward parameters...")
                                
                                # Modify reward parameters to encourage view switching
                                self.env.reward_params['switch_penalty'] = 0.01
                                self.env.reward_params['exploration_bonus_weight'] = 0.5
                                self.env.reward_params['switch_incentive_weight'] = 0.3
                                
                                # Record new parameters
                                intervention['new_params'] = self.env.reward_params.copy()
                                self.intervention_history.append(intervention)
                                
                                print("Parameters adjusted:")
                                for k, v in self.env.reward_params.items():
                                    print(f"  {k}: {v}")
                
                # Reset period counts for next window
                self.current_period_counts = {}
                
        return True  # Continue training

# Callback for debugging view selection behavior
class ViewSwitchDebugCallback(BaseCallback):
    """Callback for debugging view selection behavior"""
    def __init__(self, check_freq=1000, verbose=0):
        super(ViewSwitchDebugCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.view_counts = {}
        self.rewards_by_view = {}
        self.view_transitions = {}
        self.episode_counter = 0
        self.step_in_episode = 0
        self.current_episode_views = []
        self.episode_records = []
        
    def _on_step(self):
        # Get current action from the rollout buffer
        if hasattr(self.model, 'rollout_buffer') and len(self.model.rollout_buffer.actions) > 0:
            action_idx = min(self.n_calls % len(self.model.rollout_buffer.actions), len(self.model.rollout_buffer.actions) - 1)
            action = self.model.rollout_buffer.actions[action_idx].item()
            
            # Count view selections
            self.view_counts[action] = self.view_counts.get(action, 0) + 1
            
            # Track rewards
            if hasattr(self.model.rollout_buffer, 'rewards') and len(self.model.rollout_buffer.rewards) > 0:
                reward_idx = min(self.n_calls % len(self.model.rollout_buffer.rewards), len(self.model.rollout_buffer.rewards) - 1)
                reward = self.model.rollout_buffer.rewards[reward_idx].item()
                
                if action not in self.rewards_by_view:
                    self.rewards_by_view[action] = []
                self.rewards_by_view[action].append(reward)
            
            # Track view transitions
            if hasattr(self, 'previous_action'):
                transition = (self.previous_action, action)
                self.view_transitions[transition] = self.view_transitions.get(transition, 0) + 1
            self.previous_action = action
            
            # Track episodes
            self.current_episode_views.append(action)
            self.step_in_episode += 1
            
            # Check if episode is done
            if hasattr(self.model.rollout_buffer, 'dones') and len(self.model.rollout_buffer.dones) > 0:
                done_idx = min(self.n_calls % len(self.model.rollout_buffer.dones), len(self.model.rollout_buffer.dones) - 1)
                if self.model.rollout_buffer.dones[done_idx].item():
                    self.episode_counter += 1
                    self.episode_records.append({
                        'episode': self.episode_counter,
                        'views': self.current_episode_views.copy(),
                        'switches': sum(1 for i in range(1, len(self.current_episode_views)) 
                                       if self.current_episode_views[i] != self.current_episode_views[i-1])
                    })
                    self.current_episode_views = []
                    self.step_in_episode = 0
        
        # Log detailed statistics periodically
        if self.n_calls % self.check_freq == 0:
            print(f"\n=== View Selection Debug Info (Step {self.n_calls}) ===")
            
            # View distribution
            total_views = sum(self.view_counts.values())
            if total_views > 0:
                print("\nView selection distribution:")
                for view, count in sorted(self.view_counts.items()):
                    percentage = count / total_views * 100
                    avg_reward = sum(self.rewards_by_view.get(view, [0])) / max(1, len(self.rewards_by_view.get(view, [0])))
                    print(f"  View {view}: {count} selections ({percentage:.1f}%), Avg reward: {avg_reward:.4f}")
            
            # View transitions
            if self.view_transitions:
                print("\nView transitions (most frequent):")
                sorted_transitions = sorted(self.view_transitions.items(), key=lambda x: x[1], reverse=True)
                for (prev, curr), count in sorted_transitions[:5]:
                    print(f"  {prev} -> {curr}: {count} times")
                
                # Check if there are any transitions where view changes
                has_transitions = False
                for (prev, curr), count in self.view_transitions.items():
                    if prev != curr and count > 0:
                        has_transitions = True
                        break
                
                if not has_transitions:
                    print("  WARNING: No view transitions detected! The model is not switching views.")
            
            # Recent episodes
            recent_episodes = self.episode_records[-5:] if self.episode_records else []
            if recent_episodes:
                print("\nRecent episodes:")
                for ep in recent_episodes:
                    switches = ep['switches']
                    total_steps = len(ep['views'])
                    
                    # Count occurrences of each view
                    view_counts = {}
                    for v in ep['views']:
                        view_counts[v] = view_counts.get(v, 0) + 1
                    
                    # Format view counts
                    view_dist = ", ".join([f"View {v}: {c}" for v, c in sorted(view_counts.items())])
                    
                    print(f"  Episode {ep['episode']}: {switches} switches in {total_steps} steps. {view_dist}")
            
            # Print reward inspection
            if self.rewards_by_view:
                print("\nReward inspection:")
                
                # Check if rewards are all very similar
                all_rewards = []
                for rewards in self.rewards_by_view.values():
                    all_rewards.extend(rewards)
                
                if all_rewards:
                    min_reward = min(all_rewards)
                    max_reward = max(all_rewards)
                    avg_reward = sum(all_rewards) / len(all_rewards)
                    
                    print(f"  Overall rewards - Min: {min_reward:.4f}, Max: {max_reward:.4f}, Avg: {avg_reward:.4f}")
                    print(f"  Reward range: {max_reward - min_reward:.4f}")
                    
                    if max_reward - min_reward < 0.1:
                        print("  WARNING: Very small reward range! The model may not be incentivized to switch views.")
            
            print("\nDebug Analysis:")
            self._print_debug_analysis()
            
            print("=" * 50)
            
        return True
    
    def _print_debug_analysis(self):
        """Print analysis of potential issues"""
        # Check for stuck in single view
        view_percentages = {}
        total_views = sum(self.view_counts.values())
        if total_views > 0:
            for view, count in self.view_counts.items():
                view_percentages[view] = count / total_views * 100
            
            max_view = max(view_percentages.items(), key=lambda x: x[1])
            if max_view[1] > 95:
                print(f"  ISSUE: Model is stuck on View {max_view[0]} ({max_view[1]:.1f}% of the time)")
                print("  SOLUTION: Try these fixes:")
                print("    1. Lower switch penalty further (try 0.05 or even 0)")
                print("    2. Increase informativeness weight to 2.0")
                print("    3. Add exploration noise to the policy during training")
                print("    4. Initialize with random actions for first 1000 steps")
        
        # Check for reward issues
        if self.rewards_by_view:
            all_rewards = []
            for rewards in self.rewards_by_view.values():
                all_rewards.extend(rewards)
            
            if all_rewards:
                reward_range = max(all_rewards) - min(all_rewards)
                if reward_range < 0.2:
                    print(f"  ISSUE: Very small reward range ({reward_range:.4f})")
                    print("  SOLUTION: Increase reward differentiation:")
                    print("    1. Add bonus rewards for switching at narrative boundaries")
                    print("    2. Increase the informativeness weight")

# Add function to train the model with different algorithms
def train_model(args):
    """Train the model with specified algorithm and parameters"""
    # Set random seed
    set_random_seed(args.seed)
    
    # Create output directory
    algo_output_dir = os.path.join(args.output_dir, args.algorithm.lower())
    os.makedirs(algo_output_dir, exist_ok=True)
    metrics_dir = os.path.join(algo_output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Create reward parameters
    reward_params = {
        "informativeness_weight": args.informativeness_weight,
        "narration_activity_weight": args.narration_activity_weight,
        "switch_penalty": args.switch_penalty,
        "visual_similarity_weight": args.visual_similarity_weight,
        "narrative_context_weight": args.narrative_context_weight,
        "exploration_bonus_weight": args.exploration_bonus_weight,
        "switch_incentive_weight": args.switch_incentive_weight
    }
    
    # Save the parameters used
    with open(os.path.join(algo_output_dir, "reward_params.json"), 'w') as f:
        json.dump(reward_params, f, indent=2)
    
    # Create environment config
    env_config = {
        "dataset_dir": args.dataset_dir,
        "feature_extractor": None,
        "feature_dir": args.feature_dir,
        "max_views": args.max_views,
        "feature_dim": args.feature_dim,
        "episode_length": 10,
        "reward_params": reward_params
    }
    
    # Create and wrap the environment
    env = MultiViewEnv(env_config)
    
    # Add debugging flag to environment
    env.debug = args.debug
    
    # Check if there are takes available
    if len(env.takes) == 0:
        print(f"No takes found with features in {args.feature_dir}. Exiting.")
        sys.exit(1)
    
    print(f"Found {len(env.takes)} takes with available features for training")
    
    # Monitor environment for tracking episode stats
    env = Monitor(env, os.path.join(algo_output_dir, "monitor"))

    # Add Force View Switching wrapper
    env = ForceViewSwitchingWrapper(env, max_consecutive_same_view=3)
    
    # Add random action wrapper for exploration
    env = RandomInitialActionsWrapper(
        env, 
        num_random_steps=args.random_steps, 
        exploration_fraction=args.exploration_fraction
    )
    
    # Vectorize environment for stable-baselines
    env = DummyVecEnv([lambda: env])
    
    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(algo_output_dir, "checkpoints"),
        name_prefix=f"multiview_{args.algorithm.lower()}",
        verbose=1
    )
    
    # Add metrics tracking callback
    metrics_callback = MetricsCallback(
        check_freq=1000,
        log_dir=metrics_dir,
        verbose=1
    )
    
    # Add debug callback
    debug_callback = ViewSwitchDebugCallback(
        check_freq=2000,
        verbose=1
    )
    
    # Add view switching monitor callback
    monitor_callback = ViewSwitchingMonitorCallback(
        check_freq=2000,
        verbose=1,
        env=env.envs[0]
    )
    
    # Set up callbacks list
    callbacks = [checkpoint_callback, metrics_callback, debug_callback, monitor_callback]
    
    # Create the model based on selected algorithm
    if args.algorithm.upper() == "PPO":
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            clip_range=args.clip_range,
            tensorboard_log=os.path.join(algo_output_dir, "tensorboard"),
        )
    elif args.algorithm.upper() == "A2C":
        model = A2C(
            "MultiInputPolicy",
            env,
            verbose=1,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            tensorboard_log=os.path.join(algo_output_dir, "tensorboard"),
        )
    elif args.algorithm.upper() == "DQN":
        model = DQN(
            "MultiInputPolicy",
            env,
            verbose=1,
            learning_rate=args.learning_rate,
            buffer_size=10000,
            learning_starts=1000,
            batch_size=args.batch_size,
            gamma=args.gamma,
            exploration_fraction=0.3,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            tensorboard_log=os.path.join(algo_output_dir, "tensorboard"),
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    
    print(f"Starting training with {args.algorithm} and the following parameters:")
    print(f"  Reward parameters: {reward_params}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Gamma: {args.gamma}")
    
    # Train the model
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save the final model
    model_save_path = os.path.join(algo_output_dir, f"multiview_{args.algorithm.lower()}_final")
    model.save(model_save_path)
    
    print(f"Training complete. Model saved to {model_save_path}")
    
    return model, env_config, algo_output_dir

# Add function to train the model with improved parameters
def train_improved_model(args):
    """Train the model with enhanced parameters for better view switching"""
    # Set random seed
    set_random_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_dir = os.path.join(args.output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Create improved reward parameters
    reward_params = {
        "informativeness_weight": args.informativeness_weight,
        "narration_activity_weight": args.narration_activity_weight,
        "switch_penalty": args.switch_penalty,
        "visual_similarity_weight": args.visual_similarity_weight,
        "narrative_context_weight": args.narrative_context_weight,
        "exploration_bonus_weight": args.exploration_bonus_weight,
        "switch_incentive_weight": args.switch_incentive_weight
    }
    
    # Save the parameters used
    with open(os.path.join(args.output_dir, "reward_params.json"), 'w') as f:
        json.dump(reward_params, f, indent=2)
    
    # Create environment config
    env_config = {
        "dataset_dir": args.dataset_dir,
        "feature_extractor": None,  # We don't need feature extractor anymore
        "feature_dir": args.feature_dir,
        "max_views": args.max_views,
        "feature_dim": args.feature_dim,
        "episode_length": 10,
        "reward_params": reward_params
    }
    
    # Create and wrap the environment
    env = MultiViewEnv(env_config)
    
    # Add debugging flag to environment
    env.debug = args.debug
    
    # Check if there are takes available
    if len(env.takes) == 0:
        print(f"No takes found with features in {args.feature_dir}. Exiting.")
        sys.exit(1)
    
    print(f"Found {len(env.takes)} takes with available features for training")
    
    # Monitor environment for tracking episode stats
    env = Monitor(env, os.path.join(args.output_dir, "monitor"))

    env = ForceViewSwitchingWrapper(env, max_consecutive_same_view=3)  # Force switch after 3 steps
    
    # Add random action wrapper for exploration
    env = RandomInitialActionsWrapper(
        env, 
        num_random_steps=args.random_steps, 
        exploration_fraction=args.exploration_fraction
    )
    
    # Vectorize environment for stable-baselines
    env = DummyVecEnv([lambda: env])
    
    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(args.output_dir, "checkpoints"),
        name_prefix="multiview_ppo",
        verbose=1
    )
    
    # Add metrics tracking callback
    metrics_callback = MetricsCallback(
        check_freq=1000,
        log_dir=metrics_dir,
        verbose=1
    )
    
    # Add our debug callback
    debug_callback = ViewSwitchDebugCallback(
        check_freq=2000,
        verbose=1
    )
    
    # Add view switching monitor callback with environment access
    monitor_callback = ViewSwitchingMonitorCallback(
        check_freq=2000,
        verbose=1,
        env=env.envs[0]  # Pass unwrapped env for parameter modification
    )
    
    # Set up callbacks list
    callbacks = [checkpoint_callback, metrics_callback, debug_callback, monitor_callback]
    
    # Create the model with improved hyperparameters
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        ent_coef=args.ent_coef,  # Higher entropy coefficient for more exploration
        clip_range=args.clip_range,
        tensorboard_log=os.path.join(args.output_dir, "tensorboard"),
    )
    
    print("Starting training with the following parameters:")
    print(f"  Reward parameters: {reward_params}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Entropy coefficient: {args.ent_coef}")
    print(f"  Random steps: {args.random_steps}")
    print(f"  Exploration fraction: {args.exploration_fraction}")
    
    # Train the model
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save the final model
    model.save(os.path.join(args.output_dir, "multiview_ppo_final"))
    
    print(f"Training complete. Model saved to {os.path.join(args.output_dir, 'multiview_ppo_final')}")
    
    return model, env_config

def detailed_evaluation(model, env_config, output_dir, num_episodes=50):
    """
    Perform a detailed evaluation of the model and save comprehensive metrics
    """
    # Create raw environment for evaluation
    eval_env = MultiViewEnv(env_config)
    
    # Initialize result tracking
    all_results = {}
    all_episodes = []
    
    # Aggregate metrics
    total_clips = 0
    correct_views = 0
    total_reward = 0
    total_switches = 0
    camera_counts = {i: 0 for i in range(env_config["max_views"])}
    
    # Track per-take metrics
    take_accuracies = {}
    
    # Evaluate over multiple episodes
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        obs, _ = eval_env.reset()
        take_dir = eval_env.takes[eval_env.current_take_idx]["take_dir"]
        
        episode_data = {
            "take_dir": take_dir,
            "selected_views": [],
            "rewards": [],
            "best_views": [],
            "clip_indices": [],
            "clip_narrations": []
        }
        
        done = False
        episode_reward = 0
        prev_view = None
        switches = 0
        
        clip_counter = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            selected_view = int(action.item())
            
            # Count camera selection
            camera_counts[selected_view] += 1
            
            # Count view switches
            if prev_view is not None and selected_view != prev_view:
                switches += 1
            prev_view = selected_view
            
            # Get the best view from narrations and narration text
            clip_idx = eval_env.current_clip_idx
            clip_str = str(clip_idx)
            take = eval_env.takes[eval_env.current_take_idx]
            clip_narration_text = []
            
            # Get best view
            best_view = -1
            if "clip_narrations" in take["narrations"] and clip_str in take["narrations"]["clip_narrations"]:
                clip_narrations = take["narrations"]["clip_narrations"][clip_str]
                view_names = eval_env._get_view_names(take["take_dir"])
                
                # Save narration text
                for narration in clip_narrations:
                    if "text" in narration:
                        clip_narration_text.append(narration["text"])
                
                # Find the most common best_camera in narrations
                camera_counts_local = {}
                for narration in clip_narrations:
                    best_camera = narration.get("best_camera", "")
                    if best_camera:
                        camera_counts_local[best_camera] = camera_counts_local.get(best_camera, 0) + 1
                
                if camera_counts_local:
                    best_camera = max(camera_counts_local.items(), key=lambda x: x[1])[0]
                    # Find the action index for this camera
                    for action_idx, camera in view_names.items():
                        if camera == best_camera:
                            best_view = action_idx
                            break
            
            # Take step
            next_obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            
            # Record results
            episode_data["selected_views"].append(selected_view)
            episode_data["rewards"].append(float(reward))
            episode_data["best_views"].append(int(best_view))
            episode_data["clip_indices"].append(int(clip_idx))
            episode_data["clip_narrations"].append(clip_narration_text)
            
            # Update counters
            if best_view >= 0:  # Only count clips with a defined best view
                total_clips += 1
                if selected_view == best_view:
                    correct_views += 1
            
            episode_reward += reward
            obs = next_obs
            clip_counter += 1
        
        # Calculate episode-level metrics
        episode_data["total_reward"] = float(episode_reward)
        episode_data["num_switches"] = switches
        episode_data["switch_rate"] = switches / max(1, clip_counter - 1)
        
        # Save take results
        all_episodes.append(episode_data)
        all_results[f"{take_dir}_{episode}"] = episode_data
        
        # Update take-specific accuracy
        if take_dir not in take_accuracies:
            take_accuracies[take_dir] = {"correct": 0, "total": 0}
        
        take_correct = sum(1 for s, b in zip(episode_data["selected_views"], episode_data["best_views"]) 
                          if b >= 0 and s == b)
        take_total = sum(1 for b in episode_data["best_views"] if b >= 0)
        
        take_accuracies[take_dir]["correct"] += take_correct
        take_accuracies[take_dir]["total"] += take_total
        
        total_switches += switches
        total_reward += episode_reward
    
    # Calculate overall metrics
    accuracy = correct_views / max(1, total_clips)
    avg_reward = total_reward / num_episodes
    avg_switches_per_episode = total_switches / num_episodes
    
    # Calculate take-specific accuracies
    take_specific_accuracies = {}
    for take, counts in take_accuracies.items():
        if counts["total"] > 0:
            take_specific_accuracies[take] = counts["correct"] / counts["total"]
        else:
            take_specific_accuracies[take] = 0
    
    # Calculate camera distribution
    total_actions = sum(camera_counts.values())
    camera_distribution = {cam: count/total_actions for cam, count in camera_counts.items()}
    
    # Compile metrics
    metrics = {
        "accuracy": float(accuracy),
        "correct_views": int(correct_views),
        "total_clips": int(total_clips),
        "average_reward": float(avg_reward),
        "average_switches_per_episode": float(avg_switches_per_episode),
        "camera_distribution": camera_distribution,
        "take_specific_accuracies": take_specific_accuracies
    }
    
    # Create evaluation directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    results_path = os.path.join(output_dir, "detailed_evaluation.json")
    with open(results_path, 'w') as f:
        json.dump({
            "metrics": metrics,
            "results": all_results
        }, f, indent=2)
    
    # Print summary
    print(f"Evaluation results saved to {results_path}")
    print(f"Accuracy: {accuracy:.4f} ({correct_views}/{total_clips})")
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Average switches per episode: {avg_switches_per_episode:.2f}")
    print("\nCamera distribution:")
    for cam, percentage in camera_distribution.items():
        print(f"  Camera {cam}: {percentage:.2%}")
    
    print("\nTake-specific accuracies:")
    for take, acc in sorted(take_specific_accuracies.items(), key=lambda x: x[1], reverse=True):
        if take_accuracies[take]["total"] > 0:
            print(f"  {take}: {acc:.4f} ({take_accuracies[take]['correct']}/{take_accuracies[take]['total']})")
    
    return metrics

def create_evaluation_plots(metrics, all_episodes, output_dir):
    """Create various plots to visualize the evaluation results"""
    # Create a plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Camera distribution pie chart
    plt.figure(figsize=(10, 8))
    labels = [f"Camera {i}" for i in range(len(metrics["camera_distribution"]))]
    sizes = list(metrics["camera_distribution"].values())
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Camera Selection Distribution')
    plt.savefig(os.path.join(plots_dir, 'camera_distribution.png'))
    plt.close()
    
    # 2. Take-specific accuracies bar chart
    plt.figure(figsize=(12, 6))
    takes = list(metrics["take_specific_accuracies"].keys())
    accuracies = list(metrics["take_specific_accuracies"].values())
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)[::-1]
    sorted_takes = [takes[i] for i in sorted_indices]
    sorted_accuracies = [accuracies[i] for i in sorted_indices]
    
    plt.bar(range(len(sorted_takes)), sorted_accuracies)
    plt.xlabel('Take')
    plt.ylabel('Accuracy')
    plt.title('View Selection Accuracy by Take')
    plt.xticks(range(len(sorted_takes)), sorted_takes, rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'take_accuracies.png'))
    plt.close()
    
    # 3. Reward distribution histogram
    all_rewards = [metrics["average_reward"] for _ in range(len(metrics["take_specific_accuracies"]))]
    plt.figure(figsize=(10, 6))
    plt.hist(all_rewards, bins=20)
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.title('Distribution of Episode Rewards')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'reward_distribution.png'))
    plt.close()
    
    # 4. Switch rate vs reward scatter plot
    switch_rate = metrics["average_switches_per_episode"] / 10  # Approximate
    plt.figure(figsize=(10, 6))
    plt.scatter([switch_rate], [metrics["average_reward"]])
    plt.xlabel('View Switch Rate')
    plt.ylabel('Episode Reward')
    plt.title('Relationship Between View Switching and Reward')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'switch_vs_reward.png'))
    plt.close()

def main():
    """Main function with enhanced argument parsing"""
    parser = argparse.ArgumentParser(description="Train a multi-view selection model with multiple algorithms")
    
    # Dataset and output directories
    parser.add_argument(
        "--dataset-dir",
        default="/Users/anish/Documents/UT Austin/Spring 2025/Courses/Visual Recognition/Project/Testing/mydata/dataset",
        help="path to the dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        default="output_all_algos",
        help="directory to save checkpoints and logs",
    )
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        default="train",
        help="run type of the experiment (train or eval)",
    )
    
    # Algorithm selection
    parser.add_argument(
        "--algorithm",
        choices=["PPO", "A2C", "DQN"],
        default="PPO",
        help="RL algorithm to use for training",
    )
    
    # Feature extraction parameters
    parser.add_argument(
        "--max-views",
        type=int,
        default=6,
        help="maximum number of camera views the model can handle",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=4096,
        help="dimension of features",
    )
    parser.add_argument(
        "--feature-dir",
        default="/Users/anish/Documents/UT Austin/Spring 2025/Courses/Visual Recognition/Project/Testing/multiview_rl/features_new",
        help="directory to save/load features",
    )
    
    # Model parameters
    parser.add_argument(
        "--model-path",
        default=None,
        help="path to saved model for evaluation",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=150000,
        help="total timesteps for training",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=50,
        help="number of episodes for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )
    
    # Training parameters - shared across algorithms
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=256,
        help="n_steps parameter (for PPO and A2C)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="batch size (for PPO and DQN)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="n_epochs parameter (for PPO)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.2,
        help="entropy coefficient for exploration (for PPO and A2C)",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="clip range (for PPO)",
    )
    
    # Exploration parameters
    parser.add_argument(
        "--random-steps",
        type=int,
        default=10000,
        help="Number of initial random steps for exploration",
    )
    parser.add_argument(
        "--exploration-fraction",
        type=float,
        default=0.3,
        help="Fraction of random actions after initial exploration",
    )
    
    # Reward function parameters
    parser.add_argument(
        "--informativeness-weight",
        type=float,
        default=1.5,
        help="weight for informativeness reward (matching best camera)",
    )
    parser.add_argument(
        "--narration-activity-weight",
        type=float,
        default=0.4,
        help="weight for narration activity reward",
    )
    parser.add_argument(
        "--switch-penalty",
        type=float,
        default=0.05,
        help="penalty for switching views",
    )
    parser.add_argument(
        "--visual-similarity-weight",
        type=float,
        default=0.2,
        help="weight for visual similarity penalty",
    )
    parser.add_argument(
        "--narrative-context-weight",
        type=float,
        default=0.8,
        help="weight for narrative context modulation",
    )
    parser.add_argument(
        "--exploration-bonus-weight",
        type=float,
        default=0.2,
        help="weight for exploration bonus to encourage view diversity",
    )
    parser.add_argument(
        "--switch-incentive-weight",
        type=float,
        default=0.1,
        help="weight for switch incentive when stuck on same view",
    )
    
    # Testing parameters
    parser.add_argument(
        "--test-take",
        default="minnesota_cooking_060_2",
        help="Specific take to use for testing",
    )
    
    # Debugging parameter
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debugging output",
    )
    
    args = parser.parse_args()
    
    if args.run_type == "train":
        # Train model with selected algorithm
        model, env_config, algo_output_dir = train_model(args)
        
        # Evaluate the trained model
        eval_dir = os.path.join(algo_output_dir, "evaluation")
        metrics = detailed_evaluation(
            model=model,
            env_config=env_config,
            output_dir=eval_dir,
            num_episodes=args.eval_episodes
        )
        
        # Create evaluation plots
        create_evaluation_plots(
            metrics=metrics,
            all_episodes=None,
            output_dir=eval_dir
        )
        
    elif args.run_type == "eval":
        if args.model_path is None:
            model_path = os.path.join(args.output_dir, args.algorithm.lower(), f"multiview_{args.algorithm.lower()}_final")
        else:
            model_path = args.model_path
        
        # Try to load parameters if available
        reward_params_path = os.path.join(os.path.dirname(model_path), "reward_params.json")
        
        if os.path.exists(reward_params_path):
            with open(reward_params_path, 'r') as f:
                reward_params = json.load(f)
            print(f"Loaded reward parameters from {reward_params_path}")
        else:
            # Use command line arguments
            reward_params = {
                "informativeness_weight": args.informativeness_weight,
                "narration_activity_weight": args.narration_activity_weight,
                "switch_penalty": args.switch_penalty,
                "visual_similarity_weight": args.visual_similarity_weight,
                "narrative_context_weight": args.narrative_context_weight,
                "exploration_bonus_weight": args.exploration_bonus_weight,
                "switch_incentive_weight": args.switch_incentive_weight
            }
            print("Using command line reward parameters")
        
        # Create environment config with loaded parameters
        env_config = {
            "dataset_dir": args.dataset_dir,
            "feature_extractor": None,
            "feature_dir": args.feature_dir,
            "max_views": args.max_views,
            "feature_dim": args.feature_dim,
            "episode_length": 10,
            "reward_params": reward_params,
        }
        
        # Load the model based on the algorithm
        if args.algorithm.upper() == "PPO":
            model = PPO.load(model_path)
        elif args.algorithm.upper() == "A2C":
            model = A2C.load(model_path)
        elif args.algorithm.upper() == "DQN":
            model = DQN.load(model_path)
        else:
            raise ValueError(f"Unknown algorithm: {args.algorithm}")
        
        # Run detailed evaluation
        algo_output_dir = os.path.join(args.output_dir, args.algorithm.lower())
        eval_dir = os.path.join(algo_output_dir, "evaluation")
        metrics = detailed_evaluation(
            model=model,
            env_config=env_config,
            output_dir=eval_dir,
            num_episodes=args.eval_episodes
        )
        
        # Create evaluation plots
        create_evaluation_plots(
            metrics=metrics,
            all_episodes=None,
            output_dir=eval_dir
        )

if __name__ == "__main__":
    main()