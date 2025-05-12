import gymnasium as gym
import numpy as np
import os
import json
from gymnasium import spaces
import cv2
from scipy.spatial.distance import cosine

class MultiViewEnv(gym.Env):
    """Custom Environment for multi-view selection with improved reward function"""
    
    # def __init__(self, config):
    #     super(MultiViewEnv, self).__init__()
    #     self.dataset_dir = config["dataset_dir"]
    #     self.feature_extractor = config.get("feature_extractor", None)
    #     self.feature_dir = config.get("feature_dir", None)
        
    #     # Maximum number of views the model can handle
    #     # This sets the upper limit for the action space
    #     self.max_views = config.get("max_views", 6)
        
    #     # Add reward function parameters with improved values
    #     self.reward_params = config.get("reward_params", {
    #         "informativeness_weight": 1.5,       
    #         "narration_activity_weight": 0.4,    
    #         "switch_penalty": -0.1,              # Make negative to reward switching!
    #         "visual_similarity_weight": 0.1,     # Reduced
    #         "narrative_context_weight": 0.8,     
    #         "exploration_bonus_weight": 1.0,     # Increased
    #         "switch_incentive_weight": 2.0       # Significantly increased
    #     })
        
    #     # Initialize view history trackers for the new reward components
    #     self.view_selection_history = []
    #     self.steps_since_switch = 0
    #     self.view_visit_counts = {}
        
    #     # Load takes
    #     self.takes = self._load_takes()
        
    #     # Initialize current take and clip indices
    #     self.current_take_idx = 0
    #     self.current_clip_idx = 0
    #     self.episode_length = config.get("episode_length", 10)  # Number of clips per episode
    #     self.step_count = 0
        
    #     # Define action and observation space
    #     # Using max_views for the action space
    #     self.action_space = spaces.Discrete(self.max_views)
        
    #     # Observation space: features from all views
    #     self.feature_dim = config.get("feature_dim", 2048)
    #     self.observation_space = spaces.Dict({
    #         "features": spaces.Box(
    #             low=-np.inf, high=np.inf, 
    #             shape=(self.max_views, self.feature_dim), 
    #             dtype=np.float32
    #         ),
    #         "current_view": spaces.Box(
    #             low=0, high=self.max_views-1,
    #             shape=(1,), dtype=np.int32
    #         ),
    #         "time_step": spaces.Box(
    #             low=0, high=self.episode_length-1,
    #             shape=(1,), dtype=np.int32
    #         ),
    #         "valid_views": spaces.Box(
    #             low=0, high=1,
    #             shape=(self.max_views,), dtype=np.int32
    #         )
    #     })
        
    #     # Initialize similarity cache
    #     self.clip_similarities_cache = {}

    # def __init__(self, config):
    #     super(MultiViewEnv, self).__init__()
    #     self.dataset_dir = config["dataset_dir"]
    #     self.feature_extractor = config.get("feature_extractor", None)
    #     self.feature_dir = config.get("feature_dir", None)
        
    #     # Maximum number of views the model can handle
    #     self.max_views = config.get("max_views", 6)
        
    #     # Add reward function parameters with improved values
    #     self.reward_params = config.get("reward_params", {
    #         "informativeness_weight": 1.5,       
    #         "narration_activity_weight": 0.4,    
    #         "switch_penalty": -0.1,              # Make negative to reward switching!
    #         "visual_similarity_weight": 0.1,     # Reduced
    #         "narrative_context_weight": 0.8,     
    #         "exploration_bonus_weight": 1.0,     # Increased
    #         "switch_incentive_weight": 2.0       # Significantly increased
    #     })
        
    #     # Initialize view history trackers for the new reward components
    #     self.view_selection_history = []
    #     self.steps_since_switch = 0
    #     self.view_visit_counts = {}
        
    #     # Load takes
    #     self.takes = self._load_takes()
        
    #     # Initialize current take and clip indices
    #     self.current_take_idx = 0
    #     self.current_clip_idx = 0
    #     self.episode_length = config.get("episode_length", 10)  # Number of clips per episode
    #     self.step_count = 0
        
    #     # Initialize similarity cache
    #     self.clip_similarities_cache = {}
        
    #     # Enable auto adjustment of feature dimension (new parameter)
    #     self.auto_adjust_feature_dim = config.get("auto_adjust_feature_dim", True)
        
    #     # Try to detect the actual feature dimension from the data
    #     feature_dim = config.get("feature_dim", 4096)
    #     if self.auto_adjust_feature_dim and self.takes and self.feature_dir:
    #         detected_dim = self._detect_feature_dimension()
    #         if detected_dim > 0:
    #             feature_dim = detected_dim
    #             print(f"Detected feature dimension: {feature_dim}")
        
    #     self.feature_dim = feature_dim
    #     print(f"Using feature dimension: {self.feature_dim}")
        
    #     # Define action and observation space
    #     # Using max_views for the action space
    #     self.action_space = spaces.Discrete(self.max_views)
        
    #     # Observation space: features from all views
    #     self.observation_space = spaces.Dict({
    #         "features": spaces.Box(
    #             low=-np.inf, high=np.inf, 
    #             shape=(self.max_views, self.feature_dim), 
    #             dtype=np.float32
    #         ),
    #         "current_view": spaces.Box(
    #             low=0, high=self.max_views-1,
    #             shape=(1,), dtype=np.int32
    #         ),
    #         "time_step": spaces.Box(
    #             low=0, high=self.episode_length-1,
    #             shape=(1,), dtype=np.int32
    #         ),
    #         "valid_views": spaces.Box(
    #             low=0, high=1,
    #             shape=(self.max_views,), dtype=np.int32
    #         )
    #     })

    def __init__(self, config):
        super(MultiViewEnv, self).__init__()
        self.dataset_dir = config["dataset_dir"]
        self.feature_extractor = config.get("feature_extractor", None)
        self.feature_dir = config.get("feature_dir", None)
        
        # Maximum number of views the model can handle
        self.max_views = config.get("max_views", 6)
        
        # Add reward function parameters with improved values
        self.reward_params = config.get("reward_params", {
            "informativeness_weight": 1.5,       
            "narration_activity_weight": 0.4,    
            "switch_penalty": -0.1,              # Make negative to reward switching!
            "visual_similarity_weight": 0.1,     # Reduced
            "narrative_context_weight": 0.8,     
            "exploration_bonus_weight": 1.0,     # Increased
            "switch_incentive_weight": 2.0       # Significantly increased
        })
        
        # Initialize view history trackers for the new reward components
        self.view_selection_history = []
        self.steps_since_switch = 0
        self.view_visit_counts = {}
        
        # Load takes
        self.takes = self._load_takes()
        
        # Initialize current take and clip indices
        self.current_take_idx = 0
        self.current_clip_idx = 0
        self.episode_length = config.get("episode_length", 10)  # Number of clips per episode
        self.step_count = 0
        
        # Initialize similarity cache
        self.clip_similarities_cache = {}
        
        # Enable auto adjustment of feature dimension
        self.auto_adjust_feature_dim = config.get("auto_adjust_feature_dim", True)
        
        # Set initial feature dimension from config
        self.feature_dim = config.get("feature_dim", 4096)  # Default to 4096 as most common
        
        # Try to detect the actual feature dimension from the data
        if self.auto_adjust_feature_dim and self.takes and self.feature_dir:
            detected_dim = self._detect_feature_dimension()
            if detected_dim > 0:
                self.feature_dim = detected_dim
                print(f"Using feature dimension: {self.feature_dim} (detected from data)")
            else:
                print(f"Using feature dimension: {self.feature_dim} (from config)")
        
        # Define action and observation space
        self.action_space = spaces.Discrete(self.max_views)
        
        # Observation space: features from all views
        self.observation_space = spaces.Dict({
            "features": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.max_views, self.feature_dim), 
                dtype=np.float32
            ),
            "current_view": spaces.Box(
                low=0, high=self.max_views-1,
                shape=(1,), dtype=np.int32
            ),
            "time_step": spaces.Box(
                low=0, high=self.episode_length-1,
                shape=(1,), dtype=np.int32
            ),
            "valid_views": spaces.Box(
                low=0, high=1,
                shape=(self.max_views,), dtype=np.int32
            )
        })
        
        print(f"Environment initialized with feature dimension: {self.feature_dim}")

    # def _detect_feature_dimension(self):
    #     """
    #     Detect the feature dimension by examining a sample feature file
    #     Returns the detected dimension or -1 if detection failed
    #     """
    #     if not self.takes or not self.feature_dir:
    #         return -1
        
    #     # Try to find a feature file to examine
    #     for take_idx in range(min(5, len(self.takes))):  # Check first 5 takes at most
    #         take = self.takes[take_idx]
    #         if not take["clip_dirs"]:
    #             continue
                
    #         for clip_idx in range(min(5, len(take["clip_dirs"]))):  # Check first 5 clips at most
    #             clip_dir_name = take["clip_dirs"][clip_idx]
                
    #             # Try the nested structure
    #             feature_dir = os.path.join(self.feature_dir, take["take_dir"], "clips", clip_dir_name)
    #             if os.path.exists(feature_dir) and os.path.isdir(feature_dir):
    #                 feature_files = [f for f in os.listdir(feature_dir) 
    #                             if (isinstance(f, str) and (f.endswith('.pt') or f.endswith('.npy')))]
                    
    #                 if feature_files:
    #                     feature_path = os.path.join(feature_dir, feature_files[0])
    #                     try:
    #                         if feature_path.endswith('.pt'):
    #                             import torch
    #                             feature = torch.load(feature_path).numpy()
    #                         else:
    #                             feature = np.load(feature_path)
                                
    #                         if len(feature.shape) > 1:
    #                             return feature.shape[1]  # Return the feature dimension
    #                         else:
    #                             return len(feature)  # If it's a 1D feature
    #                     except Exception as e:
    #                         print(f"Error loading feature file {feature_path} for dimension detection: {e}")
                
    #             # Try the traditional structure
    #             feature_path = os.path.join(self.feature_dir, take["take_dir"], f"{clip_dir_name}.pt")
    #             if not os.path.exists(feature_path):
    #                 feature_path = os.path.join(self.feature_dir, take["take_dir"], f"{clip_dir_name}.npy")
                    
    #             if os.path.exists(feature_path):
    #                 try:
    #                     if feature_path.endswith('.pt'):
    #                         import torch
    #                         feature = torch.load(feature_path).numpy()
    #                     else:
    #                         feature = np.load(feature_path)
                            
    #                     if len(feature.shape) > 1:
    #                         return feature.shape[1]  # Return the feature dimension
    #                     else:
    #                         return len(feature)  # If it's a 1D feature
    #                 except Exception as e:
    #                     print(f"Error loading feature file {feature_path} for dimension detection: {e}")

    def _detect_feature_dimension(self):
        """Detect feature dimensions in the dataset and choose the most appropriate"""
        if not self.takes or not self.feature_dir:
            return -1
        
        dimensions = {}  # Track dimension counts
        
        # Sample takes to get a representation of dimension distribution
        for take_idx in range(min(20, len(self.takes))):
            take = self.takes[take_idx]
            if not take["clip_dirs"]:
                continue
                
            # Check a few clips per take
            for clip_idx in range(min(3, len(take["clip_dirs"]))):
                clip_dir_name = take["clip_dirs"][clip_idx]
                
                # Try both directory structures to find features
                dim = self._get_feature_dimension_for_clip(take["take_dir"], clip_dir_name)
                if dim > 0:
                    dimensions[dim] = dimensions.get(dim, 0) + 1
        
        if not dimensions:
            print("Could not detect feature dimensions, using default from config")
            return -1
        
        # Print the discovered dimensions and their frequencies
        print(f"Detected feature dimensions across dataset: {dimensions}")
        
        # Since we know the vast majority are 4096, prefer the largest dimension
        if len(dimensions) > 1:
            max_dim = max(dimensions.keys())
            print(f"Multiple feature dimensions detected. Using largest dimension: {max_dim}")
            return max_dim
        else:
            # Otherwise use the single dimension found
            only_dim = list(dimensions.keys())[0]
            print(f"Consistent feature dimension detected: {only_dim}")
            return only_dim

    def _load_takes(self):
        """Load the list of takes from the dataset, focusing only on those with extracted features"""
        takes = []
        
        # Check which takes have features extracted
        available_takes = set()
        if self.feature_dir is not None and os.path.exists(self.feature_dir):
            # List all the take directories in the feature_dir
            potential_takes = [d for d in os.listdir(self.feature_dir) 
                            if os.path.isdir(os.path.join(self.feature_dir, d))]
            
            # Check each potential take for feature files
            for take_dir in potential_takes:
                take_feature_dir = os.path.join(self.feature_dir, take_dir)
                
                # Check if there's a clips directory (your specific structure)
                clips_dir = os.path.join(take_feature_dir, "clips")
                if os.path.exists(clips_dir) and os.path.isdir(clips_dir):
                    # Check if there are clip folders with feature files
                    clip_dirs = [d for d in os.listdir(clips_dir) 
                                if os.path.isdir(os.path.join(clips_dir, d)) and d.startswith("clip_")]
                    
                    has_features = False
                    for clip_dir in clip_dirs:
                        clip_path = os.path.join(clips_dir, clip_dir)
                        
                        # First check direct feature files
                        feature_files = [f for f in os.listdir(clip_path) 
                                        if isinstance(f, str) and (f.endswith('.pt') or f.endswith('.npy'))]
                        
                        # If no direct feature files, check subdirectories
                        if not feature_files:
                            sub_dirs = [d for d in os.listdir(clip_path) 
                                    if os.path.isdir(os.path.join(clip_path, d))]
                            
                            for sub_dir in sub_dirs:
                                sub_dir_path = os.path.join(clip_path, sub_dir)
                                if os.path.exists(sub_dir_path) and os.path.isdir(sub_dir_path):
                                    sub_feature_files = [f for f in os.listdir(sub_dir_path) 
                                                    if isinstance(f, str) and (f.endswith('.pt') or f.endswith('.npy'))]
                                    if sub_feature_files:
                                        feature_files.extend(sub_feature_files)
                        
                        if feature_files:
                            has_features = True
                            break
                    
                    if has_features:
                        available_takes.add(take_dir)
                        print(f"Found features for take: {take_dir}")
                else:
                    # Check for the traditional structure (feature files directly in take folder)
                    feature_files = [f for f in os.listdir(take_feature_dir) 
                                    if isinstance(f, str) and (f.endswith('.pt') or f.endswith('.npy')) and f.startswith('clip_')]
                    if feature_files:
                        available_takes.add(take_dir)
                        print(f"Found features for take: {take_dir}")
        
        # If no takes with features found, exit early
        if not available_takes:
            print(f"No takes found with features in {self.feature_dir}. Exiting.")
            return takes
        
        print(f"Found {len(available_takes)} takes with available features")
        
        # Look for the ego4d directory
        ego4d_dir = os.path.join(self.dataset_dir, "ego4d_256")
        
        if not os.path.exists(ego4d_dir):
            print(f"Warning: ego4d_256 directory not found at {ego4d_dir}")
            # Continue processing using just the feature directory
        
        take_count = 0
        
        # Process each take with available features
        for take_dir in available_takes:
            # Check for the new directory structure pattern
            ego4d_take_path = os.path.join(ego4d_dir, take_dir)
            found_clips_dir = None
            
            # First try the EXACT structure you specified
            exact_clips_path = os.path.join(ego4d_take_path, "frame_aligned_videos", "downscaled", "448", "clips")
            if os.path.exists(exact_clips_path) and os.path.isdir(exact_clips_path):
                found_clips_dir = exact_clips_path
                print(f"Found clips directory at: {found_clips_dir}")
            
            # If not found, try to search for any "clips" directory under the take
            if not found_clips_dir and os.path.exists(ego4d_take_path) and os.path.isdir(ego4d_take_path):
                # Try to find a "clips" directory somewhere in the take's hierarchy
                for root, dirs, files in os.walk(ego4d_take_path):
                    if "clips" in dirs:
                        candidate_clips_dir = os.path.join(root, "clips")
                        # Check if this "clips" directory contains clip_xxx directories
                        if any(os.path.isdir(os.path.join(candidate_clips_dir, d)) and d.startswith("clip_") 
                            for d in os.listdir(candidate_clips_dir)):
                            found_clips_dir = candidate_clips_dir
                            print(f"Found clips directory at: {found_clips_dir}")
                            break
            
            # If still not found, use the clips directory from features
            if not found_clips_dir:
                feature_clips_dir = os.path.join(self.feature_dir, take_dir, "clips")
                if os.path.exists(feature_clips_dir) and os.path.isdir(feature_clips_dir):
                    found_clips_dir = feature_clips_dir
                    print(f"Using clips directory from features: {found_clips_dir}")
            
            # If we found a clips directory, process it
            if found_clips_dir:
                # Check for narrations file
                narrations_path = os.path.join(found_clips_dir, "narrations.json")
                narrations = {}
                
                if os.path.exists(narrations_path):
                    try:
                        with open(narrations_path, 'r') as f:
                            narrations = json.load(f)
                    except Exception as e:
                        print(f"Warning: Could not load narrations for {take_dir}: {e}")
                
                # Get list of clip directories
                clip_dirs = [d for d in os.listdir(found_clips_dir) 
                            if os.path.isdir(os.path.join(found_clips_dir, d)) 
                            and d.startswith("clip_")]
                
                if clip_dirs:
                    takes.append({
                        "take_dir": take_dir,
                        "clips_dir": found_clips_dir,
                        "narrations": narrations,
                        "clip_dirs": sorted(clip_dirs)
                    })
                    take_count += 1
        
        print(f"Loaded {take_count} takes with available features")
        
        # Print sample of loaded takes for debugging
        if takes:
            print(f"Sample take: {takes[0]['take_dir']} with {len(takes[0]['clip_dirs'])} clips")
        
        return takes
    
    def _get_feature_dimension_for_clip(self, take_dir, clip_dir_name):
        """Helper method to find the feature dimension for a specific clip"""
        # Try the nested structure
        feature_dir = os.path.join(self.feature_dir, take_dir, "clips", clip_dir_name)
        if os.path.exists(feature_dir) and os.path.isdir(feature_dir):
            feature_files = [f for f in os.listdir(feature_dir) 
                        if (isinstance(f, str) and (f.endswith('.pt') or f.endswith('.npy')))]
            
            if feature_files:
                feature_path = os.path.join(feature_dir, feature_files[0])
                try:
                    if feature_path.endswith('.pt'):
                        import torch
                        feature = torch.load(feature_path).numpy()
                    else:
                        feature = np.load(feature_path)
                        
                    if len(feature.shape) > 1:
                        return feature.shape[1]
                    else:
                        return len(feature)
                except Exception as e:
                    print(f"Error examining feature: {e}")
        
        # Try the traditional structure
        feature_path = os.path.join(self.feature_dir, take_dir, f"{clip_dir_name}.pt")
        if not os.path.exists(feature_path):
            feature_path = os.path.join(self.feature_dir, take_dir, f"{clip_dir_name}.npy")
            
        if os.path.exists(feature_path):
            try:
                if feature_path.endswith('.pt'):
                    import torch
                    feature = torch.load(feature_path).numpy()
                else:
                    feature = np.load(feature_path)
                    
                if len(feature.shape) > 1:
                    return feature.shape[1]
                else:
                    return len(feature)
            except Exception as e:
                print(f"Error examining feature: {e}")
        
        return -1
    
    # def _get_features(self, take_idx, clip_idx):
    #     """Get features for all views in the current clip and return valid mask"""
    #     take = self.takes[take_idx]
    #     clip_dir_name = take["clip_dirs"][clip_idx]
    #     clip_dir = os.path.join(take["clips_dir"], clip_dir_name)
        
    #     # Get how many cameras are in this clip
    #     cam_files = []
    #     if os.path.exists(clip_dir):
    #         cam_files = [f for f in os.listdir(clip_dir) if f.endswith('.mp4')]
    #     num_actual_cameras = len(cam_files)
        
    #     # Use pre-extracted features
    #     if self.feature_dir is not None:
    #         # First try the nested structure: features_new/take_name/clips/clip_xxx/*.pt
    #         feature_dir = os.path.join(self.feature_dir, take["take_dir"], "clips", clip_dir_name)
            
    #         # Check if the feature_dir exists but contains camera-specific feature files
    #         if os.path.exists(feature_dir) and os.path.isdir(feature_dir):
    #             # Find any .pt or .npy files in this directory
    #             feature_files = [f for f in os.listdir(feature_dir) 
    #                         if (isinstance(f, str) and (f.endswith('.pt') or f.endswith('.npy')))]
                
    #             # If no direct feature files, look for subdirectories that might contain features
    #             if not feature_files:
    #                 # Look for any subdirectory that contains .pt or .npy files
    #                 sub_dirs = [d for d in os.listdir(feature_dir) if os.path.isdir(os.path.join(feature_dir, d))]
                    
    #                 for sub_dir in sub_dirs:
    #                     sub_dir_path = os.path.join(feature_dir, sub_dir)
    #                     sub_feature_files = [f for f in os.listdir(sub_dir_path) 
    #                                     if (isinstance(f, str) and (f.endswith('.pt') or f.endswith('.npy')))]
    #                     if sub_feature_files:
    #                         for feature_file in sub_feature_files:
    #                             feature_files.append(os.path.join(sub_dir, feature_file))
                
    #             if feature_files:
    #                 # Sort the feature files to ensure consistent order
    #                 feature_files = sorted(feature_files)
    #                 features_list = []
                    
    #                 for feature_file in feature_files:
    #                     # Handle both direct files and files in subdirectories
    #                     if os.path.dirname(feature_file):  # File is in a subdirectory
    #                         feature_path = os.path.join(feature_dir, feature_file)
    #                     else:  # File is directly in feature_dir
    #                         feature_path = os.path.join(feature_dir, feature_file)
                        
    #                     try:
    #                         # Ensure feature_path is a string, not a PosixPath
    #                         feature_path = str(feature_path)
                            
    #                         # Load features based on file extension
    #                         if feature_path.endswith('.pt'):
    #                             import torch
    #                             feature = torch.load(feature_path).numpy()
    #                         else:
    #                             feature = np.load(feature_path)
                            
    #                         # Make sure feature is 1D
    #                         if len(feature.shape) > 1:
    #                             feature = feature.flatten()
                            
    #                         features_list.append(feature)
    #                     except Exception as e:
    #                         print(f"Error loading feature file {feature_path}: {e}")
                    
    #                 # Stack features if we found any
    #                 if features_list:
    #                     # Check if feature dimension is consistent
    #                     first_feature_dim = len(features_list[0])
                        
    #                     # Update feature dimension if different from config
    #                     if first_feature_dim != self.feature_dim:
    #                         print(f"Feature dimension mismatch. Expected {self.feature_dim}, got {first_feature_dim}. Updating feature_dim.")
    #                         self.feature_dim = first_feature_dim
                            
    #                         # Also update observation space
    #                         self.observation_space = spaces.Dict({
    #                             "features": spaces.Box(
    #                                 low=-np.inf, high=np.inf, 
    #                                 shape=(self.max_views, self.feature_dim), 
    #                                 dtype=np.float32
    #                             ),
    #                             "current_view": spaces.Box(
    #                                 low=0, high=self.max_views-1,
    #                                 shape=(1,), dtype=np.int32
    #                             ),
    #                             "time_step": spaces.Box(
    #                                 low=0, high=self.episode_length-1,
    #                                 shape=(1,), dtype=np.int32
    #                             ),
    #                             "valid_views": spaces.Box(
    #                                 low=0, high=1,
    #                                 shape=(self.max_views,), dtype=np.int32
    #                             )
    #                         })
                        
    #                     # Ensure all features have the same dimension
    #                     for i, feature in enumerate(features_list):
    #                         if len(feature) != self.feature_dim:
    #                             print(f"Warning: Feature {i} has dimension {len(feature)}, expected {self.feature_dim}. Reshaping...")
    #                             # Pad or truncate to match the dimension
    #                             if len(feature) > self.feature_dim:
    #                                 features_list[i] = feature[:self.feature_dim]
    #                             else:
    #                                 padded = np.zeros(self.feature_dim, dtype=np.float32)
    #                                 padded[:len(feature)] = feature
    #                                 features_list[i] = padded
                        
    #                     features = np.vstack(features_list)
                        
    #                     # Create valid mask (1 for valid views, 0 for padded views)
    #                     num_real_features = min(features.shape[0], self.max_views)
    #                     valid_mask = np.zeros(self.max_views, dtype=np.int32)
    #                     valid_mask[:num_real_features] = 1
                        
    #                     # Check if we have more features than max_views
    #                     if features.shape[0] > self.max_views:
    #                         print(f"Warning: Found {features.shape[0]} features for {take['take_dir']}/{clip_dir_name}, but max_views is {self.max_views}. Truncating.")
    #                         features = features[:self.max_views]
                        
    #                     # If we have fewer than max_views features, pad with zeros
    #                     if features.shape[0] < self.max_views:
    #                         padding = np.zeros((self.max_views - features.shape[0], self.feature_dim), dtype=np.float32)
    #                         features = np.vstack((features, padding))
                        
    #                     return features, valid_mask
            
    #         # Try the traditional structure: features_new/take_name/clip_xxx.pt
    #         feature_path = os.path.join(
    #             self.feature_dir, take["take_dir"], 
    #             f"{clip_dir_name}.pt"  # Using .pt extension for pytorch features
    #         )
            
    #         # Try .npy as fallback
    #         if not os.path.exists(feature_path):
    #             feature_path = os.path.join(
    #                 self.feature_dir, take["take_dir"], 
    #                 f"{clip_dir_name}.npy"
    #             )
            
    #         if os.path.exists(feature_path):
    #             try:
    #                 # Ensure feature_path is a string
    #                 feature_path = str(feature_path)
                    
    #                 # Load features based on file extension
    #                 if feature_path.endswith('.pt'):
    #                     import torch
    #                     features = torch.load(feature_path).numpy()
    #                 else:
    #                     features = np.load(feature_path)
                    
    #                 # Check feature dimensions and update if needed
    #                 if len(features.shape) > 1 and features.shape[1] != self.feature_dim:
    #                     print(f"Feature dimension mismatch. Expected {self.feature_dim}, got {features.shape[1]}. Updating feature_dim.")
    #                     self.feature_dim = features.shape[1]
                        
    #                     # Also update observation space
    #                     self.observation_space = spaces.Dict({
    #                         "features": spaces.Box(
    #                             low=-np.inf, high=np.inf, 
    #                             shape=(self.max_views, self.feature_dim), 
    #                             dtype=np.float32
    #                         ),
    #                         "current_view": spaces.Box(
    #                             low=0, high=self.max_views-1,
    #                             shape=(1,), dtype=np.int32
    #                         ),
    #                         "time_step": spaces.Box(
    #                             low=0, high=self.episode_length-1,
    #                             shape=(1,), dtype=np.int32
    #                         ),
    #                         "valid_views": spaces.Box(
    #                             low=0, high=1,
    #                             shape=(self.max_views,), dtype=np.int32
    #                         )
    #                     })
                    
    #                 # Create valid mask (1 for valid views, 0 for padded views)
    #                 num_real_features = min(features.shape[0], self.max_views)
    #                 valid_mask = np.zeros(self.max_views, dtype=np.int32)
    #                 valid_mask[:num_real_features] = 1
                    
    #                 # Check if we have more features than max_views
    #                 if features.shape[0] > self.max_views:
    #                     print(f"Warning: Found {features.shape[0]} features for {take['take_dir']}/{clip_dir_name}, but max_views is {self.max_views}. Truncating.")
    #                     features = features[:self.max_views]
                    
    #                 # If we have fewer than max_views features, pad with zeros
    #                 if features.shape[0] < self.max_views:
    #                     padding = np.zeros((self.max_views - features.shape[0], self.feature_dim), dtype=np.float32)
    #                     features = np.vstack((features, padding))
                    
    #                 return features, valid_mask
    #             except Exception as e:
    #                 print(f"Error loading feature file {feature_path}: {e}")
        
    #     # If no features found, return random features (for testing)
    #     print(f"No features found for {take['take_dir']}/{clip_dir_name}, using random features")
    #     random_features = np.random.randn(self.max_views, self.feature_dim).astype(np.float32)
    #     valid_mask = np.zeros(self.max_views, dtype=np.int32)
    #     if num_actual_cameras > 0:
    #         valid_mask[:min(num_actual_cameras, self.max_views)] = 1
    #     else:
    #         # If we couldn't determine the number of cameras, assume the first one is valid
    #         valid_mask[0] = 1
            
    #     return random_features, valid_mask

    def _load_features_for_clip(self, take_dir, clip_dir_name):
        """Helper method to load features for a specific clip, handling both directory structures"""
        features_list = []
        
        # Try the nested structure first
        feature_dir = os.path.join(self.feature_dir, take_dir, "clips", clip_dir_name)
        if os.path.exists(feature_dir) and os.path.isdir(feature_dir):
            feature_files = [f for f in os.listdir(feature_dir) 
                        if (isinstance(f, str) and (f.endswith('.pt') or f.endswith('.npy')))]
            
            if feature_files:
                feature_files = sorted(feature_files)
                for feature_file in feature_files:
                    feature_path = os.path.join(feature_dir, feature_file)
                    try:
                        if feature_path.endswith('.pt'):
                            import torch
                            feature = torch.load(feature_path).numpy()
                        else:
                            feature = np.load(feature_path)
                            
                        # Make sure feature is flattened if needed
                        if len(feature.shape) > 1 and feature.shape[0] == 1:
                            feature = feature.flatten()
                            
                        features_list.append(feature)
                    except Exception as e:
                        print(f"Error loading feature file {feature_path}: {e}")
        
        # If no features found, try the traditional structure
        if not features_list:
            feature_path = os.path.join(self.feature_dir, take_dir, f"{clip_dir_name}.pt")
            if not os.path.exists(feature_path):
                feature_path = os.path.join(self.feature_dir, take_dir, f"{clip_dir_name}.npy")
                
            if os.path.exists(feature_path):
                try:
                    if feature_path.endswith('.pt'):
                        import torch
                        feature = torch.load(feature_path).numpy()
                    else:
                        feature = np.load(feature_path)
                    
                    # If this is a 2D array of features
                    if len(feature.shape) > 1:
                        # Split into individual features
                        for i in range(feature.shape[0]):
                            features_list.append(feature[i])
                    else:
                        features_list.append(feature)
                except Exception as e:
                    print(f"Error loading feature file {feature_path}: {e}")
        
        return features_list

    def _get_features(self, take_idx, clip_idx):
        """Get features for all views in the current clip and return valid mask"""
        take = self.takes[take_idx]
        clip_dir_name = take["clip_dirs"][clip_idx]
        clip_dir = os.path.join(take["clips_dir"], clip_dir_name)
        
        # Get how many cameras are in this clip
        cam_files = []
        if os.path.exists(clip_dir):
            cam_files = [f for f in os.listdir(clip_dir) if f.endswith('.mp4')]
        num_actual_cameras = len(cam_files)
        
        # Use pre-extracted features
        if self.feature_dir is not None:
            features_list = []
            
            # Try to load features from both possible structures
            found_features = self._load_features_for_clip(take["take_dir"], clip_dir_name)
            
            if found_features:
                # Process each feature vector to standardize dimensions
                for feature in found_features:
                    # Skip features that are None (failed to load)
                    if feature is None:
                        continue
                        
                    # Handle dimension mismatch - pad smaller features to 4096
                    if len(feature) < self.feature_dim:
                        # Pad with zeros if smaller (e.g., 2048 -> 4096)
                        padded = np.zeros(self.feature_dim, dtype=np.float32)
                        padded[:len(feature)] = feature
                        features_list.append(padded)
                    else:
                        # Use as-is if dimensions match
                        features_list.append(feature)
                
                if features_list:
                    # Stack features
                    features = np.vstack(features_list)
                    
                    # Create valid mask
                    num_real_features = len(features_list)
                    valid_mask = np.zeros(self.max_views, dtype=np.int32)
                    valid_mask[:min(num_real_features, self.max_views)] = 1
                    
                    # Handle if we have more features than max_views
                    if features.shape[0] > self.max_views:
                        features = features[:self.max_views]
                    
                    # Pad if we have fewer than max_views
                    if features.shape[0] < self.max_views:
                        padding = np.zeros((self.max_views - features.shape[0], self.feature_dim), dtype=np.float32)
                        features = np.vstack((features, padding))
                    
                    return features, valid_mask
        
        # If no features found, return random features
        print(f"No features found for {take['take_dir']}/{clip_dir_name}, using random features")
        random_features = np.random.randn(self.max_views, self.feature_dim).astype(np.float32)
        valid_mask = np.zeros(self.max_views, dtype=np.int32)
        if num_actual_cameras > 0:
            valid_mask[:min(num_actual_cameras, self.max_views)] = 1
        else:
            valid_mask[0] = 1
            
        return random_features, valid_mask

    # def _get_features(self, take_idx, clip_idx):
    #     """Get features for all views in the current clip and return valid mask"""
    #     take = self.takes[take_idx]
    #     clip_dir_name = take["clip_dirs"][clip_idx]
    #     clip_dir = os.path.join(take["clips_dir"], clip_dir_name)
        
    #     # Get how many cameras are in this clip
    #     cam_files = []
    #     if os.path.exists(clip_dir):
    #         cam_files = [f for f in os.listdir(clip_dir) if f.endswith('.mp4')]
    #     num_actual_cameras = len(cam_files)
        
    #     # Use pre-extracted features
    #     if self.feature_dir is not None:
    #         # First try the nested structure: features_new/take_name/clips/clip_xxx/*.pt
    #         feature_dir = os.path.join(self.feature_dir, take["take_dir"], "clips", clip_dir_name)
            
    #         # Check if the feature_dir exists but contains camera-specific feature files
    #         if os.path.exists(feature_dir) and os.path.isdir(feature_dir):
    #             # Find any .pt or .npy files in this directory
    #             feature_files = [f for f in os.listdir(feature_dir) 
    #                         if (isinstance(f, str) and (f.endswith('.pt') or f.endswith('.npy')))]
                
    #             # If no direct feature files, look for subdirectories that might contain features
    #             if not feature_files:
    #                 # Look for any subdirectory that contains .pt or .npy files
    #                 sub_dirs = [d for d in os.listdir(feature_dir) if os.path.isdir(os.path.join(feature_dir, d))]
                    
    #                 for sub_dir in sub_dirs:
    #                     sub_dir_path = os.path.join(feature_dir, sub_dir)
    #                     sub_feature_files = [f for f in os.listdir(sub_dir_path) 
    #                                     if (isinstance(f, str) and (f.endswith('.pt') or f.endswith('.npy')))]
    #                     if sub_feature_files:
    #                         for feature_file in sub_feature_files:
    #                             feature_files.append(os.path.join(sub_dir, feature_file))
                
    #             if feature_files:
    #                 # Sort the feature files to ensure consistent order
    #                 feature_files = sorted(feature_files)
    #                 features_list = []
                    
    #                 for feature_file in feature_files:
    #                     # Handle both direct files and files in subdirectories
    #                     if os.path.dirname(feature_file):  # File is in a subdirectory
    #                         feature_path = os.path.join(feature_dir, feature_file)
    #                     else:  # File is directly in feature_dir
    #                         feature_path = os.path.join(feature_dir, feature_file)
                        
    #                     try:
    #                         # Ensure feature_path is a string, not a PosixPath
    #                         feature_path = str(feature_path)
                            
    #                         # Load features based on file extension
    #                         if feature_path.endswith('.pt'):
    #                             import torch
    #                             feature = torch.load(feature_path).numpy()
    #                         else:
    #                             feature = np.load(feature_path)
                            
    #                         # Make sure feature is 1D
    #                         if len(feature.shape) > 1:
    #                             feature = feature.flatten()
                            
    #                         features_list.append(feature)
    #                     except Exception as e:
    #                         print(f"Error loading feature file {feature_path}: {e}")
                    
    #                 # Stack features if we found any
    #                 if features_list:
    #                     features = np.vstack(features_list)
                        
    #                     # Create valid mask (1 for valid views, 0 for padded views)
    #                     num_real_features = min(features.shape[0], self.max_views)
    #                     valid_mask = np.zeros(self.max_views, dtype=np.int32)
    #                     valid_mask[:num_real_features] = 1
                        
    #                     # Check if we have more features than max_views
    #                     if features.shape[0] > self.max_views:
    #                         print(f"Warning: Found {features.shape[0]} features for {take['take_dir']}/{clip_dir_name}, but max_views is {self.max_views}. Truncating.")
    #                         features = features[:self.max_views]
                        
    #                     # If we have fewer than max_views features, pad with zeros
    #                     if features.shape[0] < self.max_views:
    #                         padding = np.zeros((self.max_views - features.shape[0], self.feature_dim), dtype=np.float32)
    #                         features = np.vstack((features, padding))
                        
    #                     return features, valid_mask
            
    #         # Try the traditional structure: features_new/take_name/clip_xxx.pt
    #         feature_path = os.path.join(
    #             self.feature_dir, take["take_dir"], 
    #             f"{clip_dir_name}.pt"  # Using .pt extension for pytorch features
    #         )
            
    #         # Try .npy as fallback
    #         if not os.path.exists(feature_path):
    #             feature_path = os.path.join(
    #                 self.feature_dir, take["take_dir"], 
    #                 f"{clip_dir_name}.npy"
    #             )
            
    #         if os.path.exists(feature_path):
    #             try:
    #                 # Ensure feature_path is a string
    #                 feature_path = str(feature_path)
                    
    #                 # Load features based on file extension
    #                 if feature_path.endswith('.pt'):
    #                     import torch
    #                     features = torch.load(feature_path).numpy()
    #                 else:
    #                     features = np.load(feature_path)
                    
    #                 # Create valid mask (1 for valid views, 0 for padded views)
    #                 num_real_features = min(features.shape[0], self.max_views)
    #                 valid_mask = np.zeros(self.max_views, dtype=np.int32)
    #                 valid_mask[:num_real_features] = 1
                    
    #                 # Check if we have more features than max_views
    #                 if features.shape[0] > self.max_views:
    #                     print(f"Warning: Found {features.shape[0]} features for {take['take_dir']}/{clip_dir_name}, but max_views is {self.max_views}. Truncating.")
    #                     features = features[:self.max_views]
                    
    #                 # If we have fewer than max_views features, pad with zeros
    #                 if features.shape[0] < self.max_views:
    #                     padding = np.zeros((self.max_views - features.shape[0], self.feature_dim), dtype=np.float32)
    #                     features = np.vstack((features, padding))
                    
    #                 return features, valid_mask
    #             except Exception as e:
    #                 print(f"Error loading feature file {feature_path}: {e}")
        
    #     # If no features found, return random features (for testing)
    #     print(f"No features found for {take['take_dir']}/{clip_dir_name}, using random features")
    #     random_features = np.random.randn(self.max_views, self.feature_dim).astype(np.float32)
    #     valid_mask = np.zeros(self.max_views, dtype=np.int32)
    #     if num_actual_cameras > 0:
    #         valid_mask[:min(num_actual_cameras, self.max_views)] = 1
    #     else:
    #         # If we couldn't determine the number of cameras, assume the first one is valid
    #         valid_mask[0] = 1
            
    #     return random_features, valid_mask
    
    def reset(self, seed=None, options=None):
        """Reset the environment to start a new episode"""
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Select a random take
        self.current_take_idx = np.random.randint(0, len(self.takes))
        
        # Start from the beginning of the take
        self.current_clip_idx = 0
        self.step_count = 0
        
        # Reset history trackers for reward components
        self.view_selection_history = []
        self.steps_since_switch = 0
        self.view_visit_counts = {i: 0 for i in range(self.max_views)}
        
        # Get initial features and valid mask
        features, valid_mask = self._get_features(self.current_take_idx, self.current_clip_idx)
        
        # Get valid camera indices (non-zero in the mask)
        valid_indices = np.where(valid_mask > 0)[0]
        if len(valid_indices) == 0:
            # Fallback if no valid cameras found
            valid_indices = np.array([0])
            valid_mask[0] = 1
        
        # Start with a random valid view
        current_view = np.random.choice(valid_indices)
        self.view_visit_counts[current_view] = 1
        
        observation = {
            "features": features,
            "current_view": np.array([current_view], dtype=np.int32),
            "time_step": np.array([self.step_count], dtype=np.int32),
            "valid_views": valid_mask
        }
        
        # In Gymnasium, reset should return (observation, info)
        return observation, {}
    
    def step(self, action):
        """Take a step in the environment by selecting a view"""
        # Ensure action is valid
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Get the current take and clip information
        take = self.takes[self.current_take_idx]
        clip_str = take["clip_dirs"][self.current_clip_idx].replace("clip_", "")
        try:
            clip_num = int(clip_str)
        except ValueError:
            clip_num = 0  # Fallback if clip_str is not a number
        
        # Get valid views
        _, valid_mask = self._get_features(self.current_take_idx, self.current_clip_idx)
        
        # Convert action to int if it's a numpy array
        if isinstance(action, np.ndarray):
            action = int(action.item())
        
        # Additional penalty if the selected view is invalid
        invalid_view_penalty = 0.0
        if valid_mask[action] == 0:
            invalid_view_penalty = -1.0  # Strong penalty for selecting an invalid view
            
            # Find a valid view to use instead
            valid_indices = np.where(valid_mask > 0)[0]
            if len(valid_indices) > 0:
                action = valid_indices[0]  # Use the first valid view
        
        # Calculate reward based on the selected view
        reward = self._calculate_reward(action, clip_num) + invalid_view_penalty
        
        # Update view history
        self.view_selection_history.append(action)
        if len(self.view_selection_history) > 20:  # Keep limited history
            self.view_selection_history.pop(0)
        
        # Update view visit counts
        if isinstance(action, np.ndarray):
            action_key = int(action.item())
        else:
            action_key = action
        self.view_visit_counts[action_key] = self.view_visit_counts.get(action_key, 0) + 1
        
        # Move to the next clip
        self.current_clip_idx += 1
        self.step_count += 1
        
        # Check if episode is done
        terminated = (self.step_count >= self.episode_length or 
                 self.current_clip_idx >= len(take["clip_dirs"]))
        truncated = False  # We're not truncating episodes
        
        # If done, return terminal state
        if terminated:
            observation = {
                "features": np.zeros((self.max_views, self.feature_dim), dtype=np.float32),
                "current_view": np.array([action], dtype=np.int32),
                "time_step": np.array([self.step_count], dtype=np.int32),
                "valid_views": np.zeros(self.max_views, dtype=np.int32)
            }
            return observation, reward, terminated, truncated, {}
        
        # Get features for the next clip
        next_features, next_valid_mask = self._get_features(self.current_take_idx, self.current_clip_idx)
        
        # Return the next state, reward, done flag, and info
        observation = {
            "features": next_features,
            "current_view": np.array([action], dtype=np.int32),
            "time_step": np.array([self.step_count], dtype=np.int32),
            "valid_views": next_valid_mask
        }
        return observation, reward, terminated, truncated, {}
        
    # def _calculate_reward(self, action, clip_num):
    #     """Enhanced reward function to encourage view switching"""
    #     # Convert action to int if it's a numpy array
    #     if isinstance(action, np.ndarray):
    #         action = int(action.item())
        
    #     # --- 1. Informativeness Reward ---
    #     info_reward = 0.0
        
    #     # Get narrations for the current clip
    #     take = self.takes[self.current_take_idx]
    #     narrations = take["narrations"]
    #     clip_str = str(clip_num)
        
    #     # Get narrations if available
    #     has_narrations = ("clip_narrations" in narrations and 
    #                     clip_str in narrations.get("clip_narrations", {}))
        
    #     if has_narrations:
    #         clip_narrations = narrations["clip_narrations"][clip_str]
            
    #         # Get the view names mapping
    #         view_names = self._get_view_names(take["take_dir"])
            
    #         # Check if the selected view matches any best_camera in narrations
    #         for narration in clip_narrations:
    #             best_camera = narration.get("best_camera", "")
    #             if best_camera and view_names.get(action) == best_camera:
    #                 info_reward += self.reward_params["informativeness_weight"]  # Weighted reward
            
    #         # Additional reward based on number of narrations (more activity = more important)
    #         info_reward += min(self.reward_params["narration_activity_weight"] * len(clip_narrations), 0.6)
    #     else:
    #         # No narrations - use a default value
    #         info_reward = 0.3  # Default reward when no narrations are available
        
    #     # --- 2. Switching Penalty ---
    #     switch_penalty = 0.0
        
    #     # Only apply switching penalty if not the first step
    #     if hasattr(self, 'previous_action'):
    #         # Simple binary penalty if views change
    #         if action != self.previous_action:
    #             switch_penalty = -self.reward_params["switch_penalty"]
    #             self.steps_since_switch = 0  # Reset counter
    #         else:
    #             self.steps_since_switch += 1  # Increment counter
    #     else:
    #         self.steps_since_switch = 0
        
    #     # --- 3. Visual Similarity Penalty (Gradual Switch Penalty) ---
    #     visual_similarity_penalty = 0.0
        
    #     if hasattr(self, 'previous_action') and action != self.previous_action:
    #         # Get current and previous clip features
    #         current_features = self._get_features(self.current_take_idx, self.current_clip_idx)
            
    #         # If we're not at the first clip
    #         if self.current_clip_idx > 0:
    #             try:
    #                 previous_features = self._get_features(self.current_take_idx, self.current_clip_idx - 1)
                    
    #                 # Calculate visual distance between the views
    #                 current_view_features = current_features[action]
    #                 prev_view_features = previous_features[self.previous_action]

    #                 # Calculate norms with safety checks
    #                 current_norm = np.linalg.norm(current_view_features)
    #                 prev_norm = np.linalg.norm(prev_view_features)

    #                 # Clamp norms to a minimum value to avoid division by zero
    #                 min_norm = 1e-8
    #                 current_norm = max(current_norm, min_norm)
    #                 prev_norm = max(prev_norm, min_norm)
                    
    #                 # Normalize vectors with clamped norms
    #                 current_view_norm = current_view_features / current_norm
    #                 prev_view_norm = prev_view_features / prev_norm
                    
    #                 # Calculate dot product for cosine similarity
    #                 dot_product = np.dot(current_view_norm, prev_view_norm)
                    
    #                 # Clamp dot product to [-1, 1] to ensure valid cosine distance
    #                 dot_product = max(-1.0, min(1.0, dot_product))
                    
    #                 # Manual cosine distance calculation (1 - similarity)
    #                 visual_distance = 1.0 - dot_product
                    
    #                 # Ensure visual_distance is non-negative and finite
    #                 visual_distance = max(0.0, min(2.0, visual_distance))
                    
    #                 # Scale penalty based on visual distance (more different = higher penalty)
    #                 visual_similarity_penalty = -self.reward_params["visual_similarity_weight"] * visual_distance
    #             except Exception as e:
    #                 # If there's an error calculating visual similarity, skip this penalty
    #                 print(f"Error calculating visual similarity: {e}")
    #                 visual_similarity_penalty = 0.0
        
    #     # --- 4. Narrative Context Modulation ---
    #     narrative_context_reward = 0.0
        
    #     if (hasattr(self, 'previous_clip_narrations') and 
    #         has_narrations and 
    #         self.previous_clip_narrations):
            
    #         current_narrations = narrations["clip_narrations"][clip_str]
    #         previous_narrations = self.previous_clip_narrations
            
    #         # Simple step change detection
    #         step_change_score = self._detect_narrative_step_change(previous_narrations, current_narrations)
            
    #         # If there's a significant change in narration, reduce the penalty for switching views
    #         if action != self.previous_action:
    #             narrative_context_reward = self.reward_params["narrative_context_weight"] * step_change_score
        
    #     # --- 5. NEW: Exploration Bonus ---
    #     exploration_bonus = 0.0
        
    #     # Calculate total visits
    #     total_visits = sum(self.view_visit_counts.values()) or 1  # Avoid division by zero
        
    #     if total_visits > self.num_views:  # Only add bonus when we have enough data
    #         # Calculate probability distribution over views
    #         view_probs = {v: count/total_visits for v, count in self.view_visit_counts.items()}
            
    #         # Get least and most visited views
    #         min_visits = min(view_probs.values())
    #         max_visits = max(view_probs.values())
            
    #         # Calculate a score based on how frequently this view has been visited
    #         # The less visited, the higher the bonus
    #         if max_visits > min_visits:  # Avoid division by zero
    #             normalized_score = (max_visits - view_probs[action]) / (max_visits - min_visits)
    #             exploration_bonus = self.reward_params["exploration_bonus_weight"] * normalized_score
        
    #     # --- 6. NEW: Switch Incentive (to avoid getting stuck) ---
    #     switch_incentive = 0.0
        
    #     # If we've been stuck on the same view for too long
    #     if self.steps_since_switch > 3:  # After 5 consecutive same views
    #         # Add an increasing incentive to switch
    #         switch_incentive = self.reward_params["switch_incentive_weight"] * (2 ** self.steps_since_switch - 1)
        
    #     # Store current action and narrations for next step
    #     self.previous_action = action
        
    #     # Store current narrations if available
    #     if has_narrations:
    #         self.previous_clip_narrations = narrations["clip_narrations"][clip_str]
    #     else:
    #         self.previous_clip_narrations = []
        
    #     # --- 7. Combine all reward components ---
    #     total_reward = (
    #         info_reward + 
    #         switch_penalty + 
    #         visual_similarity_penalty + 
    #         narrative_context_reward + 
    #         exploration_bonus + 
    #         switch_incentive
    #     )
        
    #     return total_reward

    def _calculate_reward(self, action, clip_num):
        """Enhanced reward function to encourage view switching"""
        # Convert action to int if it's a numpy array
        if isinstance(action, np.ndarray):
            action = int(action.item())
        
        # --- 1. Informativeness Reward ---
        info_reward = 0.0
        
        # Get narrations for the current clip
        take = self.takes[self.current_take_idx]
        narrations = take["narrations"]
        clip_str = str(clip_num)
        
        # Get narrations if available
        has_narrations = ("clip_narrations" in narrations and 
                        clip_str in narrations.get("clip_narrations", {}))
        
        if has_narrations:
            clip_narrations = narrations["clip_narrations"][clip_str]
            
            # Get the view names mapping
            view_names = self._get_view_names(take["take_dir"])
            
            # Check if the selected view matches any best_camera in narrations
            for narration in clip_narrations:
                best_camera = narration.get("best_camera", "")
                if best_camera and view_names.get(action) == best_camera:
                    info_reward += self.reward_params["informativeness_weight"]  # Weighted reward
            
            # Additional reward based on number of narrations (more activity = more important)
            info_reward += min(self.reward_params["narration_activity_weight"] * len(clip_narrations), 0.6)
        else:
            # No narrations - use a default value
            info_reward = 0.3  # Default reward when no narrations are available
        
        # --- 2. Switching Penalty ---
        switch_penalty = 0.0
        
        # Only apply switching penalty if not the first step
        if hasattr(self, 'previous_action'):
            # Simple binary penalty if views change
            if action != self.previous_action:
                switch_penalty = self.reward_params["switch_penalty"]  # This can be negative to reward switching
                self.steps_since_switch = 0  # Reset counter
            else:
                self.steps_since_switch += 1  # Increment counter
        else:
            self.steps_since_switch = 0
        
        # --- 3. Visual Similarity Penalty (Gradual Switch Penalty) ---
        visual_similarity_penalty = 0.0
        
        if hasattr(self, 'previous_action') and action != self.previous_action:
            # Get current and previous clip features
            current_features, _ = self._get_features(self.current_take_idx, self.current_clip_idx)
            
            # If we're not at the first clip
            if self.current_clip_idx > 0:
                try:
                    previous_features, _ = self._get_features(self.current_take_idx, self.current_clip_idx - 1)
                    
                    # Calculate visual distance between the views
                    if len(current_features) > action and len(previous_features) > self.previous_action:
                        current_view_features = current_features[action]
                        prev_view_features = previous_features[self.previous_action]
                        
                        # Calculate norms with safety checks
                        current_norm = np.linalg.norm(current_view_features)
                        prev_norm = np.linalg.norm(prev_view_features)
                        
                        # Clamp norms to a minimum value to avoid division by zero
                        min_norm = 1e-8
                        current_norm = max(current_norm, min_norm)
                        prev_norm = max(prev_norm, min_norm)
                        
                        # Normalize vectors with clamped norms
                        current_view_norm = current_view_features / current_norm
                        prev_view_norm = prev_view_features / prev_norm
                        
                        # Calculate dot product for cosine similarity
                        dot_product = np.dot(current_view_norm, prev_view_norm)
                        
                        # Clamp dot product to [-1, 1] to ensure valid cosine distance
                        dot_product = max(-1.0, min(1.0, dot_product))
                        
                        # Manual cosine distance calculation (1 - similarity)
                        visual_distance = 1.0 - dot_product
                        
                        # Ensure visual_distance is non-negative and finite
                        visual_distance = max(0.0, min(2.0, visual_distance))
                        
                        # Scale penalty based on visual distance (more different = higher penalty)
                        visual_similarity_penalty = -self.reward_params["visual_similarity_weight"] * visual_distance
                except Exception as e:
                    # If there's an error calculating visual similarity, skip this penalty
                    print(f"Error calculating visual similarity: {e}")
                    visual_similarity_penalty = 0.0
        
        # --- 4. Narrative Context Modulation ---
        narrative_context_reward = 0.0
        
        if (hasattr(self, 'previous_clip_narrations') and 
            has_narrations and 
            self.previous_clip_narrations):
            
            current_narrations = narrations["clip_narrations"][clip_str]
            previous_narrations = self.previous_clip_narrations
            
            # Simple step change detection
            step_change_score = self._detect_narrative_step_change(previous_narrations, current_narrations)
            
            # If there's a significant change in narration, reduce the penalty for switching views
            if action != self.previous_action:
                narrative_context_reward = self.reward_params["narrative_context_weight"] * step_change_score
        
        # --- 5. NEW: Exploration Bonus ---
        exploration_bonus = 0.0
        
        # Calculate total visits
        total_visits = sum(self.view_visit_counts.values()) or 1  # Avoid division by zero
        
        # Define num_views as the total number of valid views (max_views)
        num_views = self.max_views
        
        if total_visits > num_views:  # Only add bonus when we have enough data
            # Calculate probability distribution over views
            view_probs = {v: count/total_visits for v, count in self.view_visit_counts.items()}
            
            # Get least and most visited views
            min_visits = min(view_probs.values())
            max_visits = max(view_probs.values())
            
            # Calculate a score based on how frequently this view has been visited
            # The less visited, the higher the bonus
            if max_visits > min_visits:  # Avoid division by zero
                normalized_score = (max_visits - view_probs[action]) / (max_visits - min_visits)
                exploration_bonus = self.reward_params["exploration_bonus_weight"] * normalized_score
        
        # --- 6. NEW: Switch Incentive (to avoid getting stuck) ---
        switch_incentive = 0.0
        
        # If we've been stuck on the same view for too long
        if self.steps_since_switch > 3:  # After 3 consecutive same views
            # Add an increasing incentive to switch
            switch_incentive = self.reward_params["switch_incentive_weight"] * (2 ** (self.steps_since_switch - 3))
        
        # Store current action and narrations for next step
        self.previous_action = action
        
        # Store current narrations if available
        if has_narrations:
            self.previous_clip_narrations = narrations["clip_narrations"][clip_str]
        else:
            self.previous_clip_narrations = []
        
        # --- 7. Combine all reward components ---
        total_reward = (
            info_reward + 
            switch_penalty + 
            visual_similarity_penalty + 
            narrative_context_reward + 
            exploration_bonus + 
            switch_incentive
        )
        
        # For debugging
        if hasattr(self, 'debug') and self.debug:
            print(f"Reward components for action {action}:")
            print(f"  Info reward: {info_reward:.4f}")
            print(f"  Switch penalty: {switch_penalty:.4f}")
            print(f"  Visual similarity: {visual_similarity_penalty:.4f}")
            print(f"  Narrative context: {narrative_context_reward:.4f}")
            print(f"  Exploration bonus: {exploration_bonus:.4f}")
            print(f"  Switch incentive: {switch_incentive:.4f}")
            print(f"  Total reward: {total_reward:.4f}")
        
        return total_reward
    
    def _detect_narrative_step_change(self, prev_narrations, curr_narrations):
        """
        Enhanced detection of narrative step changes using more sophisticated NLP techniques
        """
        # If either list is empty, we can't detect a change
        if not prev_narrations or not curr_narrations:
            return 0.0
        
        # Extract text from narrations
        prev_texts = [n.get("text", "").lower() for n in prev_narrations if "text" in n]
        curr_texts = [n.get("text", "").lower() for n in curr_narrations if "text" in n]
        
        if not prev_texts or not curr_texts:
            return 0.0
            
        # Combine texts
        prev_text = " ".join(prev_texts)
        curr_text = " ".join(curr_texts)
        
        # Define action verbs and transitional phrases that indicate step changes
        action_verbs = ["picks", "takes", "puts", "places", "moves", "opens", "closes", 
                        "starts", "stops", "begins", "turns", "applies", "removes", "adds",
                        "cuts", "mixes", "stirs", "pours", "prepares", "finishes"]
        
        transition_phrases = ["next", "then", "after that", "following this", "now", 
                            "first", "second", "third", "finally", "lastly"]
        
        # Check for new action verbs in current narration that weren't in previous
        new_actions = 0
        for verb in action_verbs:
            if verb in curr_text and verb not in prev_text:
                new_actions += 1
        
        # Check for transition phrases
        transition_score = 0
        for phrase in transition_phrases:
            if phrase in curr_text:
                transition_score += 0.5  # Add weight for transition phrases
        
        # Calculate total score (normalized)
        max_possible_score = len(action_verbs) + 0.5 * len(transition_phrases)
        step_change_score = min((new_actions + transition_score) / max_possible_score, 1.0)
        
        return step_change_score
    
    def _get_view_names(self, take_dir):
        """
        Map action indices to camera names for the current take,
        detecting available cameras dynamically
        """
        # Get the clips directory for this take
        take = None
        for t in self.takes:
            if t["take_dir"] == take_dir:
                take = t
                break
        
        if take is None:
            # Default mapping if take not found
            return {i: f"cam{i+1:02d}" for i in range(self.max_views)}
        
        # Use the first clip to determine available cameras
        if take["clip_dirs"]:
            first_clip_dir = os.path.join(take["clips_dir"], take["clip_dirs"][0])
            if os.path.exists(first_clip_dir):
                # Find all camera files (both exo and ego)
                cam_files = [f for f in os.listdir(first_clip_dir) 
                             if f.endswith('.mp4')]
                
                # Extract camera names (removing .mp4 extension)
                cam_names = [os.path.splitext(f)[0] for f in cam_files]
                
                # Sort exo cameras first, then ego cameras
                exo_cams = sorted([c for c in cam_names if c.startswith('cam')])
                ego_cams = sorted([c for c in cam_names if 'aria' in c.lower()])
                
                # Combine and create mapping
                all_cams = exo_cams + ego_cams
                
                # Print camera information for debugging
                print(f"Take {take_dir} has {len(all_cams)} cameras: {', '.join(all_cams)}")
                
                # Create mapping up to max_views or actual camera count
                view_names = {}
                for i in range(min(self.max_views, len(all_cams))):
                    view_names[i] = all_cams[i]
                
                # If we have fewer cameras than max_views, pad with dummy names
                for i in range(len(all_cams), self.max_views):
                    view_names[i] = f"dummy_cam{i+1}"
                
                return view_names
        
        # Default mapping if no cameras found
        return {i: f"cam{i+1:02d}" for i in range(self.max_views)}