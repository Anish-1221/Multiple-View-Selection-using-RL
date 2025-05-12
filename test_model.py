import os
import numpy as np
import argparse
import torch
import cv2
import json
from stable_baselines3 import PPO, A2C, DQN
from multiview_env import MultiViewEnv
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def test_on_specific_take(model_path, dataset_dir, take_name, output_dir, feature_dir, feature_dim=4096, fps=10):
    """
    Test the model on a specific take and create a visualization video
    showing the selected views.
    
    Args:
        model_path: Path to the trained model
        dataset_dir: Path to the dataset directory
        take_name: Name of the take to test on
        output_dir: Directory to save the output video
        feature_dir: Directory containing pre-extracted features
        feature_dim: Dimension of feature vectors (defaults to 4096)
        fps: Frames per second for the output video
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine the algorithm from the model path
    if "ppo" in os.path.basename(model_path).lower():
        algo = "PPO"
        model = PPO.load(model_path)
    elif "a2c" in os.path.basename(model_path).lower():
        algo = "A2C"
        model = A2C.load(model_path)
    elif "dqn" in os.path.basename(model_path).lower():
        algo = "DQN"
        model = DQN.load(model_path)
    else:
        # Default to PPO if can't determine
        algo = "Unknown"
        model = PPO.load(model_path)
    
    print(f"Testing {algo} model: {model_path}")
    
    # Try to load reward parameters if available
    reward_params_path = os.path.join(os.path.dirname(model_path), "reward_params.json")
    best_params_path = os.path.join(os.path.dirname(model_path), "best_params.json")
    
    if os.path.exists(reward_params_path):
        with open(reward_params_path, 'r') as f:
            reward_params = json.load(f)
        print(f"Loaded reward parameters from {reward_params_path}")
    elif os.path.exists(best_params_path):
        with open(best_params_path, 'r') as f:
            reward_params = json.load(f)
        print(f"Loaded best parameters from {best_params_path}")
    else:
        # Use default parameters matching your current settings
        reward_params = {
            "informativeness_weight": 1.5,       
            "narration_activity_weight": 0.4,    
            "switch_penalty": -0.1,              
            "visual_similarity_weight": 0.1,     
            "narrative_context_weight": 0.8,     
            "exploration_bonus_weight": 1.0,     
            "switch_incentive_weight": 2.0       
        }
        print("Using default reward parameters")
    
    # Set up environment config with the same feature dimension used for training
    env_config = {
        "dataset_dir": dataset_dir,
        "feature_extractor": None,
        "feature_dir": feature_dir,
        "max_views": 6,
        "feature_dim": feature_dim,  # Use the specified feature dimension
        "episode_length": 1000,      # Large value to process all clips
        "reward_params": reward_params,
        "auto_adjust_feature_dim": True  # Enable automatic detection
    }
    
    # Create the environment
    env = MultiViewEnv(env_config)
    print(f"Environment initialized with feature dimension: {env.feature_dim}")
    
    # Check if take exists in environment takes
    take_idx = -1
    for i, take in enumerate(env.takes):
        if take["take_dir"] == take_name:
            take_idx = i
            break
    
    if take_idx == -1:
        print(f"Take {take_name} not found in environment takes. Make sure it has features in {feature_dir}")
        return
    
    # Set the environment to the specified take
    env.current_take_idx = take_idx
    take = env.takes[take_idx]
    
    # Find video directory for this take
    take_path = os.path.join(dataset_dir, "ego4d_256", take_name)
    
    # Check if take exists
    clips_dir = None
    
    # Try to find the clips directory in the common locations
    potential_clips_dirs = [
        os.path.join(take_path, "clips"),
        os.path.join(take_path, "frame_aligned_videos", "downscaled", "448", "clips"),
        os.path.join(feature_dir, take_name, "clips")
    ]
    
    for potential_dir in potential_clips_dirs:
        if os.path.exists(potential_dir) and os.path.isdir(potential_dir):
            clips_dir = potential_dir
            print(f"Found clips directory at: {clips_dir}")
            break
    
    if not clips_dir:
        print(f"Could not find clips directory for take {take_name}")
        return
    
    # Get all clip directories directly from the filesystem as a fallback
    all_clip_dirs = sorted([
        d for d in os.listdir(clips_dir)
        if os.path.isdir(os.path.join(clips_dir, d)) and d.startswith("clip_")
    ])
    print(f"Take has {len(take['clip_dirs'])} clips in environment")
    print(f"Found {len(all_clip_dirs)} clip directories on filesystem")
    
    # Decide which list to use (environment's or filesystem's)
    use_filesystem_clips = len(all_clip_dirs) > len(take['clip_dirs'])
    clip_list = all_clip_dirs if use_filesystem_clips else take['clip_dirs']
    
    if use_filesystem_clips:
        print(f"Using filesystem clips instead of environment clips")
    
    # Reset environment to start at the beginning
    env.current_clip_idx = 0
    env.step_count = 0
    obs, _ = env.reset()
    
    # Process each clip and record the selected views
    selected_views = []
    clips = []
    
    # Enhanced metrics tracking
    metrics = {
        "accuracy": 0,  # How often model selects the best view
        "total_clips": 0,
        "correct_views": 0,
        "total_reward": 0,
        "view_switches": 0,
        "clip_metrics": [],  # Per-clip metrics
        "view_distribution": {},  # Distribution of selected views
        "switches_distribution": {},  # When switches happen
    }
    
    print(f"Processing take {take_name} with {len(clip_list)} clips")
    
    # Function to check if a file is a standard camera
    def is_standard_camera(filename):
        return filename.startswith("cam")
    
    # Function to check if a file is an allowed Aria camera
    def is_allowed_aria_camera(filename):
        return filename.startswith("aria") and "_214-1" in filename
    
    prev_view = None
    
    for clip_idx, clip_dir_name in enumerate(tqdm(clip_list)):
        try:
            # Get the current clip directory
            clip_dir = os.path.join(clips_dir, clip_dir_name)
            
            # Skip if the directory doesn't exist
            if not os.path.isdir(clip_dir):
                print(f"Directory not found: {clip_dir}, skipping...")
                continue
                
            # Get all camera video files for this clip
            all_video_files = [f for f in os.listdir(clip_dir) if f.endswith('.mp4')]
            
            if not all_video_files:
                print(f"No camera files found in {clip_dir}, skipping...")
                continue
            
            # Filter to only include standard cameras and allowed Aria cameras
            standard_cameras = sorted([f for f in all_video_files if is_standard_camera(f)])
            aria_cameras = sorted([f for f in all_video_files if is_allowed_aria_camera(f)])
            
            # Combine in the right order: standard cameras first, then allowed Aria cameras
            sorted_video_files = standard_cameras + aria_cameras
            
            # If no allowed cameras found, skip this clip
            if not sorted_video_files:
                print(f"No allowed cameras found in clip {clip_idx}, skipping...")
                continue
            
            # Display available cameras for this clip
            print(f"\nClip {clip_idx} ({clip_dir_name}) allowed cameras:")
            for i, file in enumerate(sorted_video_files):
                print(f"  {i}: {file}")
            
            # If using filesystem clips, need to manually set the environment's clip index
            if use_filesystem_clips:
                env.current_clip_idx = clip_idx
            
            # Get model's action (selected view)
            try:
                action, _ = model.predict(obs, deterministic=True)
                raw_selected_view = int(action.item())  # Convert numpy array to int
                selected_views.append(raw_selected_view)
                print(f"Model selected view index: {raw_selected_view}")
                
                # Track view distribution
                metrics["view_distribution"][raw_selected_view] = metrics["view_distribution"].get(raw_selected_view, 0) + 1
                
                # Track view switches
                if prev_view is not None and prev_view != raw_selected_view:
                    metrics["view_switches"] += 1
                    switch_key = f"{prev_view}->{raw_selected_view}"
                    metrics["switches_distribution"][switch_key] = metrics["switches_distribution"].get(switch_key, 0) + 1
                
                prev_view = raw_selected_view
                
                # Map the model's selected view to our available cameras
                # If the selected view is out of range, use view 0 (first camera)
                if raw_selected_view < len(sorted_video_files):
                    actual_view = raw_selected_view
                else:
                    actual_view = 0
                
                selected_file = sorted_video_files[actual_view]
                video_path = os.path.join(clip_dir, selected_file)
                clips.append((clip_idx, video_path, actual_view, selected_file))
                print(f"Added clip {clip_idx} with view {actual_view} (file: {selected_file})")
                
                # Get the best view from narrations
                best_view = -1
                clip_str = str(clip_idx)
                
                if "clip_narrations" in take["narrations"] and clip_str in take["narrations"]["clip_narrations"]:
                    clip_narrations = take["narrations"]["clip_narrations"][clip_str]
                    view_names = env._get_view_names(take["take_dir"])
                    
                    # Find the most common best_camera in narrations
                    camera_counts = {}
                    for narration in clip_narrations:
                        best_camera = narration.get("best_camera", "")
                        if best_camera:
                            camera_counts[best_camera] = camera_counts.get(best_camera, 0) + 1
                    
                    if camera_counts:
                        best_camera = max(camera_counts.items(), key=lambda x: x[1])[0]
                        # Find the action index for this camera
                        for action_idx, camera in view_names.items():
                            if camera == best_camera:
                                best_view = action_idx
                                break
                
                # Calculate reward and accuracy metrics
                if best_view >= 0:
                    metrics["total_clips"] += 1
                    if raw_selected_view == best_view:
                        metrics["correct_views"] += 1
                
                # Take a step in the environment to get reward
                next_obs, reward, terminated, truncated, _ = env.step(raw_selected_view)
                metrics["total_reward"] += reward
                
                # Store per-clip metrics
                clip_metric = {
                    "clip_idx": clip_idx,
                    "selected_view": raw_selected_view,
                    "best_view": best_view,
                    "reward": float(reward),
                    "correct": (raw_selected_view == best_view) if best_view >= 0 else None,
                    "clip_name": clip_dir_name,
                    "selected_file": selected_file
                }
                metrics["clip_metrics"].append(clip_metric)
                
                # Update observation for next iteration
                obs = next_obs
                
                # If using filesystem clips, manually update the observation for next clip
                if use_filesystem_clips and clip_idx < len(clip_list) - 1:
                    try:
                        # Manually set clip index for the next clip
                        next_clip_idx = clip_idx + 1
                        next_clip_dir_name = clip_list[next_clip_idx]
                        
                        # Get features for the next clip
                        features, valid_mask = env._get_features(take_idx, next_clip_idx)
                        
                        # Update observation
                        obs = {
                            "features": features,
                            "current_view": np.array([raw_selected_view], dtype=np.int32),
                            "time_step": np.array([next_clip_idx], dtype=np.int32),
                            "valid_views": valid_mask
                        }
                    except Exception as e:
                        print(f"Error preparing next observation: {e}, stopping")
                        break
            
            except Exception as e:
                print(f"Error predicting action for clip {clip_idx}: {e}, skipping...")
                continue
            
        except Exception as e:
            print(f"Unexpected error processing clip {clip_idx}: {e}")
            # Continue with the next clip instead of breaking
    
    # Calculate final metrics
    if metrics["total_clips"] > 0:
        metrics["accuracy"] = metrics["correct_views"] / metrics["total_clips"]
    else:
        metrics["accuracy"] = 0
        
    metrics["average_reward"] = metrics["total_reward"] / len(clip_list) if clip_list else 0
    metrics["switch_rate"] = metrics["view_switches"] / (len(clip_list) - 1) if len(clip_list) > 1 else 0
    
    # Normalize view distribution
    total_selections = sum(metrics["view_distribution"].values())
    if total_selections > 0:
        for view, count in metrics["view_distribution"].items():
            metrics["view_distribution"][view] = count / total_selections
    
    print(f"Selected {len(selected_views)} views")
    print(f"Collected {len(clips)} clips for video creation")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f"{take_name}_{algo}_metrics.json")
    with open(metrics_path, 'w') as f:
        # json.dump(metrics, f, indent=2)
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    
    # Generate metrics visualizations
    visualize_metrics(metrics, os.path.join(output_dir, f"{take_name}_{algo}_metrics"))
    
    # Create a video from the selected views
    create_selection_video(clips, os.path.join(output_dir, f"{take_name}_selection.mp4"), fps)
    
    # Save the selected views to a text file with more detailed information
    with open(os.path.join(output_dir, f"{take_name}_selected_views.txt"), 'w') as f:
        f.write(f"Take: {take_name}\n")
        f.write(f"Algorithm: {algo}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Feature dimension: {env.feature_dim}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct_views']}/{metrics['total_clips']})\n")
        f.write(f"Average Reward: {metrics['average_reward']:.4f}\n")
        f.write(f"View Switch Rate: {metrics['switch_rate']:.4f}\n\n")
        
        f.write("View Selection Distribution:\n")
        for view, percentage in sorted(metrics["view_distribution"].items()):
            f.write(f"  View {view}: {percentage:.2%}\n")
        
        f.write("\nDetailed Clip Selections:\n")
        for i, (clip_idx, _, view, filename) in enumerate(clips):
            f.write(f"Clip {clip_idx}: Selected view {view} ({filename})\n")

    # Test a specific Aria video file if selected by the model
    aria_clips = [clip for clip in clips if "aria" in clip[3]]
    if aria_clips:
        print("\nTesting a sample Aria video file:")
        sample_clip = aria_clips[0]
        test_aria_video(sample_clip[1])  # Test the video path
    
    print(f"Results saved to {output_dir}")
    return metrics

def create_selection_video(clips, output_path, fps=10):
    """Create a video from the selected clips with better handling of different camera types."""
    if not clips:
        print("No clips to create video")
        return
    
    # Initialize video writer
    writer = None
    frame_width, frame_height = 640, 480  # Default size if we can't determine from videos
    successful_clips = 0
    failed_clips = 0
    aria_clips = 0
    
    # First pass - determine video dimensions from standard cameras
    for clip_idx, video_path, selected_view, filename in clips:
        if "aria" not in filename.lower():  # Only use standard cameras for size determination
            try:
                if os.path.exists(video_path):
                    cap = cv2.VideoCapture(video_path)
                    if cap.isOpened():
                        ret, test_frame = cap.read()
                        if ret:
                            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            cap.release()
                            print(f"Using dimensions from standard camera: {frame_width}x{frame_height}")
                            break
                        cap.release()
            except Exception as e:
                print(f"Error checking dimensions for {video_path}: {e}")
    
    # Initialize writer with determined size
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height + 50)  # Add space for text
    )
    
    print(f"Creating video with dimensions: {frame_width}x{frame_height + 50}")
    
    # Second pass - actually create the video
    for clip_idx, video_path, selected_view, filename in tqdm(clips, desc="Creating video"):
        is_aria = "aria" in filename.lower()
        if is_aria:
            aria_clips += 1
            
        # Check if file exists
        if not os.path.exists(video_path):
            print(f"Error: File not found: {video_path}")
            failed_clips += 1
            
            # Create a blank frame instead
            blank_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            cv2.putText(blank_frame, f"MISSING FILE: {os.path.basename(video_path)}", 
                       (50, frame_height//2), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (0, 0, 255), 2, cv2.LINE_AA)
            
            info_frame = np.zeros((50, frame_width, 3), dtype=np.uint8)
            cv2.putText(info_frame, f"Clip: {clip_idx}, View: {selected_view} ({filename}) - FILE MISSING", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
            
            combined_frame = np.vstack((blank_frame, info_frame))
            writer.write(combined_frame)
            continue
            
        try:
            # Open the video
            cap = cv2.VideoCapture(video_path)
            
            # Check if the video opened successfully
            if not cap.isOpened():
                print(f"Error opening video {video_path}")
                failed_clips += 1
                
                # Create a blank frame with error message
                blank_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                cv2.putText(blank_frame, f"ERROR OPENING: {os.path.basename(video_path)}", 
                           (50, frame_height//2), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, (0, 0, 255), 2, cv2.LINE_AA)
                
                info_frame = np.zeros((50, frame_width, 3), dtype=np.uint8)
                cv2.putText(info_frame, f"Clip: {clip_idx}, View: {selected_view} ({filename}) - OPEN ERROR", 
                           (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
                
                combined_frame = np.vstack((blank_frame, info_frame))
                writer.write(combined_frame)
                continue
            
            # Verify we can read frames
            ret, test_frame = cap.read()
            if not ret:
                print(f"Cannot read frames from {video_path}")
                cap.release()
                failed_clips += 1
                
                # Create a blank frame with error message
                blank_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                cv2.putText(blank_frame, f"NO FRAMES: {os.path.basename(video_path)}", 
                           (50, frame_height//2), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, (0, 0, 255), 2, cv2.LINE_AA)
                
                info_frame = np.zeros((50, frame_width, 3), dtype=np.uint8)
                cv2.putText(info_frame, f"Clip: {clip_idx}, View: {selected_view} ({filename}) - NO FRAMES", 
                           (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
                
                combined_frame = np.vstack((blank_frame, info_frame))
                writer.write(combined_frame)
                continue
                
            # Reset to beginning of video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Get video properties
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Read and write frames
            frame_count = 0
            max_frames = 300  # Limit frames per clip 
            
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Handle different aspect ratios
                if video_width != frame_width or video_height != frame_height:
                    # Resize frame to match our video writer dimensions
                    frame = cv2.resize(frame, (frame_width, frame_height))
                
                # Add text overlay with clip and selected view information
                info_frame = np.zeros((50, frame_width, 3), dtype=np.uint8)
                camera_type = "ARIA CAMERA" if is_aria else "STANDARD CAMERA"
                text = f"Clip: {clip_idx}, View: {selected_view} ({filename}) - {camera_type}"
                
                cv2.putText(
                    info_frame, 
                    text, 
                    (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255) if not is_aria else (0, 255, 255),  # Yellow for Aria
                    1, 
                    cv2.LINE_AA
                )
                
                # Add a border for Aria cameras to make them more visible
                if is_aria:
                    frame = cv2.rectangle(frame, (0, 0), (frame_width-1, frame_height-1), (0, 255, 255), 3)
                
                # Combine frame and info
                combined_frame = np.vstack((frame, info_frame))
                
                # Write the frame
                writer.write(combined_frame)
                frame_count += 1
                
            # Release the video capture
            cap.release()
            successful_clips += 1
            
        except Exception as e:
            print(f"Error processing clip {clip_idx} ({filename}): {e}")
            failed_clips += 1
            
            # Create a blank frame with error message
            blank_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            cv2.putText(blank_frame, f"PROCESSING ERROR: {os.path.basename(video_path)}", 
                       (50, frame_height//2), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (0, 0, 255), 2, cv2.LINE_AA)
            
            info_frame = np.zeros((50, frame_width, 3), dtype=np.uint8)
            cv2.putText(info_frame, f"Clip: {clip_idx}, View: {selected_view} ({filename}) - ERROR: {str(e)[:30]}", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            
            combined_frame = np.vstack((blank_frame, info_frame))
            writer.write(combined_frame)
    
    # Release the video writer
    if writer is not None:
        writer.release()
    
    print(f"Video creation summary:")
    print(f"  Total clips: {len(clips)}")
    print(f"  Successful clips: {successful_clips}")
    print(f"  Failed clips: {failed_clips}")
    print(f"  Aria camera clips: {aria_clips}")
    print(f"Video saved to {output_path}")

def test_aria_video(video_path):
    """Test if an Aria video can be read properly with OpenCV."""
    print(f"Testing Aria video: {video_path}")
    
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"  Error: File not found")
        return False
        
    # Try to open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Error: Could not open video file")
        return False
        
    # Try to read frames
    ret, frame = cap.read()
    if not ret:
        print(f"  Error: Could not read frames")
        cap.release()
        return False
        
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Success: Video dimensions: {width}x{height}, {frames} frames")
    cap.release()
    return True

def visualize_metrics(metrics, output_prefix):
    """Create visualizations of the metrics"""
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    # 1. View distribution pie chart
    plt.figure(figsize=(10, 8))
    labels = [f"View {view}" for view in sorted(metrics["view_distribution"].keys())]
    sizes = [metrics["view_distribution"][int(view.split()[-1])] for view in labels]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Camera Selection Distribution')
    plt.savefig(f"{output_prefix}_view_distribution.png")
    plt.close()
    
    # 2. Reward per clip bar chart
    if metrics["clip_metrics"]:
        clip_indices = [m["clip_idx"] for m in metrics["clip_metrics"]]
        rewards = [m["reward"] for m in metrics["clip_metrics"]]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(clip_indices, rewards)
        
        # Color bars based on correct/incorrect views if best_view is available
        for i, metric in enumerate(metrics["clip_metrics"]):
            if metric["correct"] is True:
                bars[i].set_color('green')
            elif metric["correct"] is False:
                bars[i].set_color('red')
                
        plt.xlabel('Clip Index')
        plt.ylabel('Reward')
        plt.title('Reward per Clip')
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_rewards.png")
        plt.close()
    
    # 3. View switches heatmap (if enough data)
    if len(metrics["switches_distribution"]) > 1:
        # Create a matrix of view switches
        num_views = max([max(int(k.split('->')[0]), int(k.split('->')[1])) for k in metrics["switches_distribution"].keys()]) + 1
        switch_matrix = np.zeros((num_views, num_views))
        
        for switch, count in metrics["switches_distribution"].items():
            source, target = switch.split('->')
            switch_matrix[int(source), int(target)] = count
        
        plt.figure(figsize=(10, 8))
        plt.imshow(switch_matrix, cmap='viridis')
        plt.colorbar(label='Switch Count')
        plt.title('View Switching Patterns')
        plt.xlabel('To View')
        plt.ylabel('From View')
        
        # Add labels
        plt.xticks(range(num_views), [f"View {i}" for i in range(num_views)])
        plt.yticks(range(num_views), [f"View {i}" for i in range(num_views)])
        
        # Add counts as text
        for i in range(num_views):
            for j in range(num_views):
                if switch_matrix[i, j] > 0:
                    plt.text(j, i, int(switch_matrix[i, j]), ha="center", va="center", color="w")
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_switch_patterns.png")
        plt.close()
    
    # 4. Create a summary plot with key metrics
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['accuracy', 'average_reward', 'switch_rate']
    values = [metrics[m] for m in metrics_to_plot]
    labels = ['Accuracy', 'Avg Reward', 'Switch Rate']
    
    bars = plt.bar(labels, values)
    
    # Add value labels on the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.title('Key Performance Metrics')
    plt.ylim(0, max(1.0, max(values) * 1.2))  # Set y limit based on max value
    plt.savefig(f"{output_prefix}_summary.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the model on a specific take")
    parser.add_argument(
        "--model-path",
        default="output_dynamic/multiview_ppo_final",
        help="Path to the trained model"
    )
    parser.add_argument(
        "--dataset-dir",
        default="/Users/anish/Documents/UT Austin/Spring 2025/Courses/Visual Recognition/Project/Testing/mydata/dataset",
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--take-name",
        required=True,
        help="Name of the take to test on"
    )
    parser.add_argument(
        "--output-dir",
        default="test_results",
        help="Directory to save the output video"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for the output video"
    )
    parser.add_argument(
        "--feature-dir",
        default="/Users/anish/Documents/UT Austin/Spring 2025/Courses/Visual Recognition/Project/Testing/multiview_rl/features_new",
        help="Directory containing pre-extracted features"
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=4096,
        help="Dimension of feature vectors (default: 4096)"
    )
    
    args = parser.parse_args()
    
    # Test the model on the specified take
    test_on_specific_take(
        args.model_path,
        args.dataset_dir,
        args.take_name,
        args.output_dir,
        args.feature_dir,
        args.feature_dim,
        args.fps
    )