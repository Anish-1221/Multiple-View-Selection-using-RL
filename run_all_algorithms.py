#!/usr/bin/env python3
import os
import subprocess
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def run_algorithm(algo, dataset_dir, feature_dir, output_dir, total_timesteps, seed, train=True):
    """Run a specific algorithm for training or testing"""
    
    # Base command with common arguments
    cmd = [
        "python", "train.py",
        "--algorithm", algo,
        "--dataset-dir", dataset_dir,
        "--feature-dir", feature_dir,
        "--output-dir", output_dir,
        "--total-timesteps", str(total_timesteps),
        "--seed", str(seed),
        "--random-steps", "10000",
        "--exploration-fraction", "0.3"
    ]
    
    # Add train/eval mode
    if train:
        cmd.extend(["--run-type", "train"])
    else:
        cmd.extend(["--run-type", "eval"])
        algo_dir = os.path.join(output_dir, algo.lower())
        model_path = os.path.join(algo_dir, f"multiview_{algo.lower()}_final.zip")
        cmd.extend(["--model-path", model_path])
    
    # Print and run the command
    print(f"\n{'='*50}")
    print(f"Running {algo} {'training' if train else 'evaluation'}...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}\n")
    
    # Execute the command
    result = subprocess.run(cmd)
    
    # Check if the command was successful
    if result.returncode != 0:
        print(f"Error running {algo} {'training' if train else 'evaluation'}")
        return False
    return True

def run_test_on_specific_take(algo, dataset_dir, feature_dir, output_dir, take_name):
    """Run testing for a specific take"""
    
    # Get the specific model path
    algo_dir = os.path.join(output_dir, algo.lower())
    model_path = os.path.join(algo_dir, f"multiview_{algo.lower()}_final")
    
    # Create test results directory
    test_dir = os.path.join(algo_dir, "test_results", take_name)
    os.makedirs(test_dir, exist_ok=True)
    
    # Construct command for testing
    cmd = [
        "python", "test_model.py",
        "--model-path", model_path,
        "--dataset-dir", dataset_dir,
        "--take-name", take_name,
        "--output-dir", test_dir,
        "--fps", "30",
        "--feature-dir", feature_dir
    ]
    
    # Print and run the command
    print(f"\n{'='*50}")
    print(f"Running {algo} testing on {take_name}...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}\n")
    
    # Execute the command
    result = subprocess.run(cmd)
    
    # Check if the command was successful
    if result.returncode != 0:
        print(f"Error running {algo} testing on {take_name}")
        return False
    return True

def collect_metrics(output_dir, algos):
    """Collect and combine metrics from all algorithms"""
    
    all_metrics = {}
    
    for algo in algos:
        algo_lower = algo.lower()
        eval_dir = os.path.join(output_dir, algo_lower, "evaluation")
        
        # Check if detailed evaluation exists
        eval_file = os.path.join(eval_dir, "detailed_evaluation.json")
        if os.path.exists(eval_file):
            with open(eval_file, 'r') as f:
                data = json.load(f)
                all_metrics[algo] = data["metrics"]
        else:
            print(f"Warning: Metrics for {algo} not found at {eval_file}")
    
    return all_metrics

def generate_comparison_report(output_dir, all_metrics, algos):
    """Generate a comparative report of all algorithms"""
    
    # Create comparison directory
    comparison_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # 1. Create comparative metrics table
    metrics_df = pd.DataFrame(columns=["Metric"] + algos)
    
    # Key metrics to compare
    key_metrics = ["accuracy", "average_reward", "average_switches_per_episode"]
    metric_names = {
        "accuracy": "View Selection Accuracy",
        "average_reward": "Average Episode Reward",
        "average_switches_per_episode": "Avg View Switches per Episode"
    }
    
    # Fill the dataframe
    for metric in key_metrics:
        row = [metric_names.get(metric, metric)]
        for algo in algos:
            if algo in all_metrics and metric in all_metrics[algo]:
                row.append(f"{all_metrics[algo][metric]:.4f}")
            else:
                row.append("N/A")
        metrics_df.loc[len(metrics_df)] = row
    
    # Save the comparison table
    metrics_df.to_csv(os.path.join(comparison_dir, "algorithm_comparison.csv"), index=False)
    
    # 2. Create comparison plots
    # Camera distribution comparison
    plt.figure(figsize=(15, 10))
    
    # Number of algorithms to compare
    n_algos = len(algos)
    
    # Setup subplot grid
    for i, algo in enumerate(algos):
        if algo not in all_metrics:
            continue
            
        plt.subplot(n_algos, 1, i+1)
        
        if "camera_distribution" in all_metrics[algo]:
            cam_dist = all_metrics[algo]["camera_distribution"]
            cameras = sorted(cam_dist.keys())
            values = [cam_dist[str(c)] if str(c) in cam_dist else cam_dist[c] if c in cam_dist else 0 for c in cameras]
            
            plt.bar(cameras, values)
            plt.title(f"{algo} Camera Distribution")
            plt.xlabel("Camera")
            plt.ylabel("Selection Frequency")
            plt.ylim(0, 1.0)  # Standardize y-axis
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "camera_distribution_comparison.png"))
    plt.close()
    
    # 3. Create comparison metrics plots
    for metric in key_metrics:
        plt.figure(figsize=(10, 6))
        
        values = []
        labels = []
        
        for algo in algos:
            if algo in all_metrics and metric in all_metrics[algo]:
                values.append(all_metrics[algo][metric])
                labels.append(algo)
        
        if values:
            bars = plt.bar(labels, values)
            
            # Add value labels on the bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom')
            
            plt.title(f"Comparison of {metric_names.get(metric, metric)}")
            plt.ylabel(metric_names.get(metric, metric))
            plt.savefig(os.path.join(comparison_dir, f"{metric}_comparison.png"))
            plt.close()
    
    # 4. Create summary HTML report
    html_content = f"""
    <html>
    <head>
        <title>Multi-View RL Algorithm Comparison</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333366; }}
            table {{ border-collapse: collapse; width: 80%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .img-container {{ margin: 20px 0; }}
            .timestamp {{ color: #666; font-style: italic; }}
        </style>
    </head>
    <body>
        <h1>Multi-View RL Algorithm Comparison Report</h1>
        <p class="timestamp">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>Performance Metrics Comparison</h2>
        <table>
            <tr>
                <th>Metric</th>
                {' '.join([f'<th>{algo}</th>' for algo in algos])}
            </tr>
    """
    
    # Add table rows
    for _, row in metrics_df.iterrows():
        html_content += "<tr>"
        for cell in row:
            html_content += f"<td>{cell}</td>"
        html_content += "</tr>\n"
    
    html_content += """
        </table>
        
        <h2>Camera Distribution Comparison</h2>
        <div class="img-container">
            <img src="camera_distribution_comparison.png" alt="Camera Distribution Comparison" style="max-width: 100%;">
        </div>
    """
    
    # Add metric comparison images
    for metric in key_metrics:
        metric_img = f"{metric}_comparison.png"
        if os.path.exists(os.path.join(comparison_dir, metric_img)):
            html_content += f"""
            <h2>{metric_names.get(metric, metric)} Comparison</h2>
            <div class="img-container">
                <img src="{metric_img}" alt="{metric} Comparison" style="max-width: 100%;">
            </div>
            """
    
    html_content += """
    </body>
    </html>
    """
    
    # Save the HTML report
    with open(os.path.join(comparison_dir, "comparison_report.html"), 'w') as f:
        f.write(html_content)
    
    print(f"\nComparison report generated at {os.path.join(comparison_dir, 'comparison_report.html')}")
    return comparison_dir

def main():
    parser = argparse.ArgumentParser(description="Run multiple RL algorithms and compare their results")
    
    parser.add_argument(
        "--dataset-dir",
        default="/Users/anish/Documents/UT Austin/Spring 2025/Courses/Visual Recognition/Project/Testing/mydata/dataset",
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--feature-dir",
        default="/Users/anish/Documents/UT Austin/Spring 2025/Courses/Visual Recognition/Project/Testing/multiview_rl/features_new",
        help="Path to the features directory"
    )
    parser.add_argument(
        "--output-dir",
        default="output_all_algos",
        help="Directory to save all outputs"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=150000,
        help="Total timesteps for training each algorithm"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--test-take",
        default="minnesota_cooking_060_2",
        help="Take to use for testing"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and only run evaluation and testing"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation and only run testing"
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["PPO", "A2C", "DQN"],
        choices=["PPO", "A2C", "DQN"],
        help="Algorithms to run (default: all three)"
    )
    
    args = parser.parse_args()
    
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Algorithms to run
    algos = args.algorithms
    print(f"Will run the following algorithms: {', '.join(algos)}")
    
    # Step 1: Training
    if not args.skip_training:
        print("\nStarting training phase...")
        for algo in algos:
            success = run_algorithm(
                algo=algo,
                dataset_dir=args.dataset_dir,
                feature_dir=args.feature_dir,
                output_dir=args.output_dir, 
                total_timesteps=args.total_timesteps,
                seed=args.seed,
                train=True
            )
            
            if not success:
                print(f"Warning: Training for {algo} did not complete successfully")
    else:
        print("\nSkipping training phase as requested")
    
    # Step 2: Evaluation (if not skipped)
    if not args.skip_evaluation:
        print("\nStarting evaluation phase...")
        for algo in algos:
            success = run_algorithm(
                algo=algo,
                dataset_dir=args.dataset_dir,
                feature_dir=args.feature_dir,
                output_dir=args.output_dir, 
                total_timesteps=args.total_timesteps,
                seed=args.seed,
                train=False
            )
            
            if not success:
                print(f"Warning: Evaluation for {algo} did not complete successfully")
    else:
        print("\nSkipping evaluation phase as requested")
    
    # Step 3: Testing on specific take
    print(f"\nStarting testing phase on take: {args.test_take}")
    for algo in algos:
        success = run_test_on_specific_take(
            algo=algo,
            dataset_dir=args.dataset_dir,
            feature_dir=args.feature_dir,
            output_dir=args.output_dir,
            take_name=args.test_take
        )
        
        if not success:
            print(f"Warning: Testing for {algo} on take {args.test_take} did not complete successfully")
    
    # Step 4: Collect metrics and generate comparison report
    print("\nCollecting metrics and generating comparison report...")
    all_metrics = collect_metrics(args.output_dir, algos)
    
    if all_metrics:
        comparison_dir = generate_comparison_report(args.output_dir, all_metrics, algos)
        print(f"Comparison report generated in {comparison_dir}")
    else:
        print("No metrics found to generate comparison report")
    
    print("\nAll tasks completed!")

if __name__ == "__main__":
    main()