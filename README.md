# Multi-View RL Algorithm Comparison

This repository contains code for training and comparing multiple reinforcement learning algorithms for multi-view selection in videos. The system supports PPO, A2C, and DQN algorithms.

## Project Overview

This project develops a reinforcement learning approach for optimizing camera view selection in multi-view videos. By implementing a composite reward function that balances informativeness with transition quality, our system creates sequences that maintain instructional clarity while ensuring visual continuity.

The key innovation is moving beyond frame-by-frame selection to consider the temporal relationships between consecutive view choices, addressing the challenge of jarring transitions that diminish viewer experience. Our reinforcement learning framework optimizes entire sequences of views rather than isolated frames.

## Documentation

A comprehensive research report detailing the approach, methodology, and results is included in this repository. The report provides:

- Theoretical background and related work
- Detailed explanation of the composite reward function
- Analysis of algorithm performance across multiple metrics
- Ablation studies showing the contribution of individual reward components
- Qualitative analysis of generated view sequences
- Future work directions

For a deeper understanding of the research and technical details, please refer to the report document in the repository.

## Research Context

Recent advances have addressed optimal view selection, but temporal coherence remains a significant challenge. Previous approaches used natural language descriptions to identify the most informative camera views but didn't account for the temporal relationships between consecutive view choices.

Our research extends this foundation through a reinforcement learning framework designed to optimize not just individual view selections but entire sequences of views. This approach addresses the limitations of isolated timestep optimization by considering the sequential nature of video editing decisions, similar to how professional editors balance information density with visual flow.

## Technical Approach

Our approach formulates multi-view selection as a Markov Decision Process (MDP) with a composite reward function that includes:

- **Informativeness Reward**: Rewards selection of views that align well with narrative description
- **Strategic Switching Component**: Encourages view changes at appropriate moments
- **Visual Similarity Consideration**: Modulates switching based on visual feature similarity between views
- **Narrative Context Modulation**: Reduces switching penalties during narrative transitions
- **Exploration Bonus**: Encourages exploration of all available views
- **Switch Incentive**: Prevents fixation on a single view

We implemented three reinforcement learning algorithms (PPO, A2C, and DQN), with PPO showing the most balanced performance in selecting informative views while maintaining visual coherence.

## Key Results

- **PPO vs Other Algorithms**: PPO demonstrates a more balanced distribution across camera views, making strategic transitions at narrative boundaries while maintaining visual coherence
- **View Selection Strategy**: Our approach selects informative views that capture essential actions while making transitions that preserve visual continuity
- **Narrative Alignment**: The system times transitions with narrative progression, maintaining stable camera selections during continuous actions but switching views at natural narrative boundaries

## Setup

1. Make sure you have all the required dependencies installed:
   ```bash
   pip install stable-baselines3 gymnasium opencv-python numpy pandas matplotlib tqdm torch
   ```

2. Ensure your directory structure contains the necessary files:
   - multiview_env.py (the environment)
   - train.py (modified to support multiple algorithms)
   - test_model.py (modified with enhanced metrics)
   - run_all_algorithms.py (to run and compare all algorithms)

3. Make sure your dataset is correctly structured:
   ```
   /path/to/dataset/
   └── ego4. Make sure your feature directory is correctly structured:
   ```
   /path/to/features/
   └── minnesota_cooking_060_2/
       └── ...
   ```

## Running the System

### Option 1: Run All Algorithms

The easiest way to run the system is to use the `run_all_algorithms.py` script, which will train, evaluate, and test all three algorithms:

```bash
python run_all_algorithms.py \
  --dataset-dir /path/to/dataset \
  --feature-dir /path/to/features \
  --output-dir output_all_algos \
  --total-timesteps 150000 \
  --test-take minnesota_cooking_060_2
```

Options:
- `--algorithms PPO A2C DQN` - Specify which algorithms to run (default: all three)
- `--skip-training` - Skip the training phase and only run evaluation and testing
- `--skip-evaluation` - Skip the evaluation phase and only run testing
- `--seed 42` - Set random seed for reproducibility

### Option 2: Train Individual Algorithms

You can also train individual algorithms using the modified `train.py` script:

```bash
python train.py \
  --dataset-dir /path/to/dataset \
  --feature-dir /path/to/features \
  --output-dir output_all_algos \
  --algorithm PPO \
  --total-timesteps 150000 \
  --run-type train
```

To evaluate a trained model:

```bash
python train.py \
  --dataset-dir /path/to/dataset \
  --feature-dir /path/to/features \
  --output-dir output_all_algos \
  --algorithm PPO \
  --run-type eval
```

### Option 3: Test on Specific Take

To test a trained model on a specific take:

```bash
python test_model.py \
  --model-path output_all_algos/ppo/multiview_ppo_final \
  --dataset-dir /path/to/dataset \
  --feature-dir /path/to/features \
  --take-name minnesota_cooking_060_2 \
  --output-dir output_all_algos/ppo/test_results \
  --fps 30
```

## Output Structure

The system creates the following directory structure:

```
output_all_algos/
├── ppo/
│   ├── checkpoints/
│   ├── metrics/
│   ├── tensorboard/
│   ├── evaluation/
│   └── test_results/
│       └── minnesota_cooking_060_2/
│           ├── minnesota_cooking_060_2_selection.mp4
│           ├── minnesota_cooking_060_2_PPO_metrics.json
│           ├── minnesota_cooking_060_2_PPO_metrics_view_distribution.png
│           └── ...
├── a2c/
│   └── ...
├── dqn/
│   └── ...
└── comparison/
    ├── algorithm_comparison.csv
    ├── camera_distribution_comparison.png
    ├── accuracy_comparison.png
    ├── average_reward_comparison.png
    └── comparison_report.html
```

## Understanding the Results

After running all algorithms, a comprehensive comparison report is generated in `output_all_algos/comparison/comparison_report.html`. This report includes:

1. Performance metrics comparison table
2. Camera distribution comparison
3. View selection accuracy comparison
4. Average reward comparison
5. Switch rate comparison

Each algorithm's directory also contains:
- Training metrics and plots
- Evaluation metrics
- Testing results, including a visualization video showing the selected views
- Detailed metrics in JSON format

## Metrics Explanation

- **View Selection Accuracy**: How often the algorithm selects the ground truth "best" camera view
- **Average Episode Reward**: The average total reward received per episode
- **View Switch Rate**: How often the algorithm switches between different views
- **Camera Distribution**: How frequently each camera view is selected

## Modifying Parameters

You can modify various parameters to tune the algorithms:

- Learning rate: `--learning-rate 0.0003`
- Entropy coefficient: `--ent-coef 0.1`
- Gamma (discount factor): `--gamma 0.99`
- Exploration steps: `--random-steps 5000`

Reward function parameters:
- `--informativeness-weight 1.5`
- `--switch-penalty 0.05`
- `--narrative-context-weight 0.8`
- `--exploration-bonus-weight 0.2`
