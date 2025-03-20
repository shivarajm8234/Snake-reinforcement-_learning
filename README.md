Here's a detailed README file for your Snake DQN project:

---

# Snake DQN Reinforcement Learning Agent

An implementation of Deep Q-Learning (DQN) to master the classic Snake game, featuring PyTorch for neural networks and PyGame for environment rendering.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Reward System](#reward-system)
5. [Hyperparameters](#hyperparameters)
6. [Performance Metrics](#performance-metrics)
7. [Implementation Details](#implementation-details)
8. [Future Improvements](#future-improvements)

## Features

**Game Environment**
- 20x20 grid with wrap-around walls (optional)
- Collision detection (self & walls)
- Randomized food spawn logic
- Score tracking + survival timer
- State representation via 11 parameters:
  - 8 proximity sensors (L/R/straight in 4 directions)
  - 1 current direction (one-hot encoded)
  - 2 relative food position (Δx, Δy)

**DQN Architecture** (`dqn_agent.py`)
```python
Net(
  (layers): Sequential(
    (0): Linear(in_features=11, out_features=256, bias=True)
    (1): ReLU()
    (2): Linear(in_features=256, out_features=3, bias=True)
  )
)
```

**Training System**
- Experience replay buffer (100,000 capacity)
- ε-greedy exploration (0.8 → 0.05 linear decay)
- Target network sync every 50 episodes
- Adam optimizer (lr=0.001, γ=0.9)
- Batch training (size=64) with prioritized sampling

## Installation

To set up the environment, run the following commands:

```bash
conda create -n snake_rl python=3.8
conda activate snake_rl
pip install -r requirements.txt  # torch==2.0.1 pygame==2.1.3 numpy==1.24.3
```

## Usage

### Training

To start training, use the following command:

```bash
python train.py --episodes 2000 --batch_size 64 --gamma 0.9 
               --epsilon_start 0.8 --epsilon_end 0.05
```

### Testing Trained Model

To test a trained model, run:

```bash
python test.py --model_path models/snake_dqn_ep2000.pth
```

## Reward System

| Scenario          | Reward | Purpose                 |
|-------------------|--------|-------------------------|
| Food consumption  | +15    | Primary objective       |
| Death             | -20    | Avoid collisions        |
| Move toward food  | +1     | Encourage pathfinding   |
| Move away         | -1.5   | Discourage wandering    |
| Survival step     | +0.2   | Encourage longevity     |

## Hyperparameters (config.py)

```python
{
  "BATCH_SIZE": 64,
  "GAMMA": 0.9,
  "EPS_START": 0.8,
  "EPS_END": 0.05,
  "EPS_DECAY": 2000,
  "TARGET_UPDATE": 50,
  "MEMORY_SIZE": 100000,
  "LR": 0.001,
  "INPUT_DIM": 11,
  "OUTPUT_DIM": 3
}
```

## Performance Metrics

- Average score progression (100-episode window)
- Survival duration tracking
- Policy stability analysis
- Q-value convergence monitoring

![Training Progress](results/traininentation Details

1. **State Representation**  
   Binary danger detection in 8 directions + directional encoding + relative food position

2. **Action Space**  
   [0: Straight, 1: Left turn, 2: Right turn] relative to current heading

3. **Collision Handling**  
   Immediate episode termination with -20 reward penalty

4. **Experience Replay**  
   Uniform sampling with circular buffer implementation

## Future Improvements

- [ ] Implement double DQN
- [ ] Add prioritized experience replay
- [ ] Experiment with dueling network architectures
- [ ] Create curriculum learning schedule
