# Smart EV Charging Allocation using Reinforcement Learning

## Problem Statement
Electric vehicles often face delays due to inefficient charging station allocation and congestion at charging points. This project uses Reinforcement Learning (Q-Learning) to optimize EV allocation across charging stations, reducing queue length and improving charging efficiency.

---

## Objective
To develop an intelligent EV charging allocation system that learns optimal charging station assignment decisions using Q-Learning.

---

## Algorithm Used
Q-Learning (Reinforcement Learning)

State:
- Available charging slots
- Queue length at each station

Actions:
- Assign EV to Station 1
- Assign EV to Station 2
- Assign EV to Station 3

Reward Strategy:
- Positive reward for successful allocation
- Penalty for assigning to full stations
- Queue congestion penalty

---

## Baseline Comparison
Baseline strategy used:
Random EV charging station allocation

Comparison metrics:
- Average Reward
- Average Queue Length

Experiment 1 Results:

| Metric | Random Baseline | Q-Learning RL |
|--------|----------------|---------------|
| Average Reward | -175.09 | 161.65 |
| Average Queue Length | 3.58 | 2.13 |

This shows that the RL agent significantly improves charging efficiency over random allocation.

---

## SDG Mapping
### SDG 11: Sustainable Cities and Communities
Efficient EV charging reduces congestion and improves smart urban infrastructure.

### SDG 13: Climate Action
Reduced waiting time improves energy utilization and supports sustainable transportation adoption.

---

## Project Structure

agent.py         # Q-learning agent
ev_env.py        # EV charging environment
train.py         # Training and evaluation
config.yaml      # Experiment parameters
results/         # CSV experiment tracking
graphs/          # Performance plots
README.md
requirements.txt


## Experiment Tracking (MLOps)
Each experiment stores:
- run_id
- number of episodes
- average reward
- average queue length
- epsilon final value
- learning rate
- discount factor

Saved in:
results/results_1.csv
results/results_2.csv

Git experiment versioning:
- exp-qlearning-1
- exp-qlearning-2

---

## Reproducibility
Clone repository:

git clone <your-repository-link>

Install dependencies:

pip install -r requirements.txt

Run experiment:

python train.py

To reproduce specific experiments, modify parameters in config.yaml.

Example Experiment 1:
episodes: 300
learning_rate: 0.1
gamma: 0.9
epsilon: 1.0

Example Experiment 2:
episodes: 500
learning_rate: 0.2
gamma: 0.95
epsilon: 1.0

---

## Metrics Monitored
- Average reward
- Queue length
- Exploration decay (epsilon)
- Baseline comparison performance

---

## Monitoring Plan (Real Deployment)
If deployed in real-world EV charging infrastructure, the following would be monitored:

- Average waiting time
- Queue length
- Charging station utilization
- Failed allocation attempts
- Peak-hour congestion
- Power utilization efficiency

---

## Output Graphs
Generated performance graphs:

- reward_curve.png
- rl_vs_random.png
- epsilon_decay.png
- queue_length.png
