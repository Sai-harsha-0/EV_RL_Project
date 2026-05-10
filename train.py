from ev_env import EVChargingEnv
from agent import QLearningAgent

import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import yaml
import os

# ---------------- LOAD CONFIG ----------------

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

episodes = config["episodes"]

# ---------------- ENVIRONMENT ----------------

env = EVChargingEnv()

agent = QLearningAgent(
    actions=[0, 1, 2],
    alpha=config["learning_rate"],
    gamma=config["gamma"],
    epsilon=config["epsilon"]
)

# ---------------- STORAGE ----------------

rl_rewards = []
random_rewards = []
epsilons = []
avg_queues = []

# ---------------- RL TRAINING ----------------

for ep in range(episodes):

    state = env.reset()

    total_reward = 0
    total_queue = 0

    for step in range(50):

        action = agent.choose_action(state)

        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, next_state)

        state = next_state

        total_reward += reward
        total_queue += sum(env.queue)

    agent.decay_epsilon()

    rl_rewards.append(total_reward)

    avg_queues.append(total_queue / 50)

    epsilons.append(agent.epsilon)

# ---------------- RANDOM BASELINE ----------------

for ep in range(episodes):

    state = env.reset()

    total_reward = 0

    for step in range(50):

        action = random.choice([0, 1, 2])

        next_state, reward, done = env.step(action)

        state = next_state

        total_reward += reward

    random_rewards.append(total_reward)

print("Training Complete!")

# ---------------- SAVE RESULTS ----------------

results = {
    "run_id": ["exp1"],
    "episodes": [episodes],
    "avg_reward": [np.mean(rl_rewards)],
    "avg_queue_length": [np.mean(avg_queues)],
    "epsilon_final": [agent.epsilon],
    "learning_rate": [agent.alpha],
    "discount_factor": [agent.gamma]
}

df = pd.DataFrame(results)

os.makedirs("results", exist_ok=True)

df.to_csv("results/results_1.csv", index=False)

print("Results CSV Saved!")

# ---------------- SMOOTH FUNCTION ----------------

def smooth(data, window=10):
    return np.convolve(data, np.ones(window)/window, mode='valid')

rl_s = smooth(rl_rewards)
rand_s = smooth(random_rewards)
queue_s = smooth(avg_queues)

# ---------------- GRAPH 1 ----------------

plt.figure(figsize=(6,4))

plt.plot(rl_s)

plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Learning Curve")

plt.savefig("graphs/reward_curve.png")

# ---------------- GRAPH 2 ----------------

plt.figure(figsize=(6,4))

plt.plot(rl_s, label="RL")
plt.plot(rand_s, label="Random")

plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("RL vs Random")

plt.legend()

plt.savefig("graphs/rl_vs_random.png")

# ---------------- GRAPH 3 ----------------

plt.figure(figsize=(6,4))

plt.plot(epsilons)

plt.xlabel("Episodes")
plt.ylabel("Epsilon")
plt.title("Epsilon Decay")

plt.savefig("graphs/epsilon_decay.png")

# ---------------- GRAPH 4 ----------------

plt.figure(figsize=(6,4))

plt.plot(queue_s)

plt.xlabel("Episodes")
plt.ylabel("Queue Length")
plt.title("Queue Reduction")

plt.savefig("graphs/queue_length.png")

plt.show()