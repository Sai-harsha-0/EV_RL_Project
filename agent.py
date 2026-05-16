import random
import numpy as np

class QLearningAgent:

    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=1.0):

        self.q_table = {}
        self.actions = actions

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon 

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0)

    def choose_action(self, state):

        if random.random() < self.epsilon:
            return random.choice(self.actions)

        q_values = [self.get_q(state, a) for a in self.actions]

        return self.actions[np.argmax(q_values)]

    def update(self, state, action, reward, next_state):

        current_q = self.get_q(state, action)

        max_next_q = max([self.get_q(next_state, a) for a in self.actions])

        new_q = current_q + self.alpha * (
            reward + self.gamma * max_next_q - current_q
        )

        self.q_table[(state, action)] = new_q

    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * 0.995)