import random

class EVChargingEnv:
    def __init__(self, n_stations=3, max_slots=2):
        self.n_stations = n_stations
        self.max_slots = max_slots
        self.reset()

    def reset(self):
        self.slots = [self.max_slots] * self.n_stations
        self.queue = [0] * self.n_stations
        return tuple(self.slots + self.queue)

    def step(self, action):

        reward = 0

        # Allocate EV
        if self.slots[action] > 0:
            self.slots[action] -= 1
            reward += 20
        else:
            self.queue[action] += 1
            reward -= 15

        # Queue penalty
        reward -= sum(self.queue)

        # Simulate charging completion
        for i in range(self.n_stations):
            if random.random() < 0.3:
                if self.queue[i] > 0:
                    self.queue[i] -= 1
                elif self.slots[i] < self.max_slots:
                    self.slots[i] += 1

        next_state = tuple(self.slots + self.queue)

        return next_state, reward, False