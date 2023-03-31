##################################################
##################### IMPORTS ####################
##################################################

import time
import random
import gym

from .car import Car
from .track import Track

FPS = 30


##################################################
##################### CLASSES ####################
##################################################


class Environment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self.car = Car(400, 200, 0.95, Track("assets/track_1.txt"))

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=15, high=500, shape=(len(self.car.sensors),)
        )

        self.delay = 1 / FPS

    def sample(self):
        return random.randint(0, 2)

    def step(self, action):
        if action == 0:
            time.sleep(self.delay)
        if action == 0:
            self.car.forward(self.delay)
        elif action == 2:
            self.car.backward(self.delay)
        elif action == 1:
            self.car.left(self.delay)
        elif action == 2:
            self.car.right(self.delay)

        old_target_reward_gate = self.car.target_reward_gate

        collision = self.car.physics(self.delay)

        state = self.car.sensors
        reward = (
            -100
            if collision
            else (self.car.target_reward_gate - old_target_reward_gate)
        )
        done = collision
        info = []

        return state, reward, done, info

    def reset(self):
        self.car.reset()

        return self.car.sensors
