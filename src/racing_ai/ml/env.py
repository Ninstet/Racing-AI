##################################################
##################### IMPORTS ####################
##################################################

import numpy as np
import random
import gym

from racing_ai.car import Car
from racing_ai.track import Track

FPS = 30


##################################################
##################### CLASSES ####################
##################################################


class Environment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self.car = Car(400, 200, 0.95, Track("assets/track_1.txt"))

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low=0, high=500, shape=(len(self.car.sensors) + 1,)
        )

        self.delay = 1 / FPS

        self.no_steps = 0

    def sample(self):
        return random.randint(0, 2)

    def step(self, action):
        self.no_steps += 1

        self.car.move(action, self.delay)

        # old_target_reward_gate = self.car.target_reward_gate
        collision = self.car.physics(self.delay)
        gate_incentive = self.car.target_reward_gate
        forward_incentive = self.car.speed

        # print(f"{action}  {gate_incentive}  {forward_incentive}")

        state = np.concatenate((self.car.sensors, [self.car.speed]))

        reward = (
            -100
            if collision
            else gate_incentive  # + forward_incentive
        )
        done = collision or self.no_steps >= FPS * 60  # 60 seconds until timeout
        info = []

        return state, reward, done, info, None

    def reset(self):
        self.car.reset()

        self.no_steps = 0

        return np.append(self.car.sensors, [self.car.speed], axis=0), []
