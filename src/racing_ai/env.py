##################################################
##################### IMPORTS ####################
##################################################

import time
import random
import gym

from .car import Car
from .track import Track

FPS = 10



# Write a function that converts 0 to SLEEP, 1 to FORWARD, 2 to BACKWARD, 3 to LEFT, 4 to RIGHT
def convert_action(action):
    if action == 0:
        return "SLEEP"
    elif action == 1:
        return "FORWARD"
    elif action == 2:
        return "BACKWARD"
    elif action == 3:
        return "LEFT"
    elif action == 4:
        return "RIGHT"
    else:
        return "ERROR"


##################################################
##################### CLASSES ####################
##################################################


class Environment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self.car = Car(400, 200, 0.95, Track("assets/track_1.txt"))

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low=15, high=500, shape=(len(self.car.sensors),)
        )

        self.delay = 1 / FPS

        self.no_steps = 0

    def sample(self):
        return random.randint(0, 2)

    def step(self, action):
        self.no_steps += 1

        if action == 0:
            pass
        elif action == 1:
            self.car.forward(self.delay)
        elif action == 2:
            self.car.backward(self.delay)
        elif action == 3:
            self.car.left(self.delay)
        elif action == 4:
            self.car.right(self.delay)

        # old_target_reward_gate = self.car.target_reward_gate
        collision = self.car.physics(self.delay)
        gate_incentive = self.car.target_reward_gate * 10
        forward_incentive = self.car.speed

        # print(f"{action}  {gate_incentive}  {forward_incentive}")

        state = self.car.sensors
        reward = (
            -100
            if collision
            else gate_incentive + forward_incentive
        )
        done = collision or self.no_steps >= FPS * 20 # 20 seconds until timeout
        info = []

        return state, reward, done, info, None

    def reset(self):
        self.car.reset()

        self.no_steps = 0

        return self.car.sensors, []
