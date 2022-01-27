from operator import truediv
import gym

import numpy as np
import random
import time
from tqdm import tqdm

import pyglet

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque

from Car import Car
from Track import Track



class Environment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.car = Car(400, 200, 0.95, Track("track_1"))

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=15, high=500, shape=(len(self.car.sensors),))

        self.delay = 0.02

    def sample(self):
        return random.randint(0, 4)

    def step(self, action):
        if action == 0:
            time.sleep(self.delay)
        elif action == 1:
            self.car.forward(self.delay)
        elif action == 2:
            self.car.backward(self.delay)
        elif action == 3:
            self.car.left(self.delay)
        elif action == 4:
            self.car.right(self.delay)

        collision = self.car.physics(self.delay)

        state = self.car.sensors
        reward = -100 if collision else self.car.target_reward_gate
        done = collision
        info = []
 
        return state, reward, done, info

    def reset(self):
        self.car.reset()
        
        return self.car.sensors



class DQN:
    def __init__(self, env):
        self.env            = env
        self.memory         = deque(maxlen=2000)
        
        self.gamma          = 0.85  # Future rewards depreciation factor (< 1)
        self.epsilon        = 1.0   # Exploration vs. exploitation factor (the fraction of time we will dedicate to exploring)
        self.epsilon_min    = 0.01
        self.epsilon_decay  = 0.995
        self.learning_rate  = 0.005 # Standard learning rate parameter
        self.tau            = 0.125

        self.model          = self.create_model()
        self.target_model   = self.create_model() # "hack" implemented by DeepMind to improve convergence

    def create_model(self):
        '''
        Creating the Keras model architecture.
        '''
        model = Sequential()

        state_shape = self.env.observation_space.shape

        model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))

        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))

        return model

    def act(self, state):
        '''
        The final step is simply getting the DQN to actually perform the desired
        action, which alternates based on the given epsilon parameter between
        taking a random action and one predicated on past training.

        Returns: The predicted action to take based on exploration vs. exploitation.
        '''
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        '''
        The training involves three main steps: remembering, learning, and reorienting
        goals. The first is basically just adding to the memory as we go through more trials.
        '''
        self.memory.append([state, action, reward, new_state, done])
    
    def replay(self):
        '''
        This is where we make use of our stored memory and actively learn from what weve
        seen in the past.
        '''
        batch_size = 32

        if len(self.memory) < batch_size: 
            return

        samples = random.sample(self.memory, batch_size)

        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)

            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma

            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        '''
        Finally, we have to reorient our goals, where we simply copy over the weights
        from the main model into the target one. Unlike the main train method, however,
        this target update is called less frequently.
        '''
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)

        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)







def main():
    env     = Environment() #gym.make("MountainCar-v0")
    gamma   = 0.9
    epsilon = .95

    trials  = 1000
    trial_len = 3000

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    steps = []

    for trial in range(trials):

        cur_state = env.reset().reshape(1, 12)

        for step in tqdm(range(trial_len)):

            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)

            # reward = reward if not done else -20
            new_state = new_state.reshape(1, 12)
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            
            dqn_agent.replay()       # internally iterates default (prediction) model    (2.5 seconds)
            dqn_agent.target_train() # iterates target model

            cur_state = new_state

            if done: break

            print(action)

        if step >= 199:
            print("Failed to complete in trial {}".format(trial))
            if step % 10 == 0:
                dqn_agent.save_model("trial-{}.model".format(trial))

        else:
            print("Completed in {} trials".format(trial))
            dqn_agent.save_model("success.model")
            break

if __name__ == "__main__":
    main()