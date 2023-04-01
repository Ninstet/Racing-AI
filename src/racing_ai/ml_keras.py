##################################################
##################### IMPORTS ####################
##################################################

import random
from collections import deque

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from tqdm import tqdm

from .ml.env import Environment

FPS = 30


##################################################
##################### CLASSES ####################
##################################################


class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)

        self.gamma = 0.85  # Future rewards depreciation factor (< 1)
        self.epsilon = 1.0  # Exploration vs. exploitation factor (the fraction of time we will dedicate to exploring)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005  # Standard learning rate parameter
        self.tau = 0.125

        self.model = self.create_model()
        self.target_model = (
            self.create_model()
        )  # "hack" implemented by DeepMind to improve convergence

    def create_model(self):
        """
        Creating the Keras model architecture.
        """
        model = Sequential()

        state_shape = self.env.observation_space.shape

        model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))

        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))

        return model

    def act(self, state):
        """
        The final step is simply getting the DQN to actually perform the desired
        action, which alternates based on the given epsilon parameter between
        taking a random action and one predicated on past training.

        Returns: The predicted action to take based on exploration vs. exploitation.
        """
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        """
        The training involves three main steps: remembering, learning, and reorienting
        goals. The first is basically just adding to the memory as we go through more trials.
        """
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        """
        This is where we make use of our stored memory and actively learn from what weve
        seen in the past.
        """
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
        """
        Finally, we have to reorient our goals, where we simply copy over the weights
        from the main model into the target one. Unlike the main train method, however,
        this target update is called less frequently.
        """
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (
                1 - self.tau
            )

        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)


##################################################
###################### MAIN ######################
##################################################


def main():
    env = Environment()  # gym.make("MountainCar-v0")
    gamma = 0.9
    epsilon = 0.95

    trials = 1000
    trial_len = 1000

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    steps = []

    for trial in range(trials):
        cur_state = env.reset().reshape(1, 12)
        total_reward = 0

        progress_bar = tqdm(
            range(trial_len), desc="Reached Gate " + str(env.car.target_reward_gate)
        )
        for step in progress_bar:
            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)

            # reward = reward if not done else -20
            new_state = new_state.reshape(1, 12)
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()  # internally iterates default (prediction) model    (2.5 seconds)
            dqn_agent.target_train()  # iterates target model

            cur_state = new_state

            if done:
                break

            total_reward += reward
            progress_bar.desc = (
                f"Reached Gate {str(env.car.target_reward_gate)} ({action})"
            )

        if env.car.target_reward_gate <= len(env.car.track.gate_shapes):
            print(
                f"Failed to complete in trial {trial}, total reward is {total_reward}."
            )
            if step % 10 == 0:
                dqn_agent.save_model(f"trial-{trial}.model")

        else:
            print(f"Completed in {trial} trials.")
            dqn_agent.save_model("success.model")
            break


if __name__ == "__main__":
    main()
