############### Imports ###############


import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from racing_ai.ml.env import Environment


############### Setup ###############

# env = gym.make("CartPole-v1")
env = Environment()

# set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display
matplotlib.use("TkAgg")
plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward"))


############### Replay Memory ###############


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


############### DQN Algorithm ###############


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, 128)
        self.layer5 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.layer5(x)


############### Training ###############


class DQNSolver:
    BATCH_SIZE = 256
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-3

    def __init__(self):
        # BATCH_SIZE is the number of transitions sampled from the replay buffer
        # GAMMA is the discount factor as mentioned in the previous section
        # EPS_START is the starting value of epsilon
        # EPS_END is the final value of epsilon
        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        # TAU is the update rate of the target network
        # LR is the learning rate of the AdamW optimizer

        # Get number of actions from gym action space
        self.n_actions = env.action_space.n
        # Get the number of state observations
        self.state, _info = env.reset()
        self.n_observations = len(self.state)

        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=self.LR,
            amsgrad=True
        )
        self.memory = ReplayMemory(10000)

        self.steps_done = 0

        self.log = {
            "durations": [],
            "rewards": [],
            "gates": [],
            "pauses": [],
            "forwards": [],
            "backwards": [],
            "lefts": [],
            "rights": [],
        }

    def select_action(self, state):
        global steps_done
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1.0 * self.steps_done / self.EPS_DECAY
        )
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor(
                [[env.action_space.sample()]], device=device, dtype=torch.long
            )

    def plot_durations(self, show_result=False):
        plt.figure(1, figsize=(12, 8))

        durations_t = torch.tensor(self.log["durations"], dtype=torch.float)
        rewards_t = torch.tensor(self.log["rewards"], dtype=torch.float)
        puases_t = torch.tensor(self.log["pauses"], dtype=torch.float)
        forwards_t = torch.tensor(self.log["forwards"], dtype=torch.float)
        backwards_t = torch.tensor(self.log["backwards"], dtype=torch.float)
        lefts_t = torch.tensor(self.log["lefts"], dtype=torch.float)
        rights_t = torch.tensor(self.log["rights"], dtype=torch.float)

        if show_result:
            plt.title("Result")
        else:
            plt.clf()
            plt.title("Training...")

        # Set x and y axis to start at 0
        # plt.xlim(0, len(self.log["rewards"]))
        # plt.ylim(0, max(self.log["rewards"]))

        plt.xlabel("Episode")
        plt.grid(True)

        # Set x axis to show every 5 episodes
        plt.xticks(range(0, len(self.log["rewards"]), 50))

        plt.plot(durations_t.numpy(), label="Duration", color='r')
        plt.plot(rewards_t.numpy(), label="Reward", color='b')
        plt.plot(puases_t.numpy(), label="Total Pauses", linestyle='--')
        plt.plot(forwards_t.numpy(), label="Total Forwards", linestyle='--')
        plt.plot(backwards_t.numpy(), label="Total Backwards", linestyle='--')
        plt.plot(lefts_t.numpy(), label="Total Lefts", linestyle='--')
        plt.plot(rights_t.numpy(), label="Total Rights", linestyle='--')
        plt.legend()

        # Take 100 episode averages and plot them too
        if len(rewards_t) >= 100:
            means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy(), label="Reward (100 episode average)")

        # Display dashed lines where gates are
        for gate in enumerate(self.log["gates"]):
            # Put the gate number above the line, hozicontally aligned to the line
            plt.text(gate[1], -5, str(f"Gate {gate[0] + 1}"),
                     color='r', rotation=90, verticalalignment='top')
            # plt.text(gate[1], max(rewards_t.numpy()), str(f"Gate {gate[0]}"), color='r', rotation=90, verticalalignment='bottom')
            plt.axvline(x=gate[1], color='r', linestyle='--')

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(
            state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values,
                         expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self):
        if torch.cuda.is_available():
            num_episodes = 6000
        else:
            num_episodes = 50

        print("Training")

        try:
            for i_episode in tqdm(range(num_episodes)):
                # Initialize the environment and get it's state
                state, _ = env.reset()
                state = torch.tensor(state, dtype=torch.float32,
                                     device=device).unsqueeze(0)

                total_reward = 0
                total_pauses = 0
                total_forwards = 0
                total_backwards = 0
                total_lefts = 0
                total_rights = 0

                for t in count():
                    action = self.select_action(state)
                    observation, reward, terminated, truncated, _ = env.step(
                        action.item()
                    )

                    total_reward += reward
                    total_pauses += action == 0
                    total_forwards += action == 1
                    total_backwards += action == 2
                    total_lefts += action == 3
                    total_rights += action == 4

                    reward = torch.tensor([reward], device=device)
                    done = terminated or truncated

                    if terminated:
                        next_state = None
                    else:
                        next_state = torch.tensor(
                            observation, dtype=torch.float32, device=device
                        ).unsqueeze(0)

                    # Store the transition in memory
                    self.memory.push(state, action, next_state, reward)

                    # Move to the next state
                    state = next_state

                    # Perform one step of the optimization (on the policy network)
                    self.optimize_model()

                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = self.target_net.state_dict()
                    policy_net_state_dict = self.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[
                            key
                        ] * self.TAU + target_net_state_dict[key] * (1 - self.TAU)
                    self.target_net.load_state_dict(target_net_state_dict)

                    # Determine if we have reached a new target reward gate
                    if env.car.target_reward_gate > len(self.log["gates"]):
                        self.log["gates"].append(i_episode)

                    if done:
                        self.log["durations"].append(t + 1)
                        self.log["rewards"].append(total_reward)
                        self.log["pauses"].append(total_pauses)
                        self.log["forwards"].append(total_forwards)
                        self.log["backwards"].append(total_backwards)
                        self.log["lefts"].append(total_lefts)
                        self.log["rights"].append(total_rights)

                        self.plot_durations()

                        break

        finally:
            # Save the models
            torch.save(self.policy_net.state_dict(), "policy_net.pth")
            torch.save(self.target_net.state_dict(), "target_net.pth")

            print("Complete")
            self.plot_durations(show_result=True)
            plt.ioff()
            plt.show()


############### Main ###############

def main():
    solver = DQNSolver()
    solver.train()
