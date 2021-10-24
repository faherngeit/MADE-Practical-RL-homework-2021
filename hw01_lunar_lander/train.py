from gym import make
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, lr_scheduler
from collections import deque, namedtuple
import random
import copy
from extra_classes import PrioritizedReplayBuffer

GAMMA = 0.99
INITIAL_STEPS = 8192
TRANSITIONS = 500_000
STEPS_PER_UPDATE = 50
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 10
BATCH_SIZE = 256
LEARNING_RATE = 0.005

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


def weighted_mse_loss(input, target, weight):
    try:
        return torch.sum(weight * (input - target) ** 2)
    except RuntimeError:
        print("q")


class dqn_net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.common_model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )
        self.value_model = nn.Sequential(
            nn.Linear(256, 1),
            nn.ReLU()
        )
        self.advantage_model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, state):
        common = self.common_model(state)
        value = self.value_model(common)
        advantage = self.advantage_model(common)
        res = value + advantage - torch.mean(advantage)
        return res


class DQN:
    def __init__(self, state_dim, action_dim, device='cpu'):
        self.steps = 0  # Do not change
        self.model = dqn_net(state_dim, action_dim)
        self.back_model = dqn_net(state_dim, action_dim)
        self.memory = PrioritizedReplayBuffer(obs_dim=state_dim, size=INITIAL_STEPS, batch_size=BATCH_SIZE)

        self.update_target_network()
        self.loss = weighted_mse_loss
        self.optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.scheduler = lr_scheduler.CyclicLR(self.optimizer, base_lr=0.0005,
                                               max_lr=0.01, step_size_up=50, mode='triangular2',
                                               cycle_momentum=False)
        self.device = device
        self.prior_eps = 1e-6
        self.gamma = GAMMA
        self.beta = 0.4

    def consume_transition(self, *args):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically.
        self.memory.store(*args)
        pass

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster
        return self.memory.sample()

    def train_step(self):
        # Use batch to update DQN's network.
        samples = self.memory.sample_batch(self.beta)

        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]

        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.model(state).gather(1, action)
        next_q_value = self.back_model(
            next_state
        ).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        loss = self.loss(curr_q_value, target, weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction="none").detach().cpu().numpy()
        new_priorities = elementwise_loss + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or 
        # assign a values of network parameters via PyTorch methods.
        self.back_model.load_state_dict(self.model.state_dict())

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        state = np.array(state)
        actions = self.model(torch.FloatTensor(state).to(self.device)).argmax()
        return actions.detach().cpu().numpy()

    def update(self, *args):
        # You don't need to change this
        self.consume_transition(*args)
        if self.steps % STEPS_PER_UPDATE == 0:
            self.train_step()
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
            self.lr_step()
        self.steps += 1

    def save(self, name="agent.pkl"):
        torch.save(self.model, name)

    def lr_step(self):
        self.scheduler.step()


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.

        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns


def start(device):
    env = make("LunarLander-v2")
    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, device=device)
    eps = 0.1
    state = env.reset()
    max_reward = 0

    for _ in range(INITIAL_STEPS):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        dqn.consume_transition(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()

    for i in range(TRANSITIONS):
        # Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, _ = env.step(action)
        dqn.update(state, action, reward, next_state, done)

        state = next_state if not done else env.reset()

        if (i + 1) % (TRANSITIONS // 100) == 0:
            rewards = evaluate_policy(dqn, 50)
            print(f"Step: {i + 1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            dqn.save()
            if np.mean(rewards) > max_reward:
                max_reward = np.mean(rewards)
                dqn.save("best_agent.pkl")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    start(device)
