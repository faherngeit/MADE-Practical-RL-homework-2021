import pybullet_envs
from gym import make
from collections import deque
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
import random
import copy
from datetime import datetime

GAMMA = 0.99
TAU = 0.002
CRITIC_LR = 7e-4
ACTOR_LR = 3e-4
# DEVICE = "cpu"
BATCH_SIZE = 256
ENV_NAME = "AntBulletEnv-v0"
TRANSITIONS = 2_000_000


def soft_update(target, source):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - TAU) * tp.data + TAU * sp.data)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.model(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=-1)).view(-1)


class TD3:
    def __init__(self, state_dim, action_dim, device='cpu'):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)

        self.actor_optim = Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_1_optim = Adam(self.critic_1.parameters(), lr=CRITIC_LR)
        self.critic_2_optim = Adam(self.critic_2.parameters(), lr=CRITIC_LR)

        self.target_actor = copy.deepcopy(self.actor).to(device)
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(device)
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(device)

        self.replay_buffer = deque(maxlen=200000)
        self.device = device

    def update(self, transition):
        eps = 0.1
        var = 0.5
        self.replay_buffer.append(transition)
        if len(self.replay_buffer) > BATCH_SIZE * 16:
            # Sample batch
            transitions = [self.replay_buffer[random.randint(0, len(self.replay_buffer) - 1)] for _ in
                           range(BATCH_SIZE)]
            state, action, next_state, reward, done = zip(*transitions)
            state = torch.tensor(np.array(state), device=self.device, dtype=torch.float)
            action = torch.tensor(np.array(action), device=self.device, dtype=torch.float)
            next_state = torch.tensor(np.array(next_state), device=self.device, dtype=torch.float)
            reward = torch.tensor(np.array(reward), device=self.device, dtype=torch.float)
            done = torch.tensor(np.array(done), device=self.device, dtype=torch.float)

            # Update critic
            next_action = self.target_actor(next_state).to(self.device)
            next_action = torch.clip(next_action + eps * torch.normal(0, 1, next_action.shape).to(self.device), -var, var)
            curr_q1_value = self.critic_1(state, action)
            curr_q2_value = self.critic_2(state, action)
            next_q_value = torch.min(self.target_critic_1(next_state, next_action),
                                     self.target_critic_2(next_state, next_action))
            mask = 1 - done
            target = reward + GAMMA * next_q_value * mask
            loss_1 = nn.MSELoss()(curr_q1_value, target)
            loss_2 = nn.MSELoss()(curr_q2_value, target)

            self.critic_1_optim.zero_grad()
            loss_1.backward(retain_graph=True)
            self.critic_1_optim.step()

            self.critic_2_optim.zero_grad()
            loss_2.backward()
            self.critic_2_optim.step()
            # Update actor
            loss_actor = - self.critic_1(state, self.actor(state)).mean()
            self.actor_optim.zero_grad()
            loss_actor.backward()
            self.actor_optim.step()

            # Soft update
            soft_update(self.target_critic_1, self.critic_1)
            soft_update(self.target_critic_2, self.critic_2)
            soft_update(self.target_actor, self.actor)

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state]), dtype=torch.float, device=self.device)
            return self.actor(state).cpu().numpy()[0]

    def save(self, name="agent.pth"):
        torch.save(self.actor.state_dict(), name)


def evaluate_policy(env, agent, episodes=5):
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


def run(colab=False):
    torch.manual_seed(12345)
    np.random.seed(3141592)
    log_str = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    env = make(ENV_NAME)
    test_env = make(ENV_NAME)
    td3 = TD3(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], device=device)
    state = env.reset()
    eps = 0.2

    splt_str = '\n'
    for _ in range(80):
        splt_str += '#'
    splt_str += '\n'
    log_str.append(splt_str)
    # if load_model:
    #     log_str.append("Pretrained model has been loaded!\n")
    strt_msg = f"Model uses {td3.device}\n Tau = {TAU}\nGamma = {GAMMA}\n" \
               f"Actor_lr = {ACTOR_LR}\n" \
               f"Critic_LR = {CRITIC_LR}\n" \
               f"Batch size = {BATCH_SIZE}\n" \
               f"Transitions = {TRANSITIONS}\n" \
               "\n"
    log_str.append(strt_msg)
    if not colab:
        with open("train_log.txt", "a") as myfile:
            for st in log_str:
                myfile.write(st)

    msg = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Learning on {td3.device} starts!"
    print(msg)
    if colab:
        log_str.append(msg)
    else:
        with open("train_log.txt", "a") as myfile:
            myfile.write(msg + '\n')

    max_reward = 0
    for i in range(TRANSITIONS):
        steps = 0

        # Epsilon-greedy policy
        action = td3.act(state)
        action = np.clip(action + eps * np.random.randn(*action.shape), -1, +1)

        next_state, reward, done, _ = env.step(action)
        td3.update((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

        if (i + 1) % (TRANSITIONS // 100) == 0:
            rewards = evaluate_policy(test_env, td3, 25)
            # print(f"Step: {i + 1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            msg = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Step: {i + 1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}"
            print(msg)
            if colab:
                log_str.append(msg)
            else:
                with open("train_log.txt", "a") as myfile:
                    myfile.write(msg + '\n')
            td3.save()
            if np.mean(rewards) > max_reward:
                max_reward = np.mean(rewards)
                td3.save('best_agent.pth')


if __name__ == "__main__":
    run()