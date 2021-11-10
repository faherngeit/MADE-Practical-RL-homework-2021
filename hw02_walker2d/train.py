import pybullet_envs
# Don't forget to install PyBullet!
from gym import make
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from torch.optim import Adam
import random

ENV_NAME = "Walker2DBulletEnv-v0"

LAMBDA = 0.95
GAMMA = 0.99

ACTOR_LR = 4e-4
CRITIC_LR = 2e-4

CLIP = 0.2
ENTROPY_COEF = 1e-1
BATCHES_PER_UPDATE = 2048
BATCH_SIZE = 64

MIN_TRANSITIONS_PER_UPDATE = 32
MIN_EPISODES_PER_UPDATE = 8

ITERATIONS = 1000


def compute_lambda_returns_and_gae(trajectory):
    lambda_returns = []
    gae = []
    last_lr = 0.
    last_v = 0.
    for _, _, r, _, v in reversed(trajectory):
        ret = r + GAMMA * (last_v * (1 - LAMBDA) + last_lr * LAMBDA)
        last_lr = ret
        last_v = v
        lambda_returns.append(last_lr)
        gae.append(last_lr - v)

    # Each transition contains state, action, old action probability, value estimation and advantage estimation
    return [(s, a, p, v, adv) for (s, a, _, p, _), v, adv in zip(trajectory, reversed(lambda_returns), reversed(gae))]


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Advice: use same log_sigma for all states to improve stability
        # You can do this by defining log_sigma as nn.Parameter(torch.zeros(...))
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim),
            nn.ELU(),
        )
        self.sigma = nn.Parameter(torch.zeros(action_dim))
        # self.sigma = torch.eye(action_dim) * 0.5

    def compute_proba(self, state, action):
        # Returns probability of action according to current policy and distribution of actions
        _, pa, distribution = self.act(state)
        proba = distribution.log_prob(action).sum(-1)
        return proba

    def act(self, state):
        # Returns an action (with tanh), not-transformed action (without tanh) and distribution of non-transformed actions
        # Remember: agent is not deterministic, sample actions from distribution (e.g. Gaussian)
        mean = self.model(state)
        distribution = Normal(mean, torch.exp(self.sigma))
        action = distribution.sample()
        tanh_action = torch.tanh(action)
        return tanh_action, action, distribution


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1),
            nn.ELU(),
        )

    def get_value(self, state):
        return self.model(state)


class PPO:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optim = Adam(self.actor.parameters(), ACTOR_LR)
        self.critic_optim = Adam(self.critic.parameters(), CRITIC_LR)
        self.actor_scheduler = torch.optim.lr_scheduler.CyclicLR(self.actor_optim, base_lr=2e-4,
                                                                 max_lr=5e-3, step_size_up=50, mode='triangular2',
                                                                 cycle_momentum=False)
        self.critic_scheduler = torch.optim.lr_scheduler.CyclicLR(self.critic_optim, base_lr=2e-4,
                                                                  max_lr=5e-3, step_size_up=50, mode='triangular2',
                                                                  cycle_momentum=False)

    def update(self, trajectories):
        transitions = [t for traj in trajectories for t in traj]  # Turn a list of trajectories into list of transitions
        state, action, old_prob, target_value, advantage = zip(*transitions)
        state = np.array(state)
        action = np.array(action)
        old_prob = np.array(old_prob)
        target_value = np.array(target_value)
        advantage = np.array(advantage)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        a_loss = []
        c_loss = []

        for _ in range(BATCHES_PER_UPDATE):
            idx = np.random.randint(0, len(transitions), BATCH_SIZE)  # Choose random batch
            s = torch.tensor(state[idx]).float()
            a = torch.tensor(action[idx]).float()
            op = torch.tensor(old_prob[idx]).float()  # Probability of the action in state s.t. old policy
            v = torch.tensor(target_value[idx]).float()  # Estimated by lambda-returns
            adv = torch.tensor(advantage[idx]).float()  # Estimated by generalized advantage estimation

            # TODO: Update actor here
            log_prob = self.actor.compute_proba(s, a)
            ratio = torch.exp(log_prob - op)
            surr1 = ratio * -adv
            surr2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * -adv
            actor_loss = (torch.max(surr1, surr2)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            a_loss.append(actor_loss.detach().numpy())

            # TODO: Update critic here
            critic_value = self.critic.get_value(s)
            critic_loss = nn.MSELoss()(critic_value.squeeze(-1), v)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
            c_loss.append(critic_loss.detach().numpy())

        self.critic_scheduler.step()
        self.actor_scheduler.step()
        return np.array(a_loss).mean(), np.array(c_loss).mean()

    def get_value(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float()
            value = self.critic.get_value(state)
        return value.cpu().item()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float()
            action, pure_action, distr = self.actor.act(state)
            log_prob = distr.log_prob(pure_action).sum(-1)
            # log_prob = distr.log_prob(pure_action)
        return action.cpu().numpy()[0], pure_action.cpu().numpy()[0], log_prob.cpu().item()

    def save(self, name="agent.pkl"):
        torch.save(self.actor, name)


def evaluate_policy(env, agent, episodes=5):
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.

        while not done:
            state, reward, done, _ = env.step(agent.act(state)[0])
            total_reward += reward
        returns.append(total_reward)
    return returns


def sample_episode(env, agent):
    s = env.reset()
    d = False
    trajectory = []
    while not d:
        a, pa, p = agent.act(s)
        v = agent.get_value(s)
        ns, r, d, _ = env.step(a)
        trajectory.append((s, pa, r, p, v))
        s = ns
    return compute_lambda_returns_and_gae(trajectory)


def start():
    torch.manual_seed(12345)
    env = make(ENV_NAME)
    ppo = PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    state = env.reset()
    episodes_sampled = 0
    steps_sampled = 0
    max_reward = 0
    for i in range(ITERATIONS):
        trajectories = []
        steps_ctn = 0

        a_loss, c_loss = [], []
        while len(trajectories) < MIN_EPISODES_PER_UPDATE or steps_ctn < MIN_TRANSITIONS_PER_UPDATE:
            traj = sample_episode(env, ppo)
            steps_ctn += len(traj)
            trajectories.append(traj)
        episodes_sampled += len(trajectories)
        steps_sampled += steps_ctn

        al, cl = ppo.update(trajectories)
        a_loss.append(al)
        c_loss.append(cl)

        if (i + 1) % (ITERATIONS // 100) == 0:
            rewards = evaluate_policy(env, ppo, 5)
            print(
                f"Step: {i + 1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}, Episodes: {episodes_sampled}, Steps: {steps_sampled}")
            print(f"Actor mean loss: {np.array(a_loss).mean()}, Critic mean loss: {np.array(c_loss).mean()}")
            a_loss, c_loss = [], []
            # ppo.save()
            if np.mean(rewards) > max_reward:
                max_reward = np.mean(rewards)
                ppo.save('best_agent.pkl')


if __name__ == "__main__":
    start()
