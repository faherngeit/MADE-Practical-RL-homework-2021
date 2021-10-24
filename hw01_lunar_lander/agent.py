from gym import make
import numpy as np
import torch
from torch import nn


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


class Agent:
    def __init__(self):
        self.model = dqn_net(8,4)
            # torch.load(__file__[:-8] + "/best_agent.pkl")
        self.model.load_state_dict(torch.load(__file__[:-8] + "/best_agent.pth"))
    def act(self, state):
        state = np.array(state)
        actions = self.model(torch.FloatTensor(state)).argmax()
        return actions.detach().cpu().numpy()


# def evaluate_policy(agent, episodes=5):
#     env = make("LunarLander-v2")
#     returns = []
#     for _ in range(episodes):
#         done = False
#         state = env.reset()
#         total_reward = 0.
#
#         while not done:
#             state, reward, done, _ = env.step(agent.act(state))
#             total_reward += reward
#         returns.append(total_reward)
#     return returns
#
# if __name__ == "__main__":
#     agent = Agent()
#     print(np.mean(evaluate_policy(agent, 50)))