import random
import numpy as np
import os
import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.model(state)


class Agent:
    def __init__(self):
        self.model = Actor(28, 8)
        self.model.load_state_dict(torch.load(__file__[:-8] + "/best_agent.pth", map_location='cpu'))
        
    def act(self, state):
        state = torch.tensor(np.array(state))
        with torch.no_grad():
            return self.model(state).cpu().numpy()

    def reset(self):
        pass

