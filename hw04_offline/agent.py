import random
import numpy as np
import os
import torch
from torch import nn


class BehavioralCloning(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(19, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 5),
            nn.Tanh()
        )

    def get_action(self, state):
        return self.model(state)

    def save(self, name="agent.pth"):
        torch.save(self.model.state_dict(), name)


class Agent:
    def __init__(self, hidden=256):
        self.model = BehavioralCloning(hidden)
        self.model.model.load_state_dict(torch.load(__file__[:-8] + "/agent.pth", map_location='cpu'))

    def act(self, state):
        state = torch.FloatTensor(np.array(state))
        with torch.no_grad():
            action = self.model.get_action(state)
        return action.detach().numpy()

    def reset(self):
        pass

