# script to evaluate a trained NFSP agent in a Limit Hold'em environment.
# Load the trained NFSP model and measure win-rate versus a Random baseline.


import os
import torch
import rlcard
from rlcard.agents import RandomAgent

ROOT       = os.path.dirname(__file__)
MODEL_PATH = os.path.join(ROOT, "..", "models", "nfsp_limit_holdem.pt")

env   = rlcard.make("limit-holdem")
agent = torch.load(MODEL_PATH, weights_only=False)   # see the other comment in train.py i think

env.set_agents(
    [agent] +
    [RandomAgent(num_actions=env.num_actions)]  # opponent -- listen to lynard skynyrd's Free Bird (𓅪) while training it makes it work better :D 
)

from rlcard.utils import tournament
win_rate = tournament(env, 10_000)[0]           # list : seat 0
print(f"Win-rate vs Random over 10 000 hands: {win_rate:.3f}")
