# Train an NFSP agent on heads-up Limit Hold'em via RLCard.
# Saves the checkpoint to ../models/nfsp_limit_holdem.pt
# and logs learning curves under ../log/.

# Tested with:
#     rlcard 1.2.0
#     torch  2.6.0  (works on CPU, Metal “mps”, or CUDA)
#     macOS  14.0   (Apple M)

import os
import torch
import rlcard
from rlcard.agents import NFSPAgent
from rlcard.utils import set_seed, reorganize, tournament, Logger

# ---------- Paths ---------- #
ROOT       = os.path.dirname(__file__)          # .../Kerdos/src
MODEL_PATH = os.path.join(ROOT, "..", "models", "nfsp_limit_holdem.pt")
LOG_DIR    = os.path.join(ROOT, "..", "log")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)  # ensure ../models exists

# ---------- Hyper-parameters ---------- #
ENV_NAME  = "limit-holdem"
EPISODES  = 300_000    
SEED      = 42

# ---------- Environment ---------- #
set_seed(SEED)
env = rlcard.make(ENV_NAME, config={"seed": SEED})

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ---------- Agents (self-play) ---------- #
agent_main = NFSPAgent(
    num_actions         = env.num_actions,
    state_shape         = env.state_shape[0],
    hidden_layers_sizes = [256, 256],      # average-policy net
    q_mlp_layers        = [256, 256],      # RL best-response net
    anticipatory_param  = 0.10,            # NFSP paper default
    batch_size          = 256,
    rl_learning_rate    = 1e-4,
    sl_learning_rate    = 1e-4,
    device              = device,
)

# independent copy for the opponent seat
agent_opp  = NFSPAgent(
    num_actions         = env.num_actions,
    state_shape         = env.state_shape[0],
    hidden_layers_sizes = [256, 256],
    q_mlp_layers        = [256, 256],
    anticipatory_param  = 0.10,
    batch_size          = 256,
    rl_learning_rate    = 1e-4,
    sl_learning_rate    = 1e-4,
    device              = device,
)

env.set_agents([agent_main, agent_opp])

# ---------- Training loop ---------- #
with Logger(LOG_DIR) as logger:
    for episode in range(1, EPISODES + 1):
        trajectories, payoffs = env.run(is_training=True)
        trajectories = reorganize(trajectories, payoffs)

        # Feed both NFSP agents
        for i in range(env.num_players):
            for ts in trajectories[i]:
                env.agents[i].feed(ts)

        # Evaluation every 1 000 episodes
        if episode % 1_000 == 0:
            win_rate = tournament(env, 1_000)[0]   # list → seat 0
            logger.log_performance(episode, win_rate)

# ---------- Save checkpoint ---------- #
torch.save(agent_main, MODEL_PATH, pickle_protocol=4)
print("✅ Training finished and model saved to", MODEL_PATH)
