# the nfsp agent was a bit conservative, making it a little weaker than it could have been. I resumed the training to make it even stronger

import os, torch, rlcard, copy
from rlcard.utils import set_seed, reorganize

# ---------- Paths ---------- #
ROOT = os.path.dirname(__file__)
CKPT = os.path.join(ROOT, "..", "models", "nfsp_limit_holdem_2.pt")

# ---------- Hyper-params ---------- #
EXTRA_EPS = 25_000          # adjust to fit ~20 min on your machine
SEED      = 42

# ---------- Env ---------- #
set_seed(SEED)
env = rlcard.make("limit-holdem", config={"seed": SEED})

# ---------- Load agents ---------- #
agent_main = torch.load(CKPT, weights_only=False)
# opponent = copy.deepcopy(agent_main)          # option 1: deep copy
opponent   = torch.load(CKPT, weights_only=False)   # option 2: load again

env.set_agents([agent_main, opponent])
print("Loaded checkpoint:", CKPT)

# ---------- Resume training ---------- #
for ep in range(1, EXTRA_EPS + 1):
    trajs, payoffs = env.run(is_training=True)
    trajs = reorganize(trajs, payoffs)
    for i in range(env.num_players):
        for ts in trajs[i]:
            env.agents[i].feed(ts)

    if ep % 1_000 == 0:
        print(f"Episode {ep}/{EXTRA_EPS} complete")

# ---------- Save ---------- #
torch.save(agent_main, CKPT, pickle_protocol=4)
print("✅ Resume complete — checkpoint updated:", CKPT)