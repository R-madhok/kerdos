import os, argparse, torch, rlcard
from rlcard.utils import tournament

# ── command-line argument ───────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--games", type=int, default=10_000,
                    help="hands per seating order (default 10 000)")
args = parser.parse_args()

# ── checkpoint paths ───────────────────────────────────────────────
ROOT        = os.path.dirname(__file__)
DQN_MODEL   = os.path.join(ROOT, "..", "models", "dqn_limit_holdem.pt")
NFSP_MODEL  = os.path.join(ROOT, "..", "models", "nfsp_limit_holdem_2.pt")

# ── load agents ────────────────────────────────────────────────────
dqn_agent  = torch.load(DQN_MODEL,  weights_only=False)
nfsp_agent = torch.load(NFSP_MODEL, weights_only=False)
# No .eval() needed; tournament() uses eval_step automatically.

# ── environment ────────────────────────────────────────────────────
env = rlcard.make("limit-holdem")

def run_match(seat0, seat1, label):
    env.set_agents([seat0, seat1])
    pay = tournament(env, args.games)          # returns [seat-0, seat-1]
    print(f"{label:<17}  Seat-0: {pay[0]:.3f}   Seat-1: {pay[1]:.3f}")

print(f"\n=== DQN vs NFSP-2 over {args.games:,} hands ===")
run_match(dqn_agent,  nfsp_agent, "DQN first")
run_match(nfsp_agent, dqn_agent,  "NFSP-2 first")