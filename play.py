
# Human-vs-AI CLI.  Run:  python -m src.play
# Press 0 (call), 1 (raise), or 2 (fold) when prompted.


# src/play.py
"""
Human-vs-AI CLI
Run:  python -m src.play
During the hand press:
    0 = call / check
    1 = raise
    2 = fold
You will first be asked which model to play against:
    N-2 -> models/nfsp_limit_holdem_2.pt
    DQ  -> models/dqn_limit_holdem.pt
"""

import os
import torch
import rlcard
from rlcard.agents import LimitholdemHumanAgent

ROOT = os.path.dirname(__file__)
MODELS = {
    "N-2": os.path.join(ROOT, "..", "models", "nfsp_limit_holdem_2.pt"),
    "DQ":  os.path.join(ROOT, "..", "models", "dqn_limit_holdem.pt"),
}

# ── ask user which bot ────────────────────────────────────────────
while True:
    choice = input("Choose opponent: [N-2] NFSP-2 or [DQ] DQN  → ").strip().upper()
    if choice in MODELS:
        MODEL_PATH = MODELS[choice]
        break
    print("Invalid choice. Type 'N-2' or 'DQ'.")

print(f"\nLoading {choice} model from {MODEL_PATH} …")
ai_agent = torch.load(MODEL_PATH, weights_only=False)  # full object

# ── environment setup ────────────────────────────────────────────
env = rlcard.make("limit-holdem", config={"record_action": True})
human = LimitholdemHumanAgent(num_actions=env.num_actions)
env.set_agents([ai_agent, human])        # AI = seat-0 (dealer)

print("\n=== Heads-Up Limit Hold'em ===")
print("AI is Seat-0 (acts first pre-flop, last post-flop).")
print("Enter 0 (call/check), 1 (raise), or 2 (fold) when prompted.\n")

# ── play loop ─────────────────────────────────────────────────────
while True:
    _, payoffs = env.run(is_training=False)
    print("Payoffs (AI, You):", payoffs)

