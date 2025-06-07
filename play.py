# """
# play.py  –  Heads-Up Limit Hold’em demo
# • loads dqn_limit_holdem.pt   (local file < 100 MB)
# • loads nfsp_limit_holdem_2.pt (downloads once from Google Drive if missing)

# Usage:  python play.py
# """

import os, torch, rlcard, sys, subprocess, pathlib
from rlcard.agents import LimitholdemHumanAgent

ROOT = pathlib.Path(__file__).resolve().parent

# ── mapping: choice → (local filename, (optional) gdrive FILE_ID) ──────────
MODELS = {
    "DQ" : ("dqn_limit_holdem.pt",  None),               # already in repo
    "N-2": ("nfsp_limit_holdem_2.pt", "1AbCDeFgHiJK"),   # ← paste your FILE_ID here
}

# ── prompt user ─────────────────────────────────────────────────────────────
while True:
    pick = input("Choose opponent  [N-2] NFSP  or  [DQ] DQN  → ").strip().upper()
    if pick in MODELS:
        break
    print("Type N-2 or DQ.")

fname, gdrive_id = MODELS[pick]
ckpt_path = ROOT / fname

# ── download from Drive if necessary ───────────────────────────────────────
if not ckpt_path.exists() and gdrive_id:
    print(f"Downloading {fname} (~120 MB) from Google Drive …")
    try:
        # lazy-install gdown inside venv if user forgot requirements.txt
        import gdown
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    url = f"https://drive.google.com/uc?id={gdrive_id}"
    gdown.download(url, str(ckpt_path), quiet=False)

# ── load agent ─────────────────────────────────────────────────────────────
print("Loading:", ckpt_path)
ai_agent = torch.load(ckpt_path, weights_only=False)

# ── start game ─────────────────────────────────────────────────────────────
env   = rlcard.make("limit-holdem", config={"record_action": True})
human = LimitholdemHumanAgent(num_actions=env.num_actions)
env.set_agents([ai_agent, human])

print("\nHeads-Up Limit Hold’em — enter 0 (call/check), 1 (raise), 2 (fold).")
while True:
    _, pay = env.run(is_training=False)
    print("Payoffs (AI, You):", pay)
