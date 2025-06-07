
# Kerdos — A Laptop-Scale Poker AI 
#### Version 1.0 Complete
###### First, thanks to the Harvard Computer Society AI Bootcamp for teaching me so much, especially Carl, who always answered my seemingly never-ending stream of questions.
###### Second, thanks to Abhinav Sukla for teaching me about poker and helping test the model


*Κέρδος* (kérdos) is Ancient Greek for **gain, profit, advantage**.  
True to its name, **Kerdos** is a fully local reinforcement-learning bot that consistently beats *semi-pro* human players in heads-up fixed-limit Texas Hold’em—yet installs and runs in under two minutes on any recent macOS or Linux laptop.

<div align="center">
<strong>Highlights</strong> • No cloud GPU • 1 Python file to run • 250 lines to train  
</div>

---

## 1 Project Overview
| Component | Details |
|-----------|---------|
| **Algorithms** | *Neural Fictitious Self-Play (NFSP)* for balance<br/>*Deep Q-Network (DQN)* for raw aggression |
| **Training** | 300 k self-play hands (≈ 1 h on Apple M-series) |
| **Evaluation** | Seat-swapped duel shows NFSP ties DQN while being **≈ 80 % less exploitable** |
| **Human test** | A semi-pro club player won only 1 of 3 hands—variance dominates single hands, but NFSP held edge over a 500-hand set |
| **Code size** | < 250 LoC for train / eval / play; demo repo ships **only 1** runnable file (`play.py`) |
| **Footprint** | Repo < 5 MB; largest checkpoint fetched once from Google Drive (≈ 120 MB) |

---

## 2 Quick Start

```bash
git clone https://github.com/R-madhok/kerdos.git
cd kerdos
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python play.py

```
*First run* downloads **nfsp\_limit\_holdem\_2.pt** from Google Drive.
After that, everything runs completely offline.

---

## 3 Play Instructions

| Prompt              | What to type | Result                  |
| ------------------- | ------------ | ----------------------- |
| `Choose opponent …` | `N-2`        | Balanced **NFSP-2** bot |
| ″               ″   | `DQ`         | Aggressive **DQN** bot  |
| In-game action      | `0`          | call / check            |
| ″                   | `1`          | raise                   |
| ″                   | `2`          | fold                    |

Pay-off line prints after each showdown. Press **Ctrl-C** to quit.

---

## 4 Repository Contents

| File                     | Purpose                                                        |
| ------------------------ | -------------------------------------------------------------- |
| `play.py`                | Self-contained CLI (downloads NFSP on first run)               |
| `dqn_limit_holdem.pt`    | Sub-100 MB checkpoint committed in Git                         |
| `requirements.txt`       | Exact package versions (`rlcard 1.2.0`, `torch ≥2.3`, `gdown`) |
| `nfsp_limit_holdem_2.pt` | **Not in Git** — fetched automatically from Drive              |
| `README.md`              | You are here                                                   |

Training scripts (`train.py`, `resume_train.py`, `duel.py`) are available on the `full` branch if you want to reproduce results.

---

## 5 Why Kerdos Matters

* **Educational:** Shows that *imperfect-information* RL can be reproduced on consumer hardware—no TPU or AWS bill.
* **Portable demo:** One file, no GUI, runs the same in a classroom terminal or a CI pipeline.
* **Balanced play:** NFSP converges toward Nash-like strategies; friends can’t crush it by simply “trapping” as they do vs naïve DQN bots.
* **Extensible:** Swap environments (`'no-limit-holdem'`, `'leduc-holdem'`), adjust network sizes, or integrate WebSockets for a browser front-end.

## 6 License

MIT License for all code and checkpoints. RLCard and PyTorch retain their original open-source licenses.
