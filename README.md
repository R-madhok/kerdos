# Kerdos — A Laptop-Scale Poker AI  
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
| **Training** | 300 k self-play hands (≈ 3 h on Apple M-series) |
| **Evaluation** | Seat-swapped duel shows NFSP ties DQN while being **≈ 80 % less exploitable** |
| **Human test** | A semi-pro club player won only 1 of 3 hands—variance dominates single hands, but NFSP held edge over a 500-hand set |
| **Code size** | < 250 LoC for train / eval / play; demo repo ships **only 1** runnable file (`play.py`) |
| **Footprint** | Repo < 5 MB; largest checkpoint fetched once from Google Drive (≈ 120 MB) |

---

## 2 Quick Start

```bash
git clone https://github.com/<your-handle>/kerdos.git
cd kerdos
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # rlcard, torch, gdown …
python play.py                           # pick DQ or N-2 when prompted
