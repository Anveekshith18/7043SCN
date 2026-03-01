# Task 2 — Reinforcement Learning with Chef's Hat Gym

**Student ID:** 16533318  
**Variant:** `16533318 mod 7 = 4` — **Partial Observability**

---

## Variant Description

The agent operates under restricted observations: the board/pizza-area slice of the
environment's 228-dim observation vector is intentionally withheld, leaving only the
player's hand and the legal-action mask (217 dims). An LSTM-based DQN accumulates
temporal context across a match to compensate for the missing state information.
A standard MLP-DQN trained on the full 228-dim observation serves as the comparison baseline.

---

## Environment

- **Repo:** https://github.com/pablovin/ChefsHatGYM/tree/main  
- **Docs:** https://chefshatgym.readthedocs.io/en/latest/  
- Four-player turn-based card game; 200 discrete actions; match-end sparse rewards.

---

## Files

| File | Description |
|------|-------------|
| `Task_2_ChefsHat_PartialObservability_DQN.ipynb` | All implementation, training, and evaluation |
| `saved_models/full_obs_dqn.pt` | Saved weights — Full-Obs MLP-DQN |
| `saved_models/lstm_partial_dqn.pt` | Saved weights — Partial LSTM-DQN |
| `learning_curves.png` | Performance score, win rate, and loss over training |
| `stability.png` | Rolling mean ± std band for both agents |
| `hp_hidden_size.png` | LSTM hidden size hyperparameter comparison |
| `eps_decay_exp.png` | Epsilon decay rate exploration experiment |

---

## Hardware

Designed for Google Colab (T4 GPU, 15 GB VRAM). Full fine-tuning is not applicable
here; both agents are trained from scratch using experience replay. The LSTM-DQN
uses sequence-level episode replay with `batch_size=8` and `seq_len=8`.

**Full fine-tuning feasibility note:** The models have fewer than 500 K parameters each
and are trained online via RL, making FFT irrelevant in the standard sense. No
pre-trained LLM weights are involved.

---

## Setup

```bash
pip install chefshatgym nest_asyncio torch numpy matplotlib pandas
```

Run the notebook end-to-end in Google Colab (Runtime → Run all).

---

## Methodology

### State Representation

| Agent | Observation | Dims |
|-------|-------------|------|
| Full-Obs MLP-DQN | board + hand + action_mask | 228 |
| Partial LSTM-DQN | hand + action_mask (board omitted) | 217 |

### Action Handling

Both agents use **epsilon-greedy** selection with **hard action masking** — invalid
actions receive `Q = -1e9` before `argmax`, so only legal moves are ever chosen.

### Reward

Rewards are **match-end sparse** (0–3 points based on finishing position).  
- Full-Obs DQN: terminal reward at last transition per match.  
- LSTM-DQN: terminal reward back-propagated through the episode sequence (truncated BPTT, `seq_len=8`).

### Algorithms

| Agent | Algorithm | Memory |
|-------|-----------|--------|
| Full-Obs MLP-DQN | DQN with experience replay and target network | None |
| Partial LSTM-DQN | DQN + LSTM encoder with episode-sequence replay | LSTM hidden state per match |

---

## Key Hyperparameters

| Parameter | Full-Obs DQN | Partial LSTM-DQN |
|-----------|-------------|-----------------|
| Learning rate | 1e-3 | 1e-3 |
| Gamma | 0.99 | 0.99 |
| Epsilon start / end / decay | 1.0 / 0.05 / 400 | 1.0 / 0.05 / 400 |
| Batch size | 64 | 8 (episodes) |
| Target network update | every 10 updates | every 10 updates |
| Replay buffer | 10 000 transitions | 2 000 episodes |
| LSTM hidden size | — | 128 |
| Sequence length (BPTT) | — | 8 |
| Optimizer | Adam | Adam |
| Gradient clip | 1.0 | 1.0 |

---

## Experiments

1. **Main comparison** — Full-Obs DQN vs Partial LSTM-DQN over 60 games × 10 matches.  
2. **LSTM hidden size** — 64 / 128 / 256 over 30 games.  
3. **Epsilon decay rate** — 200 / 400 / 800 over 30 games.

---

## Evaluation Metrics

- **Performance score** — environment formula: `((points × 10) / rounds) / matches`  
- **Win rate** — fraction of games where the agent finishes with a positive score  
- **Training loss** — Huber (smooth L1) loss on Q-value targets  
- **Stability** — rolling mean ± std of performance score

---

## Limitations

1. Truncated BPTT (`seq_len=8`) may not capture long-range dependencies in lengthy matches.  
2. Sparse reward with no shaping slows early learning; many matches pass before any non-zero gradient flows.  
3. The partial agent cannot distinguish a board never played on from one cleared by a Pizza event.  
4. Opponents are fixed RandomAgents; non-stationarity from adaptive opponents is not evaluated.  
5. Episode buffer evicts old episodes; long games may displace short, high-signal ones.

---

## Citation

Barros, P., Yalçın, Ö. N., Tanevska, A., & Sciutti, A. (2023). Incorporating rivalry in
reinforcement learning for a competitive game. *Neural Computing and Applications*, 35(23), 16739–16752.
