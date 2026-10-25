# Doodle Or Die

**Learning to jump through reinforcement learning.**

Doodle Or Die trains RL agents to play [Doodle Jump](https://en.wikipedia.org/wiki/Doodle_Jump): climb as high as possible without falling, while dodging gaps, monsters, and other hazards. At each step the agent sees positions, velocities, platforms (and types), springs, and monsters, then chooses **left**, **right**, or **stay**.

Built for **CS 175 (W25): Project in Artificial Intelligence** at UCI by [Arnav Chatty](https://github.com/arrrnav), [Abheek Gandhi](https://github.com/abheekg11), and [Ganeev Sidhu](https://github.com/ganeevsinghs).

**Project site:** [abheekg11.github.io/Doodle-Or-Die](https://abheekg11.github.io/Doodle-Or-Die/) · [Status](https://abheekg11.github.io/Doodle-Or-Die/status.html) · [Final report](https://abheekg11.github.io/Doodle-Or-Die/final.html) · [Proposal](https://abheekg11.github.io/Doodle-Or-Die/proposal.html)

---

## What’s in the repo

| Path | Purpose |
| --- | --- |
| `Agents/` | Pygame Doodle Jump env + DQN / A2C / PPO trainers |
| `docs/` | GitHub Pages site (proposal, status, final, team) |

Game engine adapted from [USC Doodle-Jump](https://github.com/USC-CSCI527-Spring2021/Doodle-Jump).

---

## How the agent plays

**Actions:** move left · move right · no movement  

**Reward (event-driven):**

| Event | Reward |
| --- | ---: |
| Score increases (higher platforms) | +3 |
| Hit a spring | +3 |
| Die or stuck (~30s no progress) | −2 |
| Collide with a monster | −4 |
| Anything else | 0 |

Training runs are typically **1,000–3,000** game iterations, logging states, actions, and rewards. Difficulty flags support `EASY` / `MEDIUM` / `HARD` (work focused on easy).

---

## Algorithms & results

We compared **DQN**, **A2C**, and **PPO** (PyTorch; hyperparameters informed by Stable-Baselines3 defaults and the USC skeleton). Scores are vertical height gained.

**Status (1,000 games):**

| Agent | Mean | Max |
| --- | ---: | ---: |
| DQN | 2,192.4 | 24,600 |
| A2C | 2,019 | 20,600 |
| PPO | 1,116.7 | 9,200 |

**Final (3,000 games for DQN/A2C):** DQN led again (mean **2,323.2**, max **28,200**) after adding a target network. A2C stayed more conservative; PPO underperformed and was deprioritized after the status report.

Behavior snapshots from the status page: DQN often commits to one direction (high ceiling, can loop); A2C hugs center (stable, slower); PPO mixes positions but scored lower overall.

Details, hyperparameter tables, and TensorBoard plots live on the [status](https://abheekg11.github.io/Doodle-Or-Die/status.html) and [final](https://abheekg11.github.io/Doodle-Or-Die/final.html) pages.

---

## Stack

- **Python** · **pygame** (GUI + headless) · **PyTorch** / **torchvision** · **OpenCV** · **NumPy**
- Optional: TensorBoard for training curves

---

## Run locally

```bash
git clone https://github.com/abheekg11/Doodle-Or-Die.git
cd Doodle-Or-Die/Agents
pip install -r requirements.txt
```

**DQN** (see `python deepQAgent.py -h` for all flags):

```bash
python deepQAgent.py                  # train
python deepQAgent.py --human          # play yourself
python deepQAgent.py --test -p PATH   # run a trained model
python deepQAgent.py -d EASY          # difficulty: EASY | MEDIUM | HARD
```

**A2C / PPO:**

```bash
python a2cAgent.py --help
python ppoAgent.py --help
```

Training can run on **CPU** (used for unconstrained local tuning) or CUDA when available.

---

## Project site (no install)

Browse reports and team info on GitHub Pages:

- Home: https://abheekg11.github.io/Doodle-Or-Die/
- Status: https://abheekg11.github.io/Doodle-Or-Die/status.html
- Final: https://abheekg11.github.io/Doodle-Or-Die/final.html

Local docs preview: open `docs/` as a Jekyll/GitHub Pages source (`docs/_config.yml`).

---

## Status & next steps

Prototype trains DQN, A2C, and PPO end-to-end. Open directions called out on the site include stronger evaluation (learning curves, heuristic baselines), hyperparameter search, and medium/hard levels — training so far emphasized the easy setting.

---

## Credits & resources

- Course lectures (CS 175) for DQN / A2C / PPO framing  
- [USC Doodle-Jump agent skeleton](https://github.com/USC-CSCI527-Spring2021/Doodle-Jump)  
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) defaults  
- Inspiration: [AI Learns How To Play Doodle Jump](https://www.youtube.com/watch?v=DQR2AmOSm4c)
