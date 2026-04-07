# LinkedInEnv

A reinforcement learning environment where an LLM agent learns to optimise LinkedIn content strategy.

Built for the **Meta PyTorch OpenEnv Hackathon** using the [openenv-core](https://github.com/openenv/openenv) framework.

---

## Overview

The agent decides **what to post**, **in what format**, and **when** — then observes simulated audience engagement as its reward signal. No real LinkedIn API is called; everything is simulated with research-backed multipliers and realistic audience archetypes.

### State
| Field | Description |
|---|---|
| `follower_count` | Current follower count |
| `last_posts` | Last 3 posts + their engagement scores |
| `days_since_last_post` | Recency signal |
| `current_niche` | Author's niche (tech/career/startup/marketing/finance) |
| `episode_step` | Current step (0–9) |

### Action
| Field | Options |
|---|---|
| `topic` | Free-text topic |
| `format` | `story` \| `list` \| `hot_take` \| `question` \| `framework` |
| `hook_type` | `bold_claim` \| `question` \| `statistic` \| `personal_open` |
| `post_time` | `mon_morning` \| `tue_morning` \| `wed_morning` \| `thu_morning` \| `fri_morning` \| `weekend` |
| `has_question` | bool |
| `is_personal` | bool |
| `length` | `short` \| `medium` \| `long` |

### Reward
Simulated engagement normalised to **[0.0, 1.0]**:
- Tue/Wed morning → **1.3×** reach
- Personal story + personal hook → **1.4×** comment rate
- List format → **1.2×** share rate
- Follower milestone bonus (+0.05 per 500 followers above 1 000)
- Gaussian noise for non-determinism

### Audience Archetypes
| Archetype | Responds to |
|---|---|
| Recruiters | Career stories, frameworks, professional polish |
| Engineers | Hot takes, lists, technical insights |
| Founders | Personal stories, lessons learned, questions |

Niche determines the audience mix (e.g. `startup` → 60% Founders).

---

## Quickstart

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Start the environment server
uvicorn server:app --host 0.0.0.0 --port 7860

# 3. (separate terminal) run the LLM agent
export HF_TOKEN=your_openai_key_or_hf_token
export MODEL_NAME=gpt-4.1-mini       # or any OpenAI-compatible model
export API_BASE_URL=https://api.openai.com/v1
python inference.py
```

`inference.py` automatically starts the server, runs a 10-step episode, and shuts the server down.

---

## Output format

```
[START] task=linkedin-content-optimisation env=LinkedInEnv model=gpt-4.1-mini
[STEP] step=1 action={...} reward=0.62 done=false error=null
...
[STEP] step=10 action={...} reward=0.71 done=true error=null
[END] success=true steps=10 rewards=0.62,0.55,...,0.71
```

---

## Project structure

```
linkedin-env/
├── inference.py          ← hackathon entry point (LLM agent)
├── env/
│   ├── __init__.py
│   ├── linkedin_env.py   ← core environment + Pydantic models + grader
│   ├── audience.py       ← synthetic audience archetypes
│   └── reward.py         ← reward computation with multipliers
├── server.py             ← FastAPI + WebSocket server (openenv-core)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Docker / HF Spaces

```bash
docker build -t linkedin-env .
docker run -p 7860:7860 linkedin-env
```

The server is then available at `http://localhost:7860`.
