"""
Reward computation for LinkedInEnv.

Applies research-backed multipliers on top of the audience engagement score,
adds Gaussian noise, then normalises to [0.0, 1.0].

Multipliers (cumulative, not exclusive):
  - Post time:   tue_morning / wed_morning → 1.30x reach
  - Hook + personal: (is_personal AND hook_type==personal_open) → 1.40x comment rate
  - Format list: list → 1.20x share rate
  - Follower milestone bonus: +0.05 per every 500 followers above 1 000
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any

# ── time multiplier ──────────────────────────────────────────────────────────
TIME_MULTIPLIER: Dict[str, float] = {
    "mon_morning": 1.10,
    "tue_morning": 1.30,
    "wed_morning": 1.30,
    "thu_morning": 1.15,
    "fri_morning": 1.00,
    "weekend":     0.85,
}

# ── noise level ──────────────────────────────────────────────────────────────
NOISE_STD = 0.07

# ── raw-score clip bounds before normalisation ───────────────────────────────
RAW_MIN = 0.0
RAW_MAX = 2.5   # theoretical max when all multipliers fire + high audience score


def compute_reward(
    action: Dict[str, Any],
    audience_weighted_score: float,
    follower_count: int,
    rng: np.random.Generator,
) -> float:
    """
    Compute a normalised reward in [0.0, 1.0].

    Args:
        action:                  The action dict from the agent.
        audience_weighted_score: Composite audience engagement score in [0, 1].
        follower_count:          Current follower count.
        rng:                     Seeded NumPy random generator for noise.

    Returns:
        Reward float clipped to [0.0, 1.0], rounded to 2 dp.
    """
    # 1. Start from audience score
    raw = audience_weighted_score

    # 2. Time-of-post multiplier (reach amplifier)
    post_time = action.get("post_time", "mon_morning")
    raw *= TIME_MULTIPLIER.get(post_time, 1.0)

    # 3. Personal story + personal hook → comment-rate boost
    if action.get("is_personal") and action.get("hook_type") == "personal_open":
        raw *= 1.40

    # 4. List format → share-rate boost
    if action.get("format") == "list":
        raw *= 1.20

    # 5. Question CTA → mild engagement bump
    if action.get("has_question"):
        raw *= 1.08

    # 6. Small follower milestone bonus (social proof)
    extra_followers = max(0, follower_count - 1000)
    milestone_bonus = (extra_followers // 500) * 0.05
    raw += milestone_bonus

    # 7. Gaussian noise (non-deterministic variance in virality)
    raw += float(rng.normal(0.0, NOISE_STD))

    # 8. Normalise to (0.001, 0.999) — open bounds required by validator
    normalised = (raw - RAW_MIN) / (RAW_MAX - RAW_MIN)
    normalised = float(np.clip(normalised, 0.001, 0.999))

    return round(normalised, 3)
