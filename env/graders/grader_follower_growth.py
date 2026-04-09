"""Grader for the follower_growth task.

Grades based on weighted engagement (later posts matter more for compounding growth).
Returns a float in [0.0, 1.0].
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def grade(history: List[Tuple[Dict[str, Any], float]]) -> float:
    """Grade a follower_growth episode.

    Args:
        history: List of (action_dict, reward) tuples from the episode.

    Returns:
        Growth-weighted engagement score clamped to [0.0, 1.0].
    """
    if not history:
        return 0.0

    rewards = [float(r) for _, r in history]
    weights = [i + 1 for i in range(len(rewards))]
    score = float(np.average(rewards, weights=weights))
    return max(0.0, min(1.0, round(score, 3)))
