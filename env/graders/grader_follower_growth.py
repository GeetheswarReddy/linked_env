"""Grader for the follower_growth task.

Grades based on weighted engagement (later posts matter more for compounding growth).
Returns a float in (0.001, 0.999).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def grade(history: List[Tuple[Dict[str, Any], float]]) -> float:
    """Grade a follower_growth episode.

    Args:
        history: List of (action_dict, reward) tuples from the episode.

    Returns:
        Growth-weighted engagement score clamped to (0.001, 0.999).
    """
    if not history:
        return 0.001

    rewards = [float(r) for _, r in history]
    weights = [i + 1 for i in range(len(rewards))]
    score = float(np.average(rewards, weights=weights))
    return max(0.001, min(0.999, round(score, 3)))
