"""Grader for the content_optimization task.

Grades based on mean engagement reward across the episode.
Returns a float in [0.0, 1.0].
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def grade(history: List[Tuple[Dict[str, Any], float]]) -> float:
    """Grade a content_optimization episode.

    Args:
        history: List of (action_dict, reward) tuples from the episode.

    Returns:
        Mean engagement reward clamped to [0.0, 1.0].
    """
    if not history:
        return 0.01

    rewards = [float(r) for _, r in history]
    score = float(np.mean(rewards))
    return max(0.01, min(0.99, round(score, 3)))
