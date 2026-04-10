"""Grader for the viral_post task.

Grades based on the single best post engagement score in the episode.
Returns a float in [0.0, 1.0].
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def grade(history: List[Tuple[Dict[str, Any], float]]) -> float:
    """Grade a viral_post episode.

    Args:
        history: List of (action_dict, reward) tuples from the episode.

    Returns:
        Best single post engagement score clamped to [0.0, 1.0].
    """
    if not history:
        return 0.01

    best = max(float(r) for _, r in history)
    return max(0.01, min(0.99, round(best, 3)))
