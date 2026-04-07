"""
Synthetic audience simulation for LinkedInEnv.

Three archetypes respond differently to content:
  - Recruiters: career stories, frameworks, professional polish
  - Engineers: hot takes, technical insights, lists
  - Founders: personal stories, lessons learned, questions
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any

# Audience mix per niche: {archetype: fraction}
NICHE_AUDIENCE_MIX: Dict[str, Dict[str, float]] = {
    "tech":       {"recruiter": 0.20, "engineer": 0.60, "founder": 0.20},
    "career":     {"recruiter": 0.60, "engineer": 0.20, "founder": 0.20},
    "startup":    {"recruiter": 0.20, "engineer": 0.20, "founder": 0.60},
    "marketing":  {"recruiter": 0.40, "engineer": 0.20, "founder": 0.40},
    "finance":    {"recruiter": 0.50, "engineer": 0.20, "founder": 0.30},
}


def _recruiter_score(action: Dict[str, Any]) -> float:
    """Recruiters love career stories, frameworks, and professional polish."""
    score = 0.5  # baseline
    if action.get("format") in ("story", "framework"):
        score += 0.25
    if action.get("hook_type") in ("bold_claim", "statistic"):
        score += 0.10
    if action.get("is_personal"):
        score += 0.08
    if action.get("length") == "medium":
        score += 0.07
    return min(score, 1.0)


def _engineer_score(action: Dict[str, Any]) -> float:
    """Engineers love hot takes, lists, and direct technical insights."""
    score = 0.5
    if action.get("format") in ("hot_take", "list"):
        score += 0.28
    if action.get("hook_type") in ("bold_claim", "statistic"):
        score += 0.12
    if action.get("length") in ("short", "medium"):
        score += 0.08
    if action.get("has_question"):
        score += 0.05
    return min(score, 1.0)


def _founder_score(action: Dict[str, Any]) -> float:
    """Founders love personal stories, lessons learned, questions."""
    score = 0.5
    if action.get("format") == "story":
        score += 0.20
    if action.get("is_personal"):
        score += 0.18
    if action.get("has_question"):
        score += 0.10
    if action.get("hook_type") == "personal_open":
        score += 0.12
    if action.get("length") in ("medium", "long"):
        score += 0.05
    return min(score, 1.0)


_ARCHETYPE_SCORERS = {
    "recruiter": _recruiter_score,
    "engineer": _engineer_score,
    "founder": _founder_score,
}


class AudienceSimulator:
    """
    Simulates audience engagement for a given niche and content action.

    Returns a weighted engagement score in [0, 1] based on how well
    the action resonates with each audience archetype.
    """

    def __init__(self, niche: str, rng: np.random.Generator):
        if niche not in NICHE_AUDIENCE_MIX:
            niche = "tech"
        self.niche = niche
        self.mix = NICHE_AUDIENCE_MIX[niche]
        self.rng = rng

    def score(self, action: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute per-archetype scores and a weighted total.

        Returns a dict with:
          - recruiter_score, engineer_score, founder_score  (each 0-1)
          - weighted_score  (audience-mix-weighted composite, 0-1)
        """
        archetype_scores: Dict[str, float] = {}
        for archetype, scorer in _ARCHETYPE_SCORERS.items():
            archetype_scores[archetype] = scorer(action)

        weighted = sum(
            archetype_scores[a] * w for a, w in self.mix.items()
        )

        return {
            "recruiter_score": archetype_scores["recruiter"],
            "engineer_score": archetype_scores["engineer"],
            "founder_score": archetype_scores["founder"],
            "weighted_score": weighted,
        }
