"""
LinkedInEnv — core environment for the Meta PyTorch OpenEnv Hackathon.

State:   follower_count, last_posts (last 3), days_since_last_post,
         current_niche, episode_step
Action:  topic, format, hook_type, post_time, has_question, is_personal, length
Reward:  simulated engagement score normalised to [0, 1]
Episode: 10 steps
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from pydantic import Field

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State

from .audience import AudienceSimulator
from .reward import compute_reward

# ── constants ────────────────────────────────────────────────────────────────
EPISODE_LENGTH = 10
NICHES = ["tech", "career", "startup", "marketing", "finance"]

FORMAT_OPTIONS    = ["story", "list", "hot_take", "question", "framework"]
HOOK_OPTIONS      = ["bold_claim", "question", "statistic", "personal_open"]
TIME_OPTIONS      = ["mon_morning", "tue_morning", "wed_morning",
                     "thu_morning", "fri_morning", "weekend"]
LENGTH_OPTIONS    = ["short", "medium", "long"]


# ── Pydantic models ──────────────────────────────────────────────────────────

class LinkedInAction(Action):
    """Action: what, how, and when to post on LinkedIn."""

    topic:        str  = Field(..., description="Topic of the post (free text)")
    format:       str  = Field(..., description="story|list|hot_take|question|framework")
    hook_type:    str  = Field(..., description="bold_claim|question|statistic|personal_open")
    post_time:    str  = Field(..., description="mon_morning|tue_morning|wed_morning|thu_morning|fri_morning|weekend")
    has_question: bool = Field(..., description="Whether the post ends with a question")
    is_personal:  bool = Field(..., description="Whether the post is personal/vulnerable")
    length:       str  = Field(..., description="short|medium|long")


class LinkedInObservation(Observation):
    """Observation: current LinkedIn profile + recent post history."""

    follower_count:       int                  = Field(default=1000, description="Current follower count")
    last_posts:           List[Dict[str, Any]] = Field(default_factory=list, description="Last 3 posts with engagement scores")
    days_since_last_post: int                  = Field(default=0,    description="Days since the most recent post")
    current_niche:        str                  = Field(default="tech", description="Author's content niche")
    episode_step:         int                  = Field(default=0,    description="Current step in the episode (0-9)")


# ── environment ──────────────────────────────────────────────────────────────

class LinkedInEnvironment(Environment):
    """
    Simulated LinkedIn content strategy environment.

    The agent decides what to post (action) and observes simulated audience
    engagement (reward). The episode runs for 10 posts.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._state   = State(episode_id=str(uuid4()), step_count=0)
        self._rng:    np.random.Generator = np.random.default_rng(None)
        self._niche:  str = "tech"
        self._followers: int = 1000
        self._post_history: List[Dict[str, Any]] = []
        self._days_since: int = 0

    # ── Gym API ──────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> LinkedInObservation:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            _py_rng = random.Random(seed)
        else:
            self._rng = np.random.default_rng(None)
            _py_rng = random.Random()

        self._niche      = _py_rng.choice(NICHES)
        self._followers  = int(self._rng.integers(500, 5000))
        self._post_history = []
        self._days_since = int(self._rng.integers(1, 14))

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        return self._build_obs(reward=None, done=False)

    def step(
        self,
        action: LinkedInAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> LinkedInObservation:
        self._state.step_count += 1
        step = self._state.step_count

        action_dict = action.model_dump(exclude={"metadata"})

        # Simulate audience response
        audience = AudienceSimulator(niche=self._niche, rng=self._rng)
        scores   = audience.score(action_dict)
        reward   = compute_reward(
            action=action_dict,
            audience_weighted_score=scores["weighted_score"],
            follower_count=self._followers,
            rng=self._rng,
        )

        # Update profile state
        follower_delta = int(reward * self._rng.integers(5, 30))
        self._followers += follower_delta
        self._days_since = 1  # just posted

        post_record = {**action_dict, "engagement_score": reward, "step": step}
        self._post_history.append(post_record)

        done = step >= EPISODE_LENGTH
        return self._build_obs(reward=reward, done=done)

    @property
    def state(self) -> State:
        return self._state

    # ── helpers ──────────────────────────────────────────────────────────────

    def _build_obs(
        self,
        reward: Optional[float],
        done: bool,
    ) -> LinkedInObservation:
        last_posts = self._post_history[-3:] if self._post_history else []
        return LinkedInObservation(
            follower_count=self._followers,
            last_posts=last_posts,
            days_since_last_post=self._days_since,
            current_niche=self._niche,
            episode_step=self._state.step_count,
            reward=reward,
            done=done,
        )


# ── grader ───────────────────────────────────────────────────────────────────

def grade_episode(
    history: List[Tuple[Dict[str, Any], float]],
) -> Dict[str, Any]:
    """
    Grade a completed episode.

    Args:
        history: List of (action_dict, reward) tuples from the episode.

    Returns:
        Dict with mean_reward, best_post_format, learned_best_time,
        improvement_rate, passed.
    """
    if not history:
        return {
            "mean_reward": 0.0,
            "best_post_format": "unknown",
            "learned_best_time": "unknown",
            "improvement_rate": 0.0,
            "passed": False,
        }

    rewards = [r for _, r in history]
    mean_reward = float(np.mean(rewards))

    # Best post format by average reward
    format_rewards: Dict[str, List[float]] = {}
    time_rewards:   Dict[str, List[float]] = {}
    for action, reward in history:
        fmt = action.get("format", "unknown")
        t   = action.get("post_time", "unknown")
        format_rewards.setdefault(fmt, []).append(reward)
        time_rewards.setdefault(t, []).append(reward)

    best_post_format = max(
        format_rewards, key=lambda f: float(np.mean(format_rewards[f]))
    )
    learned_best_time = max(
        time_rewards, key=lambda t: float(np.mean(time_rewards[t]))
    )

    # Improvement rate: mean of last 5 vs first 5
    first_5 = rewards[:5]
    last_5  = rewards[5:]
    improvement_rate = (
        float(np.mean(last_5)) - float(np.mean(first_5))
        if first_5 and last_5 else 0.0
    )

    return {
        "mean_reward":       round(mean_reward, 2),
        "best_post_format":  best_post_format,
        "learned_best_time": learned_best_time,
        "improvement_rate":  round(improvement_rate, 2),
        "passed":            mean_reward > 0.4,
    }
