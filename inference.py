"""
LinkedInEnv inference entry point — Meta PyTorch OpenEnv Hackathon.

Starts the environment server as a subprocess, then runs a full 10-step episode
where an LLM agent (via OpenAI-compatible API) decides what to post on LinkedIn
based on the current state and receives a simulated engagement reward.

Required environment variables:
    HF_TOKEN     — Hugging Face token (mandatory)

Optional environment variables:
    API_BASE_URL — OpenAI-compatible base URL  (default: https://api.openai.com/v1)
    MODEL_NAME   — Model to use                (default: gpt-4.1-mini)
    ENV_PORT     — Port for the local server   (default: 7860)

Usage:
    python inference.py
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

# ── env vars ──────────────────────────────────────────────────────────────────
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME:   str = os.environ.get("MODEL_NAME",   "gpt-4.1-mini")
HF_TOKEN:     str = os.environ.get("HF_TOKEN", "")
ENV_PORT:     int = int(os.environ.get("ENV_PORT", "7860"))

EPISODE_LENGTH = 10

FORMAT_OPTIONS = ["story", "list", "hot_take", "question", "framework"]
HOOK_OPTIONS   = ["bold_claim", "question", "statistic", "personal_open"]
TIME_OPTIONS   = ["mon_morning", "tue_morning", "wed_morning",
                  "thu_morning", "fri_morning", "weekend"]
LENGTH_OPTIONS = ["short", "medium", "long"]

ACTION_SCHEMA = {
    "topic":        "string — topic or theme of the post (free text)",
    "format":       f"one of: {FORMAT_OPTIONS}",
    "hook_type":    f"one of: {HOOK_OPTIONS}",
    "post_time":    f"one of: {TIME_OPTIONS}",
    "has_question": "boolean — whether the post ends with a question",
    "is_personal":  "boolean — whether the post shares a personal story/vulnerability",
    "length":       f"one of: {LENGTH_OPTIONS}",
}


# ── openenv client ────────────────────────────────────────────────────────────

class LinkedInObservationData:
    """Lightweight wrapper around the raw observation dict."""

    def __init__(self, data: Dict[str, Any]):
        obs = data.get("observation", data)
        self.follower_count:       int              = obs.get("follower_count", 1000)
        self.last_posts:           List[Dict]       = obs.get("last_posts", [])
        self.days_since_last_post: int              = obs.get("days_since_last_post", 0)
        self.current_niche:        str              = obs.get("current_niche", "tech")
        self.episode_step:         int              = obs.get("episode_step", 0)
        self.reward:               Optional[float]  = data.get("reward")
        self.done:                 bool             = data.get("done", False)


class LinkedInEnvClient(EnvClient):
    """openenv-core client for LinkedInEnv."""

    def _step_payload(self, action: Dict[str, Any]) -> Dict[str, Any]:
        return action

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult:
        obs = LinkedInObservationData(payload)
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )


# ── LLM agent ─────────────────────────────────────────────────────────────────

def build_prompt(obs: LinkedInObservationData) -> str:
    last_posts_text = ""
    if obs.last_posts:
        for i, p in enumerate(obs.last_posts, 1):
            last_posts_text += (
                f"  Post {i}: format={p.get('format')}, hook={p.get('hook_type')}, "
                f"time={p.get('post_time')}, personal={p.get('is_personal')}, "
                f"engagement={p.get('engagement_score', 0.0):.2f}\n"
            )
    else:
        last_posts_text = "  None yet.\n"

    return f"""You are an expert LinkedIn content strategist.

## Current State
- Follower count: {obs.follower_count}
- Content niche: {obs.current_niche}
- Days since last post: {obs.days_since_last_post}
- Episode step: {obs.episode_step} / {EPISODE_LENGTH}
- Recent posts:
{last_posts_text}

## Task
Decide the next LinkedIn post to maximise audience engagement.

## Action fields (all required)
{json.dumps(ACTION_SCHEMA, indent=2)}

## Instructions
- Consider which format, hook and time performs best for the "{obs.current_niche}" niche.
- Tuesday and Wednesday mornings get 1.3x reach.
- Personal stories + personal hooks get 1.4x comment rate.
- List format gets 1.2x share rate.
- Respond ONLY with a single valid JSON object. No explanation, no markdown, no extra text.

Example valid response:
{{"topic": "lessons from my first failed startup", "format": "story", "hook_type": "personal_open", "post_time": "tue_morning", "has_question": true, "is_personal": true, "length": "medium"}}
"""


def get_llm_action(client: OpenAI, obs: LinkedInObservationData) -> Dict[str, Any]:
    prompt = build_prompt(obs)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a LinkedIn content optimisation agent. "
                    "Always respond with a single valid JSON object and nothing else."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=300,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ── server lifecycle ──────────────────────────────────────────────────────────

def _wait_for_server(port: int, timeout: float = 30.0) -> None:
    """Poll /health until the server is ready."""
    deadline = time.monotonic() + timeout
    url = f"http://localhost:{port}/health"
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Server on port {port} did not become ready within {timeout}s")


def _start_server(port: int) -> subprocess.Popen:
    """Launch the uvicorn server as a background subprocess."""
    cmd = [
        sys.executable, "-m", "uvicorn",
        "server.app:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--log-level", "warning",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    return proc


# ── main episode loop ─────────────────────────────────────────────────────────

async def run_episode() -> None:
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable is required but not set.")

    openai_client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    print(f"[START] task=linkedin-content-optimisation env=LinkedInEnv model={MODEL_NAME}")

    server_proc: Optional[subprocess.Popen] = None
    rewards: List[float] = []
    steps_done = 0
    success = False
    error_msg: Optional[str] = None

    try:
        # Start server
        server_proc = _start_server(ENV_PORT)
        _wait_for_server(ENV_PORT)

        base_url = f"http://localhost:{ENV_PORT}"

        async with LinkedInEnvClient(base_url=base_url) as env:
            # Reset
            result = await env.reset(seed=42)
            obs: LinkedInObservationData = result.observation

            # Episode loop
            for step in range(1, EPISODE_LENGTH + 1):
                step_error: Optional[str] = None
                action_dict: Optional[Dict[str, Any]] = None
                reward_val: float = 0.0
                done = False

                try:
                    action_dict = get_llm_action(openai_client, obs)
                    result = await env.step(action_dict)
                    obs = result.observation
                    reward_val = float(result.reward) if result.reward is not None else 0.0
                    done = result.done
                    rewards.append(reward_val)
                    steps_done = step
                except Exception as e:
                    step_error = str(e)
                    done = True

                action_str = json.dumps(action_dict) if action_dict else "null"
                print(
                    f"[STEP] step={step} "
                    f"action={action_str} "
                    f"reward={reward_val:.2f} "
                    f"done={str(done).lower()} "
                    f"error={step_error if step_error else 'null'}"
                )

                if done:
                    break

        success = steps_done >= EPISODE_LENGTH and not error_msg

    except Exception as e:
        error_msg = str(e)
        success = False

    finally:
        # Always emit [END]
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={str(success).lower()} "
            f"steps={steps_done} "
            f"rewards={rewards_str}"
        )

        # Shut down server
        if server_proc is not None:
            try:
                os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
            except Exception:
                server_proc.terminate()


if __name__ == "__main__":
    asyncio.run(run_episode())
