"""
Inference Script — LinkedIn Content Optimisation
===================================
MANDATORY environment variables:
    API_BASE_URL        The API endpoint for the LLM.
    MODEL_NAME          The model identifier to use for inference.
    HF_TOKEN            Your Hugging Face / API key.
    IMAGE_NAME          The local Docker image name for the environment.
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from env import LinkedInAction, LinkedInObservation

# ── env vars ──────────────────────────────────────────────────────────────────
IMAGE_NAME   = os.getenv("IMAGE_NAME")
HF_SPACE_ID  = os.getenv("HF_SPACE_ID", "GeethuR/linkedIn_env")
HF_TOKEN     = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

TASK_NAME  = os.getenv("TASK_NAME", "content_optimization")
BENCHMARK  = os.getenv("BENCHMARK", "LinkedInEnv")
MAX_STEPS  = 10
SUCCESS_SCORE_THRESHOLD = 0.4  # mean reward threshold (rewards are in [0, 1])

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

class LinkedInEnvClient(EnvClient[LinkedInAction, LinkedInObservation, State]):
    def _step_payload(self, action: LinkedInAction) -> Dict[str, Any]:
        return action.model_dump(exclude={"metadata"})

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[LinkedInObservation]:
        obs_data = payload.get("observation", payload)
        obs = LinkedInObservation(
            follower_count=obs_data.get("follower_count", 1000),
            last_posts=obs_data.get("last_posts", []),
            days_since_last_post=obs_data.get("days_since_last_post", 0),
            current_niche=obs_data.get("current_niche", "tech"),
            episode_step=obs_data.get("episode_step", 0),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )
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


# ── logging helpers ───────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM agent ─────────────────────────────────────────────────────────────────

def build_prompt(obs: Any, step: int) -> str:
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

    return textwrap.dedent(f"""
        You are an expert LinkedIn content strategist.

        ## Current State
        - Follower count: {obs.follower_count}
        - Content niche: {obs.current_niche}
        - Days since last post: {obs.days_since_last_post}
        - Episode step: {step} / {MAX_STEPS}
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
    """).strip()


def get_llm_action(client: OpenAI, obs: Any, step: int) -> Dict[str, Any]:
    prompt = build_prompt(obs, step)
    try:
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
            stream=False,
        )
        raw = (response.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return {
            "topic": "productivity tips",
            "format": "list",
            "hook_type": "bold_claim",
            "post_time": "tue_morning",
            "has_question": True,
            "is_personal": False,
            "length": "medium",
        }


# ── main episode loop ─────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    if IMAGE_NAME:
        env = await LinkedInEnvClient.from_docker_image(IMAGE_NAME)
    else:
        env = await LinkedInEnvClient.from_env(HF_SPACE_ID)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=TASK_NAME)
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_dict: Optional[Dict[str, Any]] = None
            reward_val: float = 0.0
            done = False
            error: Optional[str] = None

            try:
                action_dict = get_llm_action(client, obs, step)
                result = await env.step(LinkedInAction(**action_dict))
                obs = result.observation
                reward_val = float(result.reward) if result.reward is not None else 0.0
                done = result.done
            except Exception as exc:
                error = str(exc)
                done = True

            rewards.append(reward_val)
            steps_taken = step

            action_str = json.dumps(action_dict) if action_dict else "null"
            log_step(step=step, action=action_str, reward=reward_val, done=done, error=error)

            if done:
                break

        score = sum(rewards) / MAX_STEPS if MAX_STEPS > 0 else 0.0
        score = min(max(score, 0.01), 0.99)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
