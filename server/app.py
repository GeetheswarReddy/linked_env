"""
FastAPI + WebSocket server for LinkedInEnv.

Exposes the environment over the openenv-core HTTP/WebSocket protocol.

Endpoints (provided by create_app):
    POST /reset        — reset the environment
    POST /step         — execute an action
    GET  /state        — current state
    GET  /schema       — action / observation schemas
    GET  /health       — health check
    WS   /ws           — persistent WebSocket session

Additional endpoints:
    GET  /tasks        — list all tasks with grader info
    POST /grade        — run a grader on episode history
"""

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError("openenv-core is required. Install with: pip install openenv-core") from e

try:
    from env.linkedin_env import LinkedInAction, LinkedInEnvironment, LinkedInObservation
    from env.graders import grade_content_optimization, grade_follower_growth, grade_viral_post
except ModuleNotFoundError:
    from linkedin_env import LinkedInAction, LinkedInEnvironment, LinkedInObservation
    from graders import grade_content_optimization, grade_follower_growth, grade_viral_post

from fastapi import FastAPI
from fastapi.routing import APIRouter
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Tuple

app = create_app(
    LinkedInEnvironment,
    LinkedInAction,
    LinkedInObservation,
    env_name="linkedin_env",
    max_concurrent_envs=5,
)

# ── Task registry ─────────────────────────────────────────────────────────────

TASKS = [
    {
        "id": "content_optimization",
        "description": "Maximise mean engagement reward across a 10-step episode.",
        "difficulty": "easy",
        "max_steps": 10,
        "score_field": "mean_reward",
        "pass_threshold": 0.4,
        "grader": "env.graders.grader_content_optimization",
    },
    {
        "id": "follower_growth",
        "description": "Achieve at least 3 high-engagement posts (>=0.6) to drive follower growth.",
        "difficulty": "medium",
        "max_steps": 10,
        "score_field": "growth_score",
        "pass_threshold": 0.6,
        "grader": "env.graders.grader_follower_growth",
    },
    {
        "id": "viral_post",
        "description": "Find the optimal post combination to achieve a single viral post (>=0.7 engagement).",
        "difficulty": "hard",
        "max_steps": 10,
        "score_field": "best_reward",
        "pass_threshold": 0.7,
        "grader": "env.graders.grader_viral_post",
    },
]

_GRADER_FN = {
    "content_optimization": grade_content_optimization,
    "follower_growth":      grade_follower_growth,
    "viral_post":           grade_viral_post,
}


# ── /tasks endpoint ───────────────────────────────────────────────────────────

@app.get("/tasks", tags=["Tasks"])
def list_tasks() -> Dict[str, Any]:
    """Return all available tasks with their grader information."""
    return {"tasks": TASKS}


# ── /grade endpoint ───────────────────────────────────────────────────────────

class GradeRequest(BaseModel):
    task: str
    history: List[Tuple[Dict[str, Any], float]] = []


@app.post("/grade", tags=["Tasks"])
def grade(request: GradeRequest) -> Dict[str, Any]:
    """
    Run the grader for a given task on provided episode history.

    Returns a result dict containing a normalised score in [0.0, 1.0].
    """
    grader_fn = _GRADER_FN.get(request.task)
    if grader_fn is None:
        return {
            "error": f"Unknown task '{request.task}'. Valid tasks: {list(_GRADER_FN.keys())}",
            "score": 0.0,
            "passed": False,
        }
    score = float(grader_fn(request.history))
    score = min(max(score, 0.0), 1.0)
    task_meta = next((t for t in TASKS if t["id"] == request.task), {})
    return {
        "score": round(score, 3),
        "passed": score >= task_meta.get("pass_threshold", 0.4),
    }


def main() -> None:
    """Entry point for uv run and openenv tooling."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
