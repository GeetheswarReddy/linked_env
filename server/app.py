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

Run locally:
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

HF Spaces / Docker:
    CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
"""

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError("openenv-core is required. Install with: pip install openenv-core") from e

try:
    from env.linkedin_env import LinkedInAction, LinkedInEnvironment, LinkedInObservation
except ModuleNotFoundError:
    from linkedin_env import LinkedInAction, LinkedInEnvironment, LinkedInObservation

app = create_app(
    LinkedInEnvironment,
    LinkedInAction,
    LinkedInObservation,
    env_name="linkedin_env",
    max_concurrent_envs=5,
)


def main() -> None:
    """Entry point for uv run and openenv tooling."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
