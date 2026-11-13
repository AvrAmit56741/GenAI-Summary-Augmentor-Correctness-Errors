"""API keys loaded from the repo-root .env file (never hardcode keys in code).

Create a `.env` file in the repository root (it is git-ignored) with:
    OPENAI_API_KEY=...
    GEMINI_API_KEY=...
    CLAUDE_API_KEY=...
    AZURE_OPENAI_API_KEY_4o=...
Variables already set in the environment take precedence over .env.
"""
import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


_load_env_file(ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
AZURE_OPENAI_API_KEY_4o = os.getenv("AZURE_OPENAI_API_KEY_4o", "")
