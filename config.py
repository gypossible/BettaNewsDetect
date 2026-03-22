"""Global configuration management for the public opinion analysis system."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""

    # LLM Settings
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Search Settings
    SEARCH_MAX_RESULTS: int = int(os.getenv("SEARCH_MAX_RESULTS", "10"))

    # Server Settings
    FLASK_HOST: str = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT: int = int(os.getenv("FLASK_PORT", "5001"))
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "true").lower() == "true"

    # Report Settings
    REPORT_OUTPUT_DIR: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "final_reports"
    )

    # Forum Settings
    FORUM_MAX_ROUNDS: int = int(os.getenv("FORUM_MAX_ROUNDS", "3"))

    @classmethod
    def validate(cls) -> list[str]:
        """Validate required configuration. Returns list of warnings."""
        warnings = []
        if not cls.LLM_API_KEY:
            warnings.append("LLM_API_KEY is not set. Please configure it in .env file.")
        return warnings
