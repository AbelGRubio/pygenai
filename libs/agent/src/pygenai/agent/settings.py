"""Agent settings and default runtime configuration."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

__version__ = "0.1.0"


class AgentSettings(BaseSettings):
    """Settings parsed from environment variables for the agent runtime."""

    jwt_protected: bool = Field(default=True, alias="JWT_PROTECTED")
    jwt_url: str = Field(
        default="http://localhost:9000/api/token",
        alias="JWT_URL",
    )
    client_id: str | None = Field(default=None, alias="CLIENT_ID")
    client_secret: str | None = Field(default=None, alias="CLIENT_SECRET")

    mcp_url: str = Field(default="http://localhost:8000/mcp", alias="MCP_LOCAL_URL")

    model_name: str = Field(default="gemini/gemini-3.6-flash", alias="MODEL_NAME")
    llm_api_key: SecretStr = Field(default=SecretStr(""), alias="LITELLM_MASTER_KEY")
    model_base_url: str = Field(default="http://localhost:5000", alias="MODEL_BASE_URL")

    model_config = SettingsConfigDict(
        env_prefix="",
        populate_by_name=True,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> AgentSettings:
    """Return a cached settings instance."""
    return AgentSettings()
