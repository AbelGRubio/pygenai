"""Application settings.

========================================================================================================================
Name:         src/agile_monkeys/settings.py
Description:  Load project environment variables with Pydantic settings
Project:      agile-monkeys
Date:         2026-11-18
Status:       Development

Copyright ©2026 All rights reserved.
========================================================================================================================
"""

from functools import cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment settings used by Agile Monkeys commands."""

    hiring_lab_base_url: str = Field(alias="HIRING_LAB_BASE_URL")
    hiring_lab_project_key: str = Field(alias="HIRING_LAB_PROJECT_KEY")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@cache
def get_settings() -> Settings:
    """Return the application settings, loading them once per process."""
    return Settings()
