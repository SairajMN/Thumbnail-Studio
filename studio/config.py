from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    nebius_api_key: str = ""  # every model call goes to Nebius Token Factory, on this key
    budget_usd: float = 0.15  # most one cover may spend on models; checked before every call at its worst case
    data_dir: Path = ROOT / "data"  # covers/<id>/: cover.json, frame.png, cover.png


settings = Settings()
