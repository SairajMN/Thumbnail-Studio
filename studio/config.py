from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    nebius_api_key: str = ""  # every model call goes to Nebius Token Factory, on this key
    openrouter_api_key: str = ""  # OpenRouter key for reference-image generation ("make it like that")
    imggen_model: str = "openai/gpt-image-2"
    imggen_fallbacks: tuple[str, ...] = ("minimax/minimax-h3-image", "openai/gpt-image-2.5-sunburst")
    imggen_est_usd: float = 0.01
    budget_usd: float = 0.15  # most one cover may spend on models; checked before every call at its worst case
    data_dir: Path = ROOT / "data"  # covers/<id>/: cover.json, frame.png, cover.png


settings = Settings()
