from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    HF_MODEL_ID: str
    DEVICE: str
    MONGO_URI: str
    MONGO_DB_NAME: str
    INTERNAL_API_KEY: str
    MAX_HISTORY_LENGTH: int
    MAX_NEW_TOKENS: int

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()