from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Configuración del servicio ML de AgroVision."""

    # General
    APP_NAME: str = "AgroVision ML Service"
    APP_VERSION: str = "0.1.0"
    APP_ENV: str = "development"
    APP_PORT: int = 8000
    DEBUG: bool = True

    # Base de datos (lectura de parcelas, cultivos, clima)
    DATABASE_URL: str = "postgresql://agrovision_user:agrovision_pass_2026@localhost:5432/agrovision_db"

    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "agrovision-predictions"

    # Modelos
    MODEL_PATH: str = "./data/models"

    # Backend API (para comunicación con NestJS)
    BACKEND_API_URL: str = "http://localhost:4000/api/v1"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
