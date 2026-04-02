import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import get_settings
from src.api.routes import health_router, predictions_router, models_router
from src.api.middleware.logging_middleware import LoggingMiddleware

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agrovision-ml")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup y shutdown del servicio."""
    settings = get_settings()
    logger.info("Iniciando %s v%s [%s]", settings.APP_NAME, settings.APP_VERSION, settings.APP_ENV)
    logger.info("Modelos path: %s", settings.MODEL_PATH)
    logger.info("Swagger UI disponible en: http://localhost:%s/docs", settings.APP_PORT)

    yield

    logger.info("Apagando %s", settings.APP_NAME)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Microservicio de Machine Learning para predicción de rendimiento agrícola. "
            "Modelos: XGBoost (features estáticas), LSTM (series temporales climáticas), "
            "Ensemble (combinación ponderada). Parte del sistema AgroVision Predictor & RAG-Support."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:4000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Logging middleware
    app.add_middleware(LoggingMiddleware)

    # Routes
    app.include_router(health_router)
    app.include_router(predictions_router)
    app.include_router(models_router)

    return app


app = create_app()
