from datetime import datetime
from fastapi import APIRouter, Depends

from src.config.settings import Settings, get_settings
from src.schemas.prediction import HealthResponse
from src.services.prediction_service import PredictionService

router = APIRouter(tags=["Health"])

# Singleton del servicio
_prediction_service = PredictionService()


def get_prediction_service() -> PredictionService:
    return _prediction_service


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check del servicio ML",
    description="Verifica el estado del servicio, modelos cargados y conectividad.",
)
async def health_check(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    service = get_prediction_service()

    return HealthResponse(
        status="ok",
        service="agrovision-ml",
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
        models_loaded=service.models_loaded_count,
        database_connected=False,  # TODO: verificar conexión real en HU-037
        uptime_seconds=round(service.uptime, 2),
        timestamp=datetime.utcnow(),
    )
