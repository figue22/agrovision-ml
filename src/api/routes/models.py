from fastapi import APIRouter

from src.schemas.prediction import ModelsListResponse
from src.api.routes.health import get_prediction_service

router = APIRouter(prefix="/models", tags=["Models"])


@router.get(
    "",
    response_model=ModelsListResponse,
    summary="Listar modelos disponibles",
    description="Retorna la lista de modelos ML disponibles con su estado y métricas.",
)
async def list_models() -> ModelsListResponse:
    service = get_prediction_service()
    models = service.get_available_models()
    return ModelsListResponse(modelos=models, total=len(models))
