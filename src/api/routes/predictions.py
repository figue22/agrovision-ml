from fastapi import APIRouter, HTTPException

from src.schemas.prediction import PredictionRequest, PredictionResponse
from src.api.routes.health import get_prediction_service

router = APIRouter(prefix="/predict", tags=["Predictions"])


@router.post(
    "",
    response_model=PredictionResponse,
    summary="Generar predicción de rendimiento",
    description=(
        "Genera una predicción de rendimiento para un cultivo en una parcela específica. "
        "Usa el modelo seleccionado (xgboost, lstm o ensemble). "
        "En esta fase retorna datos stub; modelos reales se integran en HU-037 a HU-039."
    ),
)
async def create_prediction(request: PredictionRequest) -> PredictionResponse:
    try:
        service = get_prediction_service()
        prediction = await service.predict(request)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar predicción: {str(e)}")
