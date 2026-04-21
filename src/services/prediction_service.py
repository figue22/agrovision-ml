import os
import time
from datetime import datetime

from src.config.settings import get_settings
from src.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    ModelInfo,
    NivelRiesgo,
    TipoModelo,
)


class PredictionService:
    """Servicio de predicción de rendimiento de cultivos.

    En esta fase (HU-012) se implementa la estructura base.
    Los modelos reales (XGBoost, LSTM, Ensemble) se entrenan
    e integran en las HU-037 a HU-043.
    """

    MODEL_VERSION = "0.1.0-stub"

    def __init__(self):
        self.settings = get_settings()
        self._models_loaded: dict[str, bool] = {
            "xgboost": False,
            "lstm": False,
            "ensemble": False,
        }
        self._start_time = time.time()
        self._load_models()

    def _load_models(self) -> None:
        """Intenta cargar modelos serializados desde disco."""
        model_path = self.settings.MODEL_PATH

        if os.path.exists(os.path.join(model_path, "xgboost_model.joblib")):
            self._models_loaded["xgboost"] = True

        if os.path.exists(os.path.join(model_path, "lstm_model.h5")):
            self._models_loaded["lstm"] = True

        if self._models_loaded["xgboost"] or self._models_loaded["lstm"]:
            self._models_loaded["ensemble"] = True

    @property
    def models_loaded_count(self) -> int:
        return sum(1 for v in self._models_loaded.values() if v)

    @property
    def uptime(self) -> float:
        return time.time() - self._start_time

    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Genera una predicción de rendimiento.

        En esta fase retorna datos stub. Los modelos reales
        se integran en HU-037 (XGBoost), HU-038 (LSTM), HU-039 (Ensemble).
        """
        # TODO: Cargar datos reales de parcela, cultivo y clima desde la BD
        # TODO: Feature engineering pipeline
        # TODO: Ejecutar modelo real

        # Stub: simula una predicción
        rendimiento = 4.5  # ton/ha placeholder
        confianza = 72.0
        inferior = rendimiento * 0.85
        superior = rendimiento * 1.15

        return PredictionResponse(
            parcela_id=request.parcela_id,
            cultivo_parcela_id=request.cultivo_parcela_id,
            tipo_cultivo_id=request.tipo_cultivo_id,
            version_modelo=self.MODEL_VERSION,
            tipo_modelo=request.modelo.value,
            rendimiento_predicho_ton=round(rendimiento, 2),
            puntaje_confianza=confianza,
            intervalo_conf_inferior=round(inferior, 2),
            intervalo_conf_superior=round(superior, 2),
            nivel_riesgo=NivelRiesgo.MEDIO,
            factores_riesgo={
                "clima": "Sin datos climáticos suficientes",
                "nota": "Predicción stub — modelos reales pendientes HU-037",
            },
            datos_clima_usados={},
            importancia_features={
                "nota": "Feature importance disponible después del entrenamiento",
            },
            fecha_prediccion=datetime.utcnow(),
        )

    def get_available_models(self) -> list[ModelInfo]:
        """Retorna la lista de modelos disponibles."""
        models = [
            ModelInfo(
                nombre="XGBoost — Predicción Base",
                tipo=TipoModelo.XGBOOST,
                version=self.MODEL_VERSION,
                estado="cargado" if self._models_loaded["xgboost"] else "pendiente",
                descripcion="Modelo gradient boosting para features estáticas de suelo, cultivo y clima.",
                metricas={"rmse": None, "r2": None},
            ),
            ModelInfo(
                nombre="LSTM — Series Temporales",
                tipo=TipoModelo.LSTM,
                version=self.MODEL_VERSION,
                estado="cargado" if self._models_loaded["lstm"] else "pendiente",
                descripcion="Red neuronal recurrente para series temporales climáticas (90 días).",
                metricas={"rmse": None, "r2": None},
            ),
            ModelInfo(
                nombre="Ensemble — Combinación Ponderada",
                tipo=TipoModelo.ENSEMBLE,
                version=self.MODEL_VERSION,
                estado="cargado" if self._models_loaded["ensemble"] else "pendiente",
                descripcion="Combinación: XGBoost (60%) + LSTM (40%). Score de confianza y riesgo.",
                metricas={"rmse": None, "r2": None},
            ),
        ]
        return models
