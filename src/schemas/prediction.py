from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


# ── Enums ──

class NivelRiesgo(str, Enum):
    BAJO = "bajo"
    MEDIO = "medio"
    ALTO = "alto"
    CRITICO = "critico"


class TipoModelo(str, Enum):
    XGBOOST = "xgboost"
    LSTM = "lstm"
    ENSEMBLE = "ensemble"


# ── Schemas de Predicción ──

class PredictionRequest(BaseModel):
    """Request para generar una predicción de rendimiento."""
    parcela_id: str = Field(..., description="UUID de la parcela")
    cultivo_parcela_id: str = Field(..., description="UUID del cultivo en parcela")
    tipo_cultivo_id: str = Field(..., description="UUID del tipo de cultivo")
    modelo: TipoModelo = Field(default=TipoModelo.ENSEMBLE, description="Tipo de modelo a usar")

    model_config = {"json_schema_extra": {
        "example": {
            "parcela_id": "550e8400-e29b-41d4-a716-446655440000",
            "cultivo_parcela_id": "660e8400-e29b-41d4-a716-446655440001",
            "tipo_cultivo_id": "770e8400-e29b-41d4-a716-446655440002",
            "modelo": "ensemble",
        }
    }}


class PredictionResponse(BaseModel):
    """Respuesta con la predicción de rendimiento."""
    prediccion_id: Optional[str] = None
    parcela_id: str
    cultivo_parcela_id: str
    tipo_cultivo_id: str
    version_modelo: str
    tipo_modelo: str
    rendimiento_predicho_ton: float = Field(..., description="Rendimiento predicho en toneladas/ha")
    puntaje_confianza: float = Field(..., ge=0, le=100, description="Score de confianza 0-100")
    intervalo_conf_inferior: float
    intervalo_conf_superior: float
    nivel_riesgo: NivelRiesgo
    factores_riesgo: dict = Field(default_factory=dict)
    datos_clima_usados: dict = Field(default_factory=dict)
    importancia_features: dict = Field(default_factory=dict)
    fecha_prediccion: datetime


class ModelInfo(BaseModel):
    """Información de un modelo disponible."""
    nombre: str
    tipo: TipoModelo
    version: str
    fecha_entrenamiento: Optional[datetime] = None
    metricas: dict = Field(default_factory=dict)
    estado: str = "disponible"
    descripcion: str = ""


class ModelsListResponse(BaseModel):
    """Lista de modelos disponibles."""
    modelos: list[ModelInfo]
    total: int


# ── Health ──

class HealthResponse(BaseModel):
    """Respuesta del health check."""
    status: str = "ok"
    service: str = "agrovision-ml"
    version: str
    environment: str
    models_loaded: int = 0
    database_connected: bool = False
    uptime_seconds: float = 0
    timestamp: datetime
