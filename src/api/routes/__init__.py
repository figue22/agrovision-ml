from src.api.routes.health import router as health_router
from src.api.routes.predictions import router as predictions_router
from src.api.routes.models import router as models_router

__all__ = ["health_router", "predictions_router", "models_router"]
