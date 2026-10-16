from fastapi import APIRouter
from .health import router as health_router
from .project import router as project_router
from .dataset import router as dataset_router
from .models import router as models_router
from .training import router as training_router
from .evaluation import router as evaluation_router
from .prediction import router as prediction_router
from .analytics import router as analytics_router
from .export import router as export_router
from .activity import router as activity_router
from .predictions import router as predictions_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(project_router)
api_router.include_router(dataset_router)
api_router.include_router(models_router)
api_router.include_router(training_router, prefix="/train")  # /train/...
api_router.include_router(training_router, prefix="/training")  # alias /training/...
api_router.include_router(evaluation_router)
api_router.include_router(prediction_router)
api_router.include_router(analytics_router)
api_router.include_router(export_router)
api_router.include_router(activity_router)
api_router.include_router(predictions_router)

__all__ = ["api_router"]


