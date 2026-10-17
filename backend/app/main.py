import logging
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.api import api_router

# Configure server-side logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("medai.backend")

# Initialize FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Research Platform API for 'Medical Image Classification Using Transfer Learning: "
        "A Comparative Study of ResNet50 and EfficientNetB0 on PneumoniaMNIST'. "
        "Strictly for Educational and Academic Deep Learning Research."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

import os

# Configure CORS allowed origins for research platform frontend clients
configured_origins = []
if settings.FRONTEND_ORIGIN:
    configured_origins.extend([o.strip() for o in settings.FRONTEND_ORIGIN.split(",") if o.strip()])

raw_origins = [
    *configured_origins,
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:5174",
    "http://localhost:5174",
    "http://127.0.0.1:5175",
    "http://localhost:5175",
    "http://127.0.0.1:5176",
    "http://localhost:5176",
]

# Clean trailing slashes, strip whitespace, and preserve order while deduplicating
origins: list[str] = []
for origin in raw_origins:
    if isinstance(origin, str):
        cleaned_origin = origin.strip().rstrip("/")
        if cleaned_origin and cleaned_origin not in origins:
            origins.append(cleaned_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"^https://.*(\.vercel\.app|\.onrender\.com)$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global unhandled exception handler to avoid leaking internal stack traces
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled server exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred while processing the research request. Technical details logged server-side.",
        },
    )


from app.api.health import get_health
from app.schemas.health import HealthResponse

# Attach aggregate API routers under /api prefix (e.g. /api/health, /api/predict, etc.)
app.include_router(api_router, prefix=settings.API_PREFIX)


# Root discovery endpoints (supports /, /api, /api/)
@app.get("/", tags=["General"])
@app.get("/api", tags=["General"])
@app.get("/api/", tags=["General"])
def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "purpose": "Educational & Research Use Only",
        "documentation": "/docs",
        "health": f"{settings.API_PREFIX}/health",
    }


# Direct health check aliases at root level (supports /health, /health/)
# in addition to standard /api/health from api_router
@app.get("/health", response_model=HealthResponse, tags=["Health"])
@app.get("/health/", response_model=HealthResponse, tags=["Health"])
def health_root_alias():
    """Direct root health check alias so both /health and /api/health return full health status."""
    return get_health()


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", settings.PORT))
    host = os.environ.get("HOST", settings.HOST)
    uvicorn.run("app.main:app", host=host, port=port, reload=False)

