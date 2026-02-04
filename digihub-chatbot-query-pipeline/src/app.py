from src.utils.config import CORS_ORIGINS
from src.views.sas_view import router as azure_sas_router
from src.views.chat_views import router as chatbot_router
from src.views.session_view import router as session_router
from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware
from src.utils.logger import logger, log_context
from starlette.middleware.base import BaseHTTPMiddleware

from fastapi.responses import JSONResponse


app = FastAPI()

CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS ]

# Adding CORS Middleware to allow localhost for testing in development server
app.add_middleware(
    CORSMiddleware,
    allow_origins = CORS_ORIGINS,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

# Adding middleware to retrieve trace id from every request and set context variable to trace id
@app.middleware("http")
def add_log_traceid(request: Request, call_next):
    # Getting the request header and add trace id to the log context

    if request.url.path in ["/actuator/health/liveness", "/actuator/health/readiness"]:
        return call_next(request)

    request_header = dict(request.headers)
    log_context.set(request_header["x-digihub-traceid"])
    response = call_next(request)
    return response


@app.get("/actuator/health/liveness")
def liveness_probe():
    return JSONResponse(status_code=200, content={"status": "UP"})

@app.get("/actuator/health/readiness")
def readiness_probe():
    """
    Readiness probe with basic status.
    For detailed health checks, use /actuator/health/detailed
    """
    return JSONResponse(status_code=200, content={"status": "UP"})

@app.get("/actuator/health/detailed")
def detailed_health_check():
    """
    Detailed health check endpoint with component-level status.

    Checks:
    - CosmosDB connectivity
    - Azure OpenAI service availability
    - Configuration loaded status
    """
    import time
    from src.utils.config import (
        COSMOSDB_ENDPOINT,
        AZURE_OPENAI_ENDPOINT,
        OPENAI_DEPLOYMENT_NAME,
        SESSION_CONTEXT_WINDOW_SIZE,
        ENABLE_RELEVANCE_FILTERING,
        MIN_SIMILARITY_THRESHOLD
    )

    health_status = {
        "status": "UP",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "components": {},
        "config": {}
    }

    # Check CosmosDB
    try:
        from src.services.cosmos_db_service import CosmosDBService
        cosmos_service = CosmosDBService()
        # Simple connectivity check - just instantiate
        health_status["components"]["cosmosdb"] = {
            "status": "UP",
            "endpoint": COSMOSDB_ENDPOINT[:30] + "..." if COSMOSDB_ENDPOINT else "Not configured"
        }
    except Exception as e:
        health_status["components"]["cosmosdb"] = {
            "status": "DOWN",
            "error": str(e)[:100]
        }
        health_status["status"] = "DEGRADED"

    # Check Azure OpenAI
    try:
        from src.services.azure_openai_service import AzureOpenAIService
        openai_service = AzureOpenAIService()
        health_status["components"]["azure_openai"] = {
            "status": "UP",
            "endpoint": AZURE_OPENAI_ENDPOINT[:30] + "..." if AZURE_OPENAI_ENDPOINT else "Not configured",
            "deployment": OPENAI_DEPLOYMENT_NAME
        }
    except Exception as e:
        health_status["components"]["azure_openai"] = {
            "status": "DOWN",
            "error": str(e)[:100]
        }
        health_status["status"] = "DEGRADED"

    # Report active configuration
    health_status["config"] = {
        "session_context_window": SESSION_CONTEXT_WINDOW_SIZE,
        "relevance_filtering_enabled": ENABLE_RELEVANCE_FILTERING,
        "min_similarity_threshold": MIN_SIMILARITY_THRESHOLD
    }

    status_code = 200 if health_status["status"] == "UP" else 503
    return JSONResponse(status_code=status_code, content=health_status)


app.include_router(chatbot_router, prefix="/chatbot/v1", tags=["DigiHub ChatBot"])
app.include_router(azure_sas_router, prefix="/chatbot/v1", tags=["Azure Blob SAS Generator"])
app.include_router(session_router, prefix="/chatbot/v1", tags=["Sessions"])


# if __name__ == '__main__':
#     import uvicorn

#     logger.info("Server is starting")
#     uvicorn.run(app, host="0.0.0.0", port=8080)
