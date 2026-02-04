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
# You can add checks here (e.g., DB connection, external service availability)
    return JSONResponse(status_code=200, content={"status": "UP"})


app.include_router(chatbot_router, prefix="/chatbot/v1", tags=["DigiHub ChatBot"])
app.include_router(azure_sas_router, prefix="/chatbot/v1", tags=["Azure Blob SAS Generator"])
app.include_router(session_router, prefix="/chatbot/v1", tags=["Sessions"])


# if __name__ == '__main__':
#     import uvicorn

#     logger.info("Server is starting")
#     uvicorn.run(app, host="0.0.0.0", port=8080)
