# =========================
# Stage 1: Builder
# =========================
FROM xscntrlregprod.azurecr.io/sis/digihub/python:latest-dev@sha256:d218aa64d6a4dbeb3065c65ccb6de152bfa8c4aecb7e88d9b28b341b37d2c463 AS build

WORKDIR /app

RUN python -m venv venv
ENV PATH="/app/venv/bin":$PATH
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt \
    && pip install --upgrade opencv-python-headless==4.12.0.88

# Final stage
FROM xscntrlregprod.azurecr.io/sis/digihub/python:latest-dev@sha256:d218aa64d6a4dbeb3065c65ccb6de152bfa8c4aecb7e88d9b28b341b37d2c463

WORKDIR /app

#Copy all code and dependencies
COPY --chown=nonroot ./src /app/src
COPY --chown=nonroot --from=build /app/venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Set permissions right after copying
RUN chmod -R 775 /app/src

# Set environment variables
ENV PYTHONPATH=/app:/app/src

ENTRYPOINT ["python", "/app/src/app.py"]