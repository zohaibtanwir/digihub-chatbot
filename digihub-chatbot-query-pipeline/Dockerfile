# Use the official Python image from the Docker Hub
FROM python:3.12-alpine

# Create a non-root user and group
RUN addgroup -S digihubuser && adduser -S digihubuser -G digihubuser

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app with permissions set to 700
COPY . /app/

#Set ownership and permissions
RUN chown -R digihubuser:digihubuser /app && chmod -R 775 /app

# Set environment variables for the application
ENV PYTHONPATH=/app:/app/src

# Install Java and other dependencies
RUN apk update && apk --no-cache add curl

# Upgrade OpenSSL and install Python dependencies
RUN apk update && \
    apk upgrade --no-cache && \
    pip install --no-cache-dir gunicorn uvicorn[standard] -r /app/requirements.txt

# Switch to the non-root user
USER digihubuser

# Install Python dependencies
#RUN pip install --no-cache-dir -r /app/requirements.txt

# Set the working directory and command
WORKDIR /app
#CMD ["python", "/app/src/app.py"]

# Set default command
CMD ["gunicorn", "src.app:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080","--timeout","1200"]