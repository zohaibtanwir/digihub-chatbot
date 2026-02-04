import os
from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse
from azure.cosmos import CosmosClient
from fastapi.middleware.cors import CORSMiddleware
from src.utils.config import cosmos_client,COSMOSDB_VECTOR_INDEX,database,COSMOSDB_DEBOUNCER_CONTAINER_NAME,origins,EVENT_HUB_CONNECTION_STRING,EVENT_HUB_NAME
from src.utils.sharepoint_subscription import SharePointSubscriptionManager
from src.scripts.eventhub import download_file
from datetime import datetime
from src.utils.logger import logger
import asyncio
from src.scripts.eventhub import start_eventhub_listener
from src.scripts.doclingmodel import download_docling
from contextlib import asynccontextmanager
from fastapi.responses import PlainTextResponse
from azure.eventhub import EventHubProducerClient, EventData 
import json 
from src.utils.pydantic import WebhookPayload
from pydantic import ValidationError



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    docling_model_path = "/app/docling/docling-model"
    if not os.path.exists(docling_model_path):
        logger.info(f"{docling_model_path} not found. Downloading model...")
        download_docling()
    else:
        logger.info(f"{docling_model_path} already exists.")

    asyncio.create_task(start_eventhub_listener())

    yield  # Application runs here

    # Shutdown logic (if needed)
    logger.info("Application is shutting down...")

app = FastAPI(lifespan=lifespan)
# Adding CORS Middleware to allow localhost for testing in development server
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)
@app.post("/digihub/sharepoint/v1/webhook")
async def webhook(request: Request):
    logger.info(f"Request received: {request}")
    validation_token = request.query_params.get("validationToken")
    if validation_token:
        logger.info("Validation token received.")
        return PlainTextResponse(content=validation_token, status_code=200)

    try:
        body = await request.json()
        logger.info(f"body : {body}")
        try:
            payload = WebhookPayload(**body)  # Pydantic validation
        except ValidationError as ve:
            logger.warning(f"SP-WebHook Payload validation failed: {ve}")
            return PlainTextResponse(content="Invalid payload structure", status_code=400)

        logger.info(f"Validated payload: {payload.json()}")

        producer = EventHubProducerClient.from_connection_string(
            conn_str=EVENT_HUB_CONNECTION_STRING,
            eventhub_name=EVENT_HUB_NAME
        )

        event_data_str = payload.json()
        with producer:
            event_data_batch = producer.create_batch()
            event_data_batch.add(EventData(event_data_str))
            producer.send_batch(event_data_batch)

        return PlainTextResponse(content="Event sent to Event Hub", status_code=202)

    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        return PlainTextResponse(content="Internal server error", status_code=500)

@app.post("/chatbot/v1/sharepoint/subscription")
def manage_sharepoint_subscription():
    manager = SharePointSubscriptionManager()
    manager.manage_subscriptions()
    return {"message": "Subscriptions renewed successfully"}

@app.get("/chatbot/v1/sharepoint/debounce")
def sharepoint_debounce():
    """
    This endpoint processes debounced updates from the Cosmos DB container.

    It checks for items with a scheduled trigger time that has passed and processes them.
    """
    try:
        logger.info("Processing debounced updates...")
        debouncer_container = database.get_container_client(COSMOSDB_DEBOUNCER_CONTAINER_NAME)
        now = datetime.utcnow().isoformat()

        query = "SELECT c.drive_id, c.item_id, c.id, c.item_path, c.scheduled_trigger_time, c.status FROM c WHERE c.scheduled_trigger_time <= @now AND c.status = 'pending'"
        parameters = [{"name": "@now", "value": now}]
        items_to_process = list(debouncer_container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))

        for item in items_to_process:
            drive_id = item["drive_id"]
            Drivename = item["id"]
            filepath=item["item_path"]
            # Check if the scheduled trigger time is beyond 5 minutes from the current time
            scheduled_trigger_time_str = item["scheduled_trigger_time"]
            
            if scheduled_trigger_time_str.endswith('Z'):
                scheduled_trigger_time_dt = datetime.strptime(scheduled_trigger_time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
            else:
                scheduled_trigger_time_dt = datetime.strptime(scheduled_trigger_time_str, "%Y-%m-%dT%H:%M:%S.%f")


            if (datetime.utcnow() - scheduled_trigger_time_dt).total_seconds() >= 300:
                logger.info(f"Processing debounced update for: {Drivename}")
                
                # Delete corresponding vector index entries before reindexing
                index_container = database.get_container_client(COSMOSDB_VECTOR_INDEX)
                index_query = "SELECT c.id, c.serviceName, c.metadata FROM c WHERE c.metadata.filepath = @filepath"
                index_params = [{"name": "@filepath", "value": filepath}]
                index_items = list(index_container.query_items(
                    query=index_query,
                    parameters=index_params,
                    enable_cross_partition_query=True
                ))
                
                for index_item in index_items:
                    logger.info(f"Deleting vector index for: {filepath}")
                    partitionkey=f"{index_item['serviceName']}-{index_item['metadata']['filename'].replace(' ', '')}"
                    index_container.delete_item(item=index_item['id'], partition_key=partitionkey)
                
                download_file(drive_id,item["item_id"],item["item_path"],Drivename)  # Your existing logic here

                item["status"] = "processed"
                debouncer_container.upsert_item(body=item)

        return {"status": "success", "processed_items": len(items_to_process)}
    
    except Exception as e:
        logger.error(f"Error processing debounced updates: {str(e)}", exc_info=True)
        return {"status": "failure", "message": str(e)}
    
@app.get("/actuator/health/liveness")
def liveness_probe():
    return JSONResponse(status_code=200, content={"status": "UP"})

@app.get("/actuator/health/readiness")
def readiness_probe():
# You can add checks here (e.g., DB connection, external service availability)
    return JSONResponse(status_code=200, content={"status": "UP"})
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
