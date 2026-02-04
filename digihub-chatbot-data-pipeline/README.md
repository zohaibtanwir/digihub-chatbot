# Digihub chatbot data pipeline service
DigiHub chatbot data pipeline service is responsible for different below functionalities:


**Fetch Files from SharePoint:**
    Documents are fetched from a SharePoint folder.
    Supported formats include .pdf, .docx
**Maintain SharePoint Subscription for Change Notifications**
    A SharePoint subscription is established to monitor changes in the folder.
    The subscription is automatically renewed before expiration to ensure continuous monitoring.
    Change notifications are sent to an Azure Event Hub, which triggers downstream processing.    
**SharePoint Debouncer Design**
    Sharepoint subscription for change detection whenever someone edits Word document in a browser Its sends a notification immediately. Implemented debouncer, So if a modification Is active in last five minutes it will be indexing.
**Flie Splitliting  for Larger files**
    If Chunk size > 0, the PDF is split into chunks of that size and each chunk is processed and stored as a separate .md file in Blob Storage (e.g., filename_chunk_0-with-images_artifacts).
    If Chunk size = 0, the entire PDF is processed as a single unit without splitting.
    This allows flexible control over how granular the document processing should be.

**Process Documents with Docling:**
    Extracts text and metadata from documents using the Docling library.
    Converts documents into Markdown format with embedded or referenced images.
**Store Processed Output in Azure Blob Storage:**
    The processed Markdown files are uploaded to Azure Blob Storage for long-term storage.
**Generate Embeddings with Azure OpenAI:**
    Text chunks are split and passed to Azure OpenAI to generate embeddings.
**Store Data in Azure Cosmos DB:**
    The embeddings, along with metadata, are stored in Azure Cosmos DB for querying and retrieval.
## Environment setup
1. .azuredevops folder contains the yaml files for linting, renovate, semver versioning.
2. ci forder contains the build pipeline for different environments which calls the template pipeline for build, push and deploy the image to aks cluster.
3. Dockerfile contains the dockerfile commands to create the image
4. requirement.txt contains the dependent libraries for python app.
5. src folder contains the app.py python app code running inside the docker container.


# Introduction 
To create a new container in cosmos-digihub-chatbot database for storing the index processing logs in it. 

# Getting Started
When user runs the application, action_log.py file also runs a script file. 
And in this file, if a container is not available in cosmos db then a new container is created in cosmos-digihub-chatbot database and after that index processed log actions stored here. 
If the container is already exists in cosmos db, then container creation process is skipped and index processed log actions storage happens.


# Deployment Step
 
1. Provision Cosmos Containers
2. Bulk indexing - it will inedex all the sharepoint documents, and update dh-chatbot-sharepoint-sync with the file names and status of the file indexed

 
# Running Bulk indexing job
 
# Running job to trigger a failed file indexing