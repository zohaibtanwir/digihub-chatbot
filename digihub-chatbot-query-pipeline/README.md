# DigiHub Chatbot Query Pipeline

## Overview
This project is a chatbot query pipeline that integrates with Azure OpenAI and CosmosDB to provide intelligent responses based on user queries.

## Setup Instructions

### Prerequisites
1. **Azure Key Vault**: Ensure you have an Azure Key Vault set up with the following secrets:
   - `azure-openai-api-key`
   - `openai-api-version`
   - `azure-openai-endpoint`
   - `cosmosdb-endpoint`
   - `cosmosdb-key`

2. **Environment Variables**: Set the following environment variables:
   - `KEY_VAULT_URL`: The URL of your Azure Key Vault.

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Digihub_Chatbot/Chatbot_Query_pipeline


## Secrets Management

### VaultManager
The `vault_manager.py` file handles the initialization and caching of Azure Key Vault secrets. It uses a singleton pattern to ensure that secrets are retrieved only once and reused across the application.

### Usage
To access secrets, import the `vault_manager` and use the `get_secret` method:

```python
from vault_manager import vault_manager

api_key = vault_manager.get_secret("azure-openai-api-key")

## Environment setup
1. .azuredevops folder contains the yaml files for linting, renovate, semver versioning.
2. ci forder contains the build pipeline for different environments which calls the template pipeline for build, push and deploy the image to aks cluster.
3. Dockerfile contains the dockerfile commands to create the image
4. requirement.txt contains the dependent libraries for python app.
5. src folder contains the app.py python app code running inside the docker container.

#Linting wiki link:
https://dev.azure.com/SITA-PSE/Communication%20and%20Network/_wiki/wikis/Communication-and-Network.wiki/40180/Conventional-commits-and-linting


## Feature: SAS URL Generation for Azure Storage Account Container

This feature generates Shared Access Signature (SAS) URLs for Azure Storage Account Containers, enabling secure, time-limited access to resources. SAS URLs allow external users or services to access specific container without exposing the storage account credentials.

## Why SAS URLs?

**Secure Access:** Provides controlled, secure access to blobs or containers.

**Time-bound Access:** SAS URLs expire after a set period, ensuring limited access.

**Granular Permissions:** Supports permissions like read, write, and list for specific resources.

## Roles & Permissions

Azure **Service Principal Name(SPN)** account must have the following **role assignments** to generate the sas url for the container at the appropriate scope(Subscription/Storage Account/Container level):

- **Storage Blob Data Contributor:** Grants read/write access to blobs.

- **Storage Blob Delegator:** Allows generation of SAS URLs.

This implementation ensures secure, time-limited access to Azure Blob Storage resources without exposing sensitive credentials.

## Configuration

The following Azure Storage values are stored in a `digihub-chatbot-query-pipeline-dev.properties` file inside the shared **digihub-config** repo:

- `AZURE_STORAGE_ACCOUNT_NAME` – The name of the Azure Storage Account.
- `AZURE_STORAGE_ACCOUNT_URL` – The full URL of the Azure Storage Account (e.g.,`http://blob.core.windows.net/)`).
- `AZURE_STORAGE_CONTAINER_NAME` – The name of the container where files are stored.
- `AZURE_STORAGE_CONTAINER_SAS_URL_VALIDITY` - Access validity of generated URL.

#Please follow the link for linting rules:
https://sita-pse.visualstudio.com/Communication%20and%20Network/_wiki/wikis/Communication-and-Network.wiki/40180/Conventional-commits-and-linting

## Confidence Score Based Authorization

This module implements a security layer that ensures users can only access information related to service lines they are authorized for. The authorization check is triggered based on the confidence score of the AI-generated response.

### 🔁 Flow Overview

1. **Initial Confidence Check**
   - When a query is processed, a `confidence_score` is calculated.
   - If `confidence_score < 0.80`, the system performs an additional authorization check to ensure the user has access to the relevant service lines.

2. **Cross-Check Authorization**
   - The method `cross_check_authorization(prompt, service_line)` is invoked.
   - It retrieves the top-ranked service line IDs relevant to the query using:
     ```python
     chunk_service_line = RetreivalService().get_ranked_service_line_chunk(prompt)
     ```
   - It then compares these IDs with the user's authorized service lines.

3. **Authorization Decision**
   - If the user’s service line list does **not** include the required service line(s), an exception is raised:
     ```python
     raise UnAuthorizedServiceLineException(
         "**Query Denied: Unauthorized Service Line**<br><br>"
         "You do not have access to the required service line(s) to answer this query.<br><br>"
         "**Access Required:**<br>"
         f"- {'<br> - '.join(get_service_names(chunk_service_line))}"
     )
     ```
4. **Session dependent Query** 
   -  Improved semantic search by preserving context in follow-up queries.
   -  If a query depends on a previous one, both are merged to form an expanded query for better relevance.
   -  Example: “What is WorldTracer product?” followed by “Why do we need that?” now returns more accurate results.

### ✅ Benefits

- Ensures sensitive or restricted information is only accessible to authorized users.
- Adds an extra layer of validation when AI confidence is low.
- Provides a user-friendly error message with actionable information.

### 📌 Notes

- Admin users may bypass this check depending on implementation.
- The `get_service_names()` function maps service line IDs to human-readable names for better clarity in error messages.

### Enhancement of Query retrieval <placeholder>

