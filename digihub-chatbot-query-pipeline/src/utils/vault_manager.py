import os
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from src.utils.logger import logger


def get_secret(secret_name):
    from src.utils.config import KEY_VAULT_URL
    """
    Retrieves a secret from Azure Key Vault.  Handles authentication and client creation.
    Args:
        secret_name (str): The name of the secret to retrieve.
 
    Returns:
        str: The value of the secret, or None if an error occurs.
    """
    try:
        # 1. Obtain credentials.  DefaultAzureCredential is recommended for production.
        #    It tries various methods (environment variables, managed identity, etc.)
        credential = DefaultAzureCredential()
 
        # 2.  Get the Key Vault URL.  This should be set as an environment variable.
        key_vault_url = KEY_VAULT_URL
        # "https://kv-dev-westeurope-02.vault.azure.net/"
        if not key_vault_url:
            raise ValueError("KEY_VAULT_URL environment variable not set.")
 
        # 3. Create a SecretClient.  This is used to interact with Key Vault secrets.
        client = SecretClient(vault_url=key_vault_url, credential=credential)
 
        # 4. Retrieve the secret.
        retrieved_secret = client.get_secret(secret_name)
        return retrieved_secret.value
 
    except Exception as e:
        # logger.error("Error retrieving secret of key vault.")
        logger.info(f"Error retrieving secret {secret_name}: {e}")
        return None