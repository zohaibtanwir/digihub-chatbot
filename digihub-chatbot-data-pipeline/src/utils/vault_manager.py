import os
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
load_dotenv()


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
        if not key_vault_url:
            raise ValueError("KEY_VAULT_URL environment variable not set.")
 
        # 3. Create a SecretClient.  This is used to interact with Key Vault secrets.
        client = SecretClient(vault_url=key_vault_url, credential=credential)
 
        # 4. Retrieve the secret.
        retrieved_secret = client.get_secret(secret_name)
        return retrieved_secret.value
 
    except Exception as e:
        print(f"Error retrieving secret {secret_name}: {e}")
        return None
 
def main():
    """
    Main function to demonstrate retrieving a secret from Azure Key Vault.
    """
    #  Set the environment variable KEY_VAULT_URL.  Replace with your Key Vault URL.
    #  This is just for demonstration.  In a real application, this should be
    #  configured outside of the code (e.g., in your system's environment variables).
    #  For example:
    #  os.environ["KEY_VAULT_URL"] = "https://your-key-vault-name.vault.azure.net"
    #
    #  DO NOT HARDCODE YOUR KEY VAULT URL IN PRODUCTION CODE.
    if "KEY_VAULT_URL" not in os.environ:
        print("Warning: KEY_VAULT_URL environment variable is not set.  You will need to set this.")
        print("For example:  export KEY_VAULT_URL='https://your-key-vault-name.vault.azure.net'")
        print("The script will attempt to run, but may fail if the variable is not set.")
        #  In a real program, you might want to exit here.
        #  sys.exit(1)

    # secret_value = get_secret(secret_name)
 
    # if secret_value:
    #     print(f"Successfully retrieved secret {secret_name}: {secret_value}")
    # else:
    #     print(f"Failed to retrieve secret {secret_name}.")
    #     print("Please ensure that:")
    #     print("  1. You have set the KEY_VAULT_URL environment variable correctly.")
    #     print("  2. You have authenticated to Azure (e.g., via `az login`).")
    #     print("  3. The secret exists in your Key Vault, and you have the correct name.")
    #     print("  4. Your application has the necessary permissions to access the Key Vault.")
 
if __name__ == "__main__":
    main()