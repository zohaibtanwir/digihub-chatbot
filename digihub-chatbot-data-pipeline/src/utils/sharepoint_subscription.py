import json
import requests
from datetime import datetime, timedelta
from src.utils.config import site_id, EVENT_HUB_NOTIFICATION_URL
from src.utils.logger import logger
from src.utils.sharepoint_spn_login import TokenRetriever

class SharePointSubscriptionManager:
    def __init__(self):
        self.token_retriever = TokenRetriever()

    def get_headers(self):
        token = self.token_retriever.get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    def get_subscriptions(self):
        logger.info("Fetching existing subscriptions...")
        try:
            response = requests.get("https://graph.microsoft.com/v1.0/subscriptions", headers=self.get_headers())
            response.raise_for_status()
            subscriptions = response.json().get('value', [])
            logger.info(f"Retrieved {len(subscriptions)} subscriptions.")
            return subscriptions
        except Exception as e:
            logger.error(f"Failed to fetch subscriptions: {e}", exc_info=True)
            return []

    def delete_subscriptions(self, subscriptions):
        logger.info("Deleting existing subscriptions...")
        for subscription in subscriptions:
            subscription_id = subscription['id']
            url = f"https://graph.microsoft.com/v1.0/subscriptions/{subscription_id}"
            try:
                response = requests.delete(url, headers=self.get_headers())
                if response.status_code == 204:
                    logger.info(f"Subscription {subscription_id} deleted successfully.")
                else:
                    logger.warning(f"Failed to delete subscription {subscription_id}: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Error deleting subscription {subscription_id}: {e}", exc_info=True)

    def create_subscriptions(self):
        logger.info("Creating new subscriptions for all drives...")
        try:
            drive_url = f'https://graph.microsoft.com/v1.0/sites/{site_id}/drives'
            drive_response = requests.get(drive_url, headers=self.get_headers())
            drive_response.raise_for_status()
            drives = drive_response.json().get('value', [])
            logger.info(f"Retrieved {len(drives)} drives.")
        except Exception as e:
            logger.error(f"Failed to retrieve drives: {e}", exc_info=True)
            return []

        subscriptions = []
        for drive in drives:
            drive_id = drive['id']
            logger.info(f"Creating subscription for drive: {drive['name']} (ID: {drive_id})")
            expiration = (datetime.utcnow() + timedelta(minutes=4230)).isoformat() + 'Z'
            data = {
                "resource": f"/drives/{drive_id}/root",
                "expirationDateTime": expiration,
                "changeType": "updated",
                "notificationUrl": EVENT_HUB_NOTIFICATION_URL,
            }

            try:
                response = requests.post("https://graph.microsoft.com/v1.0/subscriptions", headers=self.get_headers(), data=json.dumps(data))
                if response.status_code == 201:
                    subscriptions.append(response.json()['id'])
                    logger.info(f"Subscription created for drive {drive_id}")
                else:
                    logger.warning(f"Failed to create subscription for drive {drive_id}: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Error creating subscription for drive {drive_id}: {e}", exc_info=True)

        return subscriptions

    def manage_subscriptions(self):
        logger.info("Managing subscriptions: deleting old and creating new ones.")
        existing = self.get_subscriptions()
        self.delete_subscriptions(existing)
        new_subs = self.create_subscriptions()
        logger.info(f"Subscription management complete. {len(new_subs)} new subscriptions created.")
