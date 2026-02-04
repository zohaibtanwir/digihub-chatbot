from enum import Enum

class SubscriptionMockResponse(Enum):
    # Mocked response of get request for subscriptions of a user
    subscriptions = {
        "payload": [
            {
                "name": "AIRPORT_SOLUTIONS",
                "status": "PENDING",
                "id": 100,
                "isFavourite": False,
            },
            {
                "name": "BILLING",
                "status": "SUBSCRIBED",
                "id": 120,
                "isFavourite": False,
            },
            {
                "name": "SERVICE_MANAGEMENT",
                "status": "SUBSCRIBED",
                "id": 140,
                "isFavourite": True,
            },
            {
                "name": "WORLD_TRACER",
                "status": "SUBSCRIBED",
                "id": 240,
                "isFavourite": False,
            },
            {
                "name": "CONNECT_SDN_AND_CONNECT_CORPORATE",
                "status": "SUBSCRIBED",
                "id": 260,
                "isFavourite": False,
            },
            {
                "name": "CONNECT_GO",
                "status": "SUBSCRIBED",
                "id": 280,
                "isFavourite": False,
            },
            {
                "name": "AIRPORT_COMMITTEE",
                "status": "FORBIDDEN",
                "id": 300,
                "isFavourite": False,
            },
            {
                "name": "AIRPORT_MANAGEMENT_SOLUTIONS",
                "status": "FORBIDDEN",
                "id": 320,
                "isFavourite": False,
            },
            {
                "name": "COMMUNITY_MESSAGING_KB",
                "status": "FORBIDDEN",
                "id": 340,
                "isFavourite": False,
            },
            {
                "name": "BAG_MANAGER",
                "status": "SUBSCRIBED",
                "id": 360,
                "isFavourite": False,
            },
            {
                "name": "BILLING_DOCUMENT_MANAGEMENT",
                "status": "SUBSCRIBED",
                "id": 400,
                "isFavourite": False,
            },
            {
                "name": "AIRPORT_SOLUTIONS_DOCUMENT_MANAGEMENT",
                "status": "FORBIDDEN",
                "id": 420,
                "isFavourite": False,
            },
            {
                "name": "SERVICE_MANAGEMENT_DOCUMENT_MANAGEMENT",
                "status": "SUBSCRIBED",
                "id": 440,
                "isFavourite": False,
            },
            {
                "name": "EURO_CUSTOMER_ADVISORY_BOARD",
                "status": "PENDING",
                "id": 460,
                "isFavourite": False,
            },
            {
                "name": "AERO_PERFORMANCE",
                "status": "SUBSCRIBED",
                "id": 500,
                "isFavourite": False,
            },
        ]
    }