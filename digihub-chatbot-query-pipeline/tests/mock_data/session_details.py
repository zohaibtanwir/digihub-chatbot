from enum import Enum


class SessionDetails(Enum):
    # Getting details for testing session service
    user_id = "f1ade0278800b4739fee5ea9ed964eb1755ce67df429e257170b64b46320ac19"
    session_details_text = [
        {
            "text": "What functionalities can users navigate to from the Billing Homepage?",
            "sender": "user",
        },
        {
            "text": "From the Billing Homepage, users can navigate to the following functionalities:  \n\n    1. Different functionalities within the Digihub Billing service.  \n\n    2. Document library.  \n\n    3. Billing & payment support cases, which are grouped into three categories:  \n        - Open Cases  \n        - Cases Pending Your Inputs  \n        - Recently Resolved & Closed cases  \n\nUsers can view the most recently updated cases in each category and click an arrow button to see all cases.  \n\nAdditionally, users can access options like:  \n    - Invoice Download  \n    - E-Invoice Tracking  \n    - Customer Movement Report  \n    - Manage Subscription  ",
            "sender": "assistant",
        },
    ]
    session_details = [
        {
            "text": "What functionalities can users navigate to from the Billing Homepage?",
            "sender": "user",
            "messageId": "da10bb07-4c88-41df-88f3-83724f48a18d",
            "timestamp": 1747300855,
            "entities": [],
            "citation": [{}],
        },
        {
            "text": "From the Billing Homepage, users can navigate to the following functionalities:  \n\n    1. Different functionalities within the Digihub Billing service.  \n\n    2. Document library.  \n\n    3. Billing & payment support cases, which are grouped into three categories:  \n        - Open Cases  \n        - Cases Pending Your Inputs  \n        - Recently Resolved & Closed cases  \n\nUsers can view the most recently updated cases in each category and click an arrow button to see all cases.  \n\nAdditionally, users can access options like:  \n    - Invoice Download  \n    - E-Invoice Tracking  \n    - Customer Movement Report  \n    - Manage Subscription  ",
            "sender": "assistant",
            "messageId": "8ddad71e-f156-4022-bafd-559bd8bc3c65",
            "timestamp": 1747300856,
            "entities": [],
            "citation": [
                {
                    "File": "Billing/Billing & Payment User Guide.pdf",
                    "Section": "View and Use Billing Homepage",
                }
            ],
        }
    ]
