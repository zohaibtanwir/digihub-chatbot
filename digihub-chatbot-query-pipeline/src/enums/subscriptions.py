# List of all available subscriptions - along with id's and names
from enum import Enum

# Enum to get all the details related to subscription
class Subscription(Enum):
    generic = {"id": 0, "name": "Generic"}
    airport_solutions = {"id": 100, "name": "Airport Solutions"}
    billing = {"id": 120, "name": "Billing"}
    case_management = {"id": 130, "name": "Case Management"}
    service_management = {"id": 140, "name": "Service Management"}
    user_management = {"id": 160, "name": "User Management"}
    announcement_management = {"id": 200, "name": "Announcement Management"}
    reporting = {"id": 220, "name": "Reporting"}
    world_tracer = {"id": 240, "name": "World Tracer"}
    connect_sdn_and_connect_corporate = {"id": 260, "name": "Connect SDN & Connect Corporate"}
    connect_go = {"id": 280, "name": "Connect Go"}
    airport_committee = {"id": 300, "name": "Airport Committee"}
    airport_management_solutions = {"id": 320, "name": "Airport Management Solutions"}
    community_messaging_kb = {"id": 340, "name": "Community Messaging KB"}
    bag_manager = {"id": 360, "name": "Bag Manager"}
    document_management = {"id": 380, "name": "Document Management"}
    billing_document_management = {"id": 400, "name": "Billing Document Management"}
    airport_solutions_document_management = {"id": 420, "name": "Airport Solution Document Management"}
    service_management_document_management = {"id": 440, "name": "Service Management Document Management"}
    euro_customer_advisory_board = {"id": 460, "name": "EURO Customer Advisory Board"}
    shared_resource = {"id": 480, "name": "Shared Resource"}
    aero_performance = {"id": 500, "name": "Aero Performance"}
    general_info = {"id": 0, "name": "General Info"}

def get_service_names(ids):
    names = []
    for item in Subscription:
        if item.value["id"] in ids:
            names.append(item.value["name"])
    return names
