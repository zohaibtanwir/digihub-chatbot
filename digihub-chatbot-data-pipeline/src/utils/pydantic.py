from pydantic import BaseModel, Field
from typing import List, Optional

class ResourceData(BaseModel):
    odata_type: Optional[str] = Field(
        default=None,
        alias="@odata.type",
        min_length=10,  # "#Microsoft.Graph.ListItem" is 29 chars
        max_length=40
    )

class WebhookEvent(BaseModel):
    subscriptionId: str = Field(..., min_length=2, max_length=150)   
    clientState: Optional[str] = Field(default=None, min_length=1, max_length=30)
    resource: str = Field(..., min_length=50, max_length=150)       
    tenantId: str = Field(..., min_length=36, max_length=40)      
    resourceData: Optional[ResourceData] = None
    subscriptionExpirationDateTime: str = Field(..., min_length=20, max_length=40)   
    changeType: str = Field(..., min_length=6, max_length=10)      

class WebhookPayload(BaseModel):
    value: List[WebhookEvent]
