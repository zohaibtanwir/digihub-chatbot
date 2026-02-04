from pydantic import BaseModel, ConfigDict, Field, EmailStr
import uuid

class QueryRequest(BaseModel):
    query: str
    chat_session_id: str
    model_config = ConfigDict(extra="forbid")

class BaseHeader(BaseModel):
    x_digihub_emailid : EmailStr
    x_digihub_system_user : str = "true"
    x_digihub_tenantid : str
    x_digihub_traceid : str