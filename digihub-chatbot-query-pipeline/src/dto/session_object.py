from dataclasses import dataclass, field
from typing import List,Optional


@dataclass
class Session:
    id: str
    messageId: str
    sessionId: str
    userId: str
    impersonated_user_id: str
    sender: str
    timestamp: int
    text: str
    citation: List[dict]
    entities: List[str] = field(default_factory=list)
    chunk_service_line: List[int] = field(default_factory=list)
    score: Optional[float] = 0.0
    confidence: Optional[float] = 0.0
    feedback_score: Optional[int] = None
    disclaimer: Optional[str] = None 