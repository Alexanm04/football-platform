from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class EventBase(BaseModel):
    match_id: UUID
    player_id: UUID | None = None
    minute: int = Field(..., ge=0)
    second: int = Field(..., ge=0, le=59)
    type: str = Field(..., max_length=50)
    location_x: float | None = None
    location_y: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)

class EventResponse(EventBase):
    id: UUID
    model_config = ConfigDict(from_attributes=True)