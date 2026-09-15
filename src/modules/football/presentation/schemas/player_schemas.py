from datetime import date
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class PlayerBase(BaseModel):
    name: str = Field(..., max_length=100)
    birth_date: date
    dominant_foot: str = Field(..., pattern="^(right|left|both)$")

class PlayerCreate(PlayerBase):
    pass

class PlayerResponse(PlayerBase):
    id: UUID
    model_config = ConfigDict(from_attributes=True)