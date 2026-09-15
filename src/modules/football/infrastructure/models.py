import uuid

from sqlalchemy import Column, Date, String
from sqlalchemy.dialects.postgresql import UUID

from src.core.database import Base


class PlayerModel(Base):
    __tablename__ = "players"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    birth_date = Column(Date, nullable=False)
    dominant_foot = Column(String(10), nullable=False)

class MatchModel(Base):
    __tablename__ = "matches"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    date = Column(Date, nullable=True)