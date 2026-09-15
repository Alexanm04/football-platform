import uuid

from sqlalchemy import Column, Float, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID

from src.core.database import Base


class EventModel(Base):
    __tablename__ = "events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    match_id = Column(UUID(as_uuid=True), ForeignKey("matches.id"), nullable=False, index=True)
    player_id = Column(UUID(as_uuid=True), ForeignKey("players.id"), nullable=True)

    minute = Column(Integer, nullable=False)
    second = Column(Integer, nullable=False)

    type = Column(String(50), nullable=False, index=True)

    location_x = Column(Float, nullable=True)
    location_y = Column(Float, nullable=True)

    details = Column(JSONB, nullable=False, default=dict)

    __table_args__ = (
        Index('ix_events_details_gin', details, postgresql_using='gin'),
    )