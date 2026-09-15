from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from src.modules.analytics.infrastructure.models import EventModel


class EventRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def bulk_insert(self, events: list[EventModel]):
        self.session.add_all(events)
        await self.session.commit()

    async def get_by_match(self, match_id: UUID, event_type: str | None = None) -> list[EventModel]:
        query = select(EventModel).where(EventModel.match_id == match_id)
        if event_type:
            query = query.where(EventModel.type == event_type)

        query = query.order_by(EventModel.minute, EventModel.second)

        result = await self.session.execute(query)
        return result.scalars().all()