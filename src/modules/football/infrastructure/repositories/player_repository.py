from datetime import date
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from src.modules.football.presentation.schemas.player_schemas import PlayerCreate

from ..models import PlayerModel


class PlayerRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, player_data: PlayerCreate) -> PlayerModel:
        new_player = PlayerModel(**player_data.model_dump())
        self.session.add(new_player)
        await self.session.commit()
        await self.session.refresh(new_player)
        return new_player

    async def get_by_id(self, player_id: UUID) -> PlayerModel | None:
        result = await self.session.execute(select(PlayerModel).where(PlayerModel.id == player_id))
        return result.scalars().first()

    async def get_by_name_and_birth_date ( self, name: str, birth_date: date) -> PlayerModel | None:
        result = await self.session.execute(select(PlayerModel).where(PlayerModel.name == name, PlayerModel.birth_date == birth_date))
        return result.scalars().first()