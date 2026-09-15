from uuid import UUID

from src.core.exceptions import ConflictException, NotFoundException
from src.modules.football.infrastructure.repositories.player_repository import (
    PlayerRepository,
)
from src.modules.football.presentation.schemas.player_schemas import PlayerCreate


class PlayerService:
    def __init__(self, repository: PlayerRepository):
        self.repository = repository

    async def register_player(self, player_data: PlayerCreate):
        existing_player = await self.repository.get_by_name_and_birth_date(
        player_data.name,
        player_data.birth_date,
        )

        if existing_player:
            raise ConflictException(detail="Player already exists")
        return await self.repository.create(player_data)

    async def fetch_player(self, player_id: UUID):
        player = await self.repository.get_by_id(player_id)
        if not player:
            raise NotFoundException(detail="Player not found")
        return player

