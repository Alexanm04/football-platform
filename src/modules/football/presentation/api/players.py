from uuid import UUID

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db_session
from src.core.rate_limit import limiter
from src.modules.football.application.player_service import PlayerService
from src.modules.football.infrastructure.repositories.player_repository import (
    PlayerRepository,
)
from src.modules.football.presentation.schemas.player_schemas import (
    PlayerCreate,
    PlayerResponse,
)

router = APIRouter(prefix="/players", tags=["Players"])

def get_player_service(session: AsyncSession = Depends(get_db_session)) -> PlayerService:
    repository = PlayerRepository(session)
    return PlayerService(repository)

@router.post("/", response_model=PlayerResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("20/minute")
async def create_player(
    request: Request,
    player: PlayerCreate,
    service: PlayerService = Depends(get_player_service),
):
    return await service.register_player(player)

@router.get("/{player_id}", response_model=PlayerResponse)
@limiter.limit("60/minute")
async def get_player(
    request: Request,
    player_id: UUID,
    service: PlayerService = Depends(get_player_service),
):
    return await service.fetch_player(player_id)
