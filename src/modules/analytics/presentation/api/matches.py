import json
import logging
import os
import asyncio
import math
from datetime import date
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from pydantic import BaseModel, ConfigDict

from src.core.database import AsyncSession, get_db_session
from src.core.rate_limit import limiter
from src.modules.analytics.application.analytics_service import (
    AnalyticsService,
    StatsBombProviderError,
    StatsBombTimeoutError,
)
from src.modules.analytics.presentation.schemas.event_schemas import EventResponse
from src.modules.football.infrastructure.models import MatchModel

router = APIRouter(prefix="/matches", tags=["Analytics"])
logger = logging.getLogger(__name__)

MAX_EVENT_UPLOAD_BYTES = int(os.getenv("MAX_EVENT_UPLOAD_BYTES", "5242880"))
MAX_EVENTS_PER_UPLOAD = int(os.getenv("MAX_EVENTS_PER_UPLOAD", "20000"))
MAX_CONCURRENT_EVENT_INGESTIONS = 1
EVENT_INGESTION_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_EVENT_INGESTIONS)

class MatchCreate(BaseModel):
    date: date

class MatchResponse(MatchCreate):
    id: UUID
    model_config = ConfigDict(from_attributes=True)

def get_analytics_service() -> AnalyticsService:
    return AnalyticsService()


def _validate_event_payload(payload: object) -> list[dict]:
    if not isinstance(payload, list) or not payload:
        raise HTTPException(status_code=422, detail="El JSON debe ser una lista no vacía de eventos.")

    for event in payload:
        if not isinstance(event, dict):
            raise HTTPException(status_code=422, detail="Cada evento debe ser un objeto JSON.")

        event_type = event.get("type")
        if event_type is not None and (
            not isinstance(event_type, dict)
            or ("name" in event_type and not isinstance(event_type["name"], str))
        ):
            raise HTTPException(status_code=422, detail="El campo type del evento no es válido.")

        minute = event.get("minute")
        if minute is not None and (
            isinstance(minute, bool) or not isinstance(minute, int) or minute < 0
        ):
            raise HTTPException(status_code=422, detail="El campo minute del evento no es válido.")

        second = event.get("second")
        if second is not None and (
            isinstance(second, bool)
            or not isinstance(second, int)
            or second < 0
            or second > 59
        ):
            raise HTTPException(status_code=422, detail="El campo second del evento no es válido.")

        location = event.get("location")
        if location is not None and (
            not isinstance(location, list)
            or len(location) < 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in location[:2]
            )
        ):
            raise HTTPException(status_code=422, detail="El campo location del evento no es válido.")

    return payload


async def _process_event_ingestion(
    service: AnalyticsService,
    match_id: UUID,
    raw_data: bytes,
) -> None:
    async with EVENT_INGESTION_SEMAPHORE:
        await service.process_and_save_events(
            match_id=match_id,
            raw_data=raw_data,
            provider="statsbomb",
        )


@router.post("/", response_model=MatchResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("20/minute")
async def create_match(
    request: Request,
    match_data: MatchCreate,
    session: AsyncSession = Depends(get_db_session),
):
    new_match = MatchModel(date=match_data.date)
    session.add(new_match)
    await session.commit()
    await session.refresh(new_match)
    return new_match


@router.post("/{match_id}/events/upload", status_code=status.HTTP_202_ACCEPTED)
@limiter.limit("5/minute")
async def upload_event_data(
    request: Request,
    match_id: UUID,
    file: UploadFile,
    background_tasks: BackgroundTasks,
    service: AnalyticsService = Depends(get_analytics_service),
):
    content_type = (file.content_type or "").split(";", 1)[0].strip().lower()
    if content_type != "application/json":
        raise HTTPException(status_code=415, detail="Solo se aceptan archivos JSON.")

    content = await file.read(MAX_EVENT_UPLOAD_BYTES + 1)
    if len(content) > MAX_EVENT_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="El archivo JSON supera el tamaño permitido.")

    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise HTTPException(status_code=400, detail="El JSON proporcionado es inválido o está corrupto.")

    payload = _validate_event_payload(payload)

    if len(payload) > MAX_EVENTS_PER_UPLOAD:
        raise HTTPException(status_code=422, detail="El número de eventos supera el límite permitido.")

    background_tasks.add_task(
        _process_event_ingestion,
        service=service,
        match_id=match_id,
        raw_data=content,
    )

    return {"message": "Data ingestion started in background", "match_id": str(match_id)}


@router.get("/{match_id}/events", response_model=list[EventResponse], status_code=status.HTTP_200_OK)
@limiter.limit("30/minute")
async def get_events(
    request: Request,
    match_id: UUID,
    event_type: str | None = None,
    service: AnalyticsService = Depends(get_analytics_service),
    session: AsyncSession = Depends(get_db_session),
):
    events = await service.get_match_events(match_id, event_type, session)
    return events


@router.get("/{match_id}/pass-network/{team_name}", status_code=status.HTTP_200_OK)
@limiter.limit("5/minute")
async def get_pass_network(
    request: Request,
    match_id: int,
    team_name: str,
    service: AnalyticsService = Depends(get_analytics_service),
):
    try:
        network_data = await service.generate_pass_network(match_id, team_name)
        return network_data
    except ValueError:
        logger.info("No se encontraron datos para la red de pases")
        raise HTTPException(status_code=404, detail="Pass network data not found.")
    except StatsBombTimeoutError:
        raise HTTPException(status_code=504, detail="El proveedor externo agotó el tiempo de espera.")
    except StatsBombProviderError:
        raise HTTPException(status_code=502, detail="El proveedor externo no está disponible.")
    except Exception:  # noqa: BLE001
        logger.exception("Error generando la red de pases")
        raise HTTPException(status_code=500, detail="Error interno generando la red.")


@router.get("/statsbomb/competitions", status_code=status.HTTP_200_OK)
@limiter.limit("20/minute")
async def get_sb_competitions(
    request: Request,
    service: AnalyticsService = Depends(get_analytics_service),
):
    try:
        return await service.get_competitions()
    except ValueError:
        logger.info("No se encontraron datos de competiciones")
        raise HTTPException(status_code=404, detail="No se encontraron competiciones.")
    except StatsBombTimeoutError:
        raise HTTPException(status_code=504, detail="El proveedor externo agotó el tiempo de espera.")
    except StatsBombProviderError:
        raise HTTPException(status_code=502, detail="El proveedor externo no está disponible.")
    except Exception:  # noqa: BLE001
        logger.exception("Error consultando competiciones")
        raise HTTPException(status_code=500, detail="Error interno consultando competiciones.")


@router.get("/statsbomb/matches/{competition_id}/{season_id}", status_code=status.HTTP_200_OK)
@limiter.limit("20/minute")
async def get_sb_matches(
    request: Request,
    competition_id: int,
    season_id: int,
    service: AnalyticsService = Depends(get_analytics_service),
):
    try:
        return await service.get_matches_by_competition(competition_id, season_id)
    except ValueError:
        logger.info("No se encontraron partidos para la competición solicitada")
        raise HTTPException(status_code=404, detail="No se encontraron partidos.")
    except StatsBombTimeoutError:
        raise HTTPException(status_code=504, detail="El proveedor externo agotó el tiempo de espera.")
    except StatsBombProviderError:
        raise HTTPException(status_code=502, detail="El proveedor externo no está disponible.")
    except Exception:  # noqa: BLE001
        logger.exception("Error consultando partidos")
        raise HTTPException(status_code=500, detail="Error interno consultando partidos.")


@router.get("/{match_id}/heatmaps/{team_name}", status_code=status.HTTP_200_OK)
@limiter.limit("5/minute")
async def get_team_heatmaps(
    request: Request,
    match_id: int,
    team_name: str,
    service: AnalyticsService = Depends(get_analytics_service),
):
    try:
        heatmap_data = await service.get_team_heatmap_data(match_id, team_name)
        return heatmap_data
    except ValueError:
        logger.info("No se encontraron datos de heatmap")
        raise HTTPException(status_code=404, detail="No se econtraron datos de heatmaps.")
    except StatsBombTimeoutError:
        raise HTTPException(status_code=504, detail="El proveedor externo agotó el tiempo de espera.")
    except StatsBombProviderError:
        raise HTTPException(status_code=502, detail="El proveedor externo no está disponible.")
    except Exception:  # noqa: BLE001
        logger.exception("Error generando heatmaps")
        raise HTTPException(status_code=500, detail="Error interno generando heatmaps.")