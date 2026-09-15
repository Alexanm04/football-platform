import json
import logging
import pandas as pd
import requests
from typing import Any
from uuid import UUID

from statsbombpy import sb

from src.core.database import AsyncSession, AsyncSessionLocal
from src.modules.analytics.infrastructure.models import EventModel
from src.modules.analytics.infrastructure.repositories.event_repository import (
    EventRepository,
)

logger = logging.getLogger(__name__)


class StatsBombProviderError(Exception):
    """Indica que StatsBomb no está disponible."""


class StatsBombTimeoutError(Exception):
    """Indica que StatsBomb agotó el tiempo de espera."""


class AnalyticsService:
    async def process_and_save_events(self, match_id: UUID, raw_data: bytes, provider: str):
        try:
            data = json.loads(raw_data.decode("utf-8"))
            events_to_insert = []

            if provider == "statsbomb":
                events_to_insert = self._parse_statsbomb_events(match_id, data)
            else:
                logger.error("Proveedor no compatible: %s", provider)
                return

            async with AsyncSessionLocal() as session:
                try:
                    repo = EventRepository(session)
                    await repo.bulk_insert(events_to_insert)
                    logger.info(
                        "Se insertaron correctamente %s eventos para el partido %s",
                        len(events_to_insert),
                        match_id,
                    )
                except Exception:  # noqa: BLE001
                    await session.rollback()
                    logger.exception("Error de base de datos durante la inserción masiva")
        except Exception:  # noqa: BLE001
            logger.exception("Error grave procesando el archivo de eventos")

    def _parse_statsbomb_events(self, match_id: UUID, data: list[dict[str, Any]]) -> list[EventModel]:
        parsed_events = []
        for item in data:
            location = item.get("location", [None, None])
            event = EventModel(
                match_id=match_id,
                player_id=None,
                minute= item.get("minute", 0),
                second= item.get("second", 0),
                type= item.get("type",{}).get("name", "unknown"),
                location_x = location[0] if len(location) > 0 else None,
                location_y = location[1] if len(location) > 1 else None,
                details=item
            )
            parsed_events.append(event)

        return parsed_events

    async def get_match_events(self, match_id: UUID, event_type: str, session: AsyncSession) -> list[EventModel]:
        repo = EventRepository(session)
        events = await repo.get_by_match(match_id, event_type)
        return events

    async def generate_pass_network(self, match_id: int, team_name: str) -> dict:
        try: 
            events = sb.events(match_id=match_id)
        except requests.Timeout as exc:
            logger.exception("Tiempo de espera agotado al obtener eventos de StatsBomb")
            raise StatsBombTimeoutError from exc
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                raise ValueError("No se encontraron datos de StatsBomb.") from exc
            logger.exception("Proveedor StatsBomb no disponible al obtener eventos")
            raise StatsBombProviderError from exc
        except requests.RequestException as exc:
            logger.exception("Proveedor StatsBomb no disponible al obtener eventos")
            raise StatsBombProviderError from exc
        except Exception:
            logger.exception("Error interno obteniendo eventos de StatsBomb")
            raise

        substitutions = events[events['type'] == 'Substitution']
        first_sub_minute = substitutions['minute'].min() if not substitutions.empty else events['minute'].max()

        events_filtered = events[events['minute'] < first_sub_minute].copy()
        
        passes = events_filtered[(events_filtered['type'] == 'Pass') & (events_filtered['team'] == team_name) & (events_filtered['pass_outcome'].isna())].copy()

        if passes.empty:
            raise ValueError("No se encontraron pases completados para este equipo.")

        if 'player' not in passes.columns or 'pass_recipient' not in passes.columns or 'location' not in passes.columns:
            raise ValueError("Faltan columnas requeridas en los datos del proveedor.")

        passes['x'] = passes['location'].apply(lambda loc: loc[0] if isinstance(loc, list) else None)
        passes['y'] = passes['location'].apply(lambda loc: loc[1] if isinstance(loc,list) else None)

        player_positions = passes.groupby('player').agg(
            x=('x', 'mean'),
            y=('y', 'mean'),
            passes_made=('player', 'count')
        ).reset_index()

        pass_links = passes.groupby(['player', 'pass_recipient']).size().reset_index(name='pass_count')
        pass_links = pass_links[pass_links['pass_count'] >= 3]

        nodes = []
        for _,row in player_positions.iterrows():
            full_name = str(row['player'])

            nickname = None
            player_events = events[events['player'] == full_name]
            if 'player_nickname' in player_events.columns and not player_events.empty:
                valid_nicknames = player_events['player_nickname'].dropna()
                if not valid_nicknames.empty:
                    nickname = valid_nicknames.iloc[0]

            if nickname and str(nickname).lower() != "nan":
                short_name = nickname
            else:
                name_parts = full_name.split()

                if len(name_parts) == 1:
                    short_name = name_parts[0]
                elif len(name_parts) == 2:    
                    short_name = f"{name_parts[0][0]}. {name_parts[1]}" 
                else:
                    last_name = name_parts[-1]
                    if len(name_parts) >= 3 and name_parts[-2].lower() in ['de', 'di', 'van', 'mac', 'le', 'la', 'del']:
                         last_name = f"{name_parts[-2]} {name_parts[-1]}"
                    elif len(name_parts) >= 4 and name_parts[-2].lower() not in ['de', 'di', 'van', 'mac', 'le', 'la', 'del']:
                         last_name = name_parts[-2]
                         
                    short_name = f"{name_parts[0][0]}. {last_name}"
            nodes.append({
                "id": row['player'],
                "label": short_name,
                "x": round(row['x'], 2),
                "y": round(row['y'], 2),
                "value": int(row['passes_made'])
            })

        links = []
        for _, row in pass_links.iterrows():
            links.append({
                "source": row['player'],
                "target": row['pass_recipient'],
                "value": int(row['pass_count'])
            })

        return {
            "match_id": match_id,
            "team": team_name,
            "nodes": nodes,
            "links": links
        }

    async def get_competitions(self) -> list[dict]:
        try:
            comps = sb.competitions()
            result = []
            for _, row in comps.iterrows():
                result.append({
                    "competition_id": int(row['competition_id']),
                    "season_id": int(row['season_id']),
                    "competition_name": row['competition_name'],
                    "season_name": row['season_name'],
                    "display_name": f"{row['competition_name']} - {row['season_name']}"
                })

            result.sort(key=lambda x: x["display_name"])
            return result
        except requests.Timeout as exc:
            logger.exception("Tiempo de espera agotado al obtener competiciones de StatsBomb")
            raise StatsBombTimeoutError from exc
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                raise ValueError("No se encontraron las competiciones.") from exc
            logger.exception("Proveedor StatsBomb no disponible al obtener competiciones")
            raise StatsBombProviderError from exc
        except requests.RequestException as exc:
            logger.exception("Proveedor StatsBomb no disponible al obtener competiciones")
            raise StatsBombProviderError from exc
        except Exception:
            logger.exception("Error interno obteniendo competiciones de StatsBomb")
            raise

    async def get_matches_by_competition(self, competition_id: int, season_id: int) -> list[dict]:
        try:
            matches = sb.matches(competition_id=competition_id, season_id=season_id)
            result = []
            for _, row in matches.iterrows():
                result.append({
                    "match_id": int(row['match_id']),
                    "home_team": row['home_team'],
                    "away_team": row['away_team'],
                    "match_date": row['match_date'],
                    "display_name": f"{row['home_team']} vs {row['away_team']} ({row['match_date']})"
                })

            result.sort(key=lambda x: x["match_date"], reverse=True)
            return result
        except requests.Timeout as exc:
            logger.exception("Tiempo de espera agotado al obtener partidos de StatsBomb")
            raise StatsBombTimeoutError from exc
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                raise ValueError("No se encontraron los partidos.") from exc
            logger.exception("Proveedor StatsBomb no disponible al obtener partidos")
            raise StatsBombProviderError from exc
        except requests.RequestException as exc:
            logger.exception("Proveedor StatsBomb no disponible al obtener partidos")
            raise StatsBombProviderError from exc
        except Exception:
            logger.exception("Error interno obteniendo partidos de StatsBomb")
            raise

    async def get_team_heatmap_data(self, match_id: int, team_name: str) -> dict:
        try:
            events = sb.events(match_id=match_id)
            lineups = sb.lineups(match_id=match_id)
        except requests.Timeout as exc:
            logger.exception("Tiempo de espera agotado al obtener datos del heatmap")
            raise StatsBombTimeoutError from exc
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                raise ValueError("No se encontraron los datos del heatmap.") from exc
            logger.exception("Proveedor StatsBomb no disponible al obtener datos del heatmap")
            raise StatsBombProviderError from exc
        except requests.RequestException as exc:
            logger.exception("Proveedor StatsBomb no disponible al obtener datos del heatmap")
            raise StatsBombProviderError from exc
        except Exception:
            logger.exception("Error interno obteniendo datos del heatmap")
            raise

        if team_name not in lineups:
            raise ValueError(f"El equipo {team_name} no se encuentra en las alineaciones del partido.")

        team_lineup = lineups[team_name]
        team_events = events[events['team'] == team_name].copy()

        player_data = {}
        for _, player_info in team_lineup.iterrows():
            player_id = player_info['player_id']
            positions = player_info.get('positions', [])

            is_starter = False
            minutes_played = 0
            main_position = "Unknown"

            if isinstance(positions, list) and len(positions) > 0:
                first_pos = positions[0]
                if first_pos.get('start_reason') == 'Starting XI':
                    is_starter = True
                main_position = first_pos.get('position', 'Unknown')

                for pos in positions:
                    start_min = pos.get('from', "00:00").split(":")[0]
                    end_min = pos.get('to', "90:00")
                    if end_min is None:
                        end_min = "90"
                    elif isinstance(end_min, str):
                        end_min = end_min.split(":")[0]

                    minutes_played += int(end_min) - int(start_min)

            nickname = player_info.get('player_nickname')

            if not pd.isna(nickname) and nickname:
                short_name = nickname
            else:
                full_name = str(player_info.get('player_name', 'Unknown'))
                name_parts = full_name.split()

                if len(name_parts) == 1:
                    short_name = name_parts[0]
                elif len(name_parts) == 2:    
                    short_name = f"{name_parts[0][0]}. {name_parts[1]}" 
                else:
                    short_name = f"{name_parts[0][0]}. {name_parts[-2]}"

            player_data[player_id] = {
                "id": player_id,
                "name": player_info['player_name'],
                "short_name": short_name,
                "jersey_number": player_info['jersey_number'],
                "is_starter": is_starter,
                "minutes_played": minutes_played,
                "position": main_position,
                "events_xy": [],
                "stats": {
                    "goals": 0,
                    "assists": 0,
                    "passes_attempted": 0,
                    "passes_completed": 0,
                    "dribbles_attempted": 0,
                    "dribbles_completed": 0,
                    "shots": 0
                },
                "avg_x": 0,
                "avg_y": 0,
                "event_count": 0
            }

        for _, event in team_events.iterrows():
            p_id = event.get('player_id')
            if pd.isna(p_id) or p_id not in player_data:
                continue

            p_data = player_data[p_id]
            e_type = event['type']

            loc = event.get('location')
            if isinstance(loc, list) and len(loc) == 2:
                p_data["events_xy"].append([loc[0], loc[1]])
                p_data["avg_x"] += loc[0]
                p_data["avg_y"] += loc[1]
                p_data["event_count"] += 1

            if e_type == 'Pass':
                p_data["stats"]["passes_attempted"] += 1
                if pd.isna(event.get('pass_outcome')):
                    p_data["stats"]["passes_completed"] += 1
                    if event.get('pass_goal_assist') == True:
                        p_data["stats"]["assists"] += 1

            elif e_type == 'Shot':
                p_data["stats"]["shots"] += 1
                if event.get('shot_outcome') == 'Goal':
                    p_data["stats"]["goals"] += 1

            elif e_type == 'Dribble':
                p_data["stats"]["dribbles_attempted"] += 1
                if event.get('dribble_outcome') == 'Complete':
                    p_data["stats"]["dribbles_completed"] += 1

        starters = []
        bench = []

        for p_id, p_data in player_data.items():
            if p_data["event_count"] > 0:
                p_data["avg_x"] = round(p_data["avg_x"] / p_data["event_count"], 2)
                p_data["avg_y"] = round(p_data["avg_y"] / p_data["event_count"], 2)
            else:
                p_data["avg_x"] = 10
                p_data["avg_y"] = 40

            del p_data["event_count"]

            if p_data["is_starter"]:
                starters.append(p_data)
            else:
                bench.append(p_data)

        bench.sort(key=lambda x: x["minutes_played"], reverse=True)

        return {
            "match_id": match_id,
            "team": team_name,
            "starters": starters,
            "bench": bench
        }
