from uuid import UUID

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from src.modules.analytics.infrastructure.models import EventModel


class FeatureBuilder:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def _fetch_raw_data(self, match_id: UUID) -> pd.DataFrame:
        query = select(EventModel).where(EventModel.match_id == match_id)
        result = await self.session.execute(query)
        events = result.scalars().all()

        if not events:
            return pd.DataFrame()

        data = []
        for e in events:
            team_info = e.details.get("team", {})

            data.append({
                "minute": e.minute,
                "second": e.second,
                "type": e.type,
                "team_name": team_info.get("name", "Unknown"),
                "x": e.location_x,
                "y": e.location_y,
                "pass_outcome": e.details.get("pass", {}).get("outcome", {}).get("name") if e.type == "Pass" else None
            })
        return pd.DataFrame(data)

    async def calculate_match_kpis(self, match_id: UUID) -> pd.DataFrame:
        df = await self._fetch_raw_data(match_id)

        if df.empty:
            raise ValueError("No events found for this match.")

        passes_df = df[df['type'] == 'Pass']
        pass_counts = passes_df.groupby('team_name').size()
        possession = (pass_counts / pass_counts.sum() * 100).round(2)

        defensive_actions = ['Duel', 'Interception', 'Clearance', 'Block']
        def_df = df[df['type'].isin(defensive_actions)]
        def_line_height = def_df.groupby('team_name')['x'].mean().round(2)

        kpi_df = pd.DataFrame({
            'possesion:pct': possession,
            'defensive_line_height_meters': def_line_height
        }).reset_index()

        return kpi_df