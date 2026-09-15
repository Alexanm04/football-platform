import asyncio
from uuid import UUID

import pandas as pd

from src.core.database import AsyncSessionLocal
from src.modules.ml_engine.domain.feature_builder import FeatureBuilder

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

async def run_pipeline(match_id_str: str):
    print(f"\n Iniciando Pipeline de Feature Engineering para el partido: {match_id_str}")

    try:
        match_id = UUID(match_id_str)
    except ValueError:
        print(" Error: El UUID proporcionado no tiene un formato válido.")
        return

    async with AsyncSessionLocal() as session:
        print("Conexión a la base de datos establecida.")

        builder = FeatureBuilder(session)

        try:
            print(" Extrayendo eventos y calculando KPIs vectorizados...")
            kpi_df = await builder.calculate_match_kpis(match_id)

            print("\n DATAFRAME RESULTANTE (Métricas por Equipo):")
            print("=" * 60)
            print(kpi_df)
            print("=" * 60)

            print("\n Tipos de datos del DataFrame:")
            print(kpi_df.dtypes)

        except ValueError as ve:
            print(f"\n Advertencia: {ve}")
        except Exception as e:
            print(f"\n Error crítico en el pipeline: {e}")

if __name__ == "__main__":
    TARGET_MATCH_ID = "f34cc698-a86c-4051-bf13-87e1b7071a0c"
    asyncio.run(run_pipeline(TARGET_MATCH_ID))
