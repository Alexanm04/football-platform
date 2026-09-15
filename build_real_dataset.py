import os
import warnings

import pandas as pd
from statsbombpy import sb
from tqdm import tqdm

warnings.filterwarnings('ignore')

def get_xg(events_df):
        shots = events_df[events_df['type'] == 'Shot']
        if 'shot_statsbomb_xg' in shots.columns:
            return shots['shot_statsbomb_xg'].sum()
        return 0.0

def calculate_team_match_features(match_events_df: pd.DataFrame, team_name: str) -> dict:
    team_events = match_events_df[match_events_df['team'] == team_name]
    opp_events = match_events_df[match_events_df['team'] != team_name]

    if team_events.empty:
        return None

    team_passes = team_events[(team_events['type'] == 'Pass') & (team_events['pass_outcome'].isna())].shape[0]
    total_passes = match_events_df[(match_events_df['type'] == 'Pass') & (match_events_df['pass_outcome'].isna())].shape[0]
    possession = (team_passes / total_passes * 100) if total_passes > 0 else 50.0

    defensive_types = ['Duel', 'Interception', 'Clearance', 'Block', 'Pressure']
    def_events = team_events[team_events['type'].isin(defensive_types)]

    def_height = 45.0
    if not def_events.empty and 'location' in def_events.columns:
        valid_locations = def_events['location'].dropna()
        if not valid_locations.empty:
            xs = [loc[0] for loc in valid_locations if isinstance(loc, list)]
            if xs:
                def_height = (sum(xs) / len(xs)) * (105.0 / 120.0)

    team_passes_events = team_events[team_events['type'] == 'Pass']
    offensive_width = 50.0
    if not team_passes_events.empty and 'location' in team_passes_events.columns:
        valid_y_locs = team_passes_events['location'].dropna().apply(
            lambda loc: loc[1] if isinstance(loc, list) and len(loc) > 1 else None
        ).dropna()

        if len(valid_y_locs) > 10: 
            width_span = valid_y_locs.quantile(0.90) - valid_y_locs.quantile(0.10)
            offensive_width = width_span * (68.0/80.0)

    opp_passes = opp_events[(opp_events['type'] == 'Pass')].shape[0]
    def_actions_count = def_events.shape[0]
    ppda = (opp_passes / def_actions_count) if def_actions_count > 0 else 20.0

    xg_for = get_xg(team_events)
    xg_against = get_xg(opp_events)

    return {
        'team': team_name,
        'possession': round(possession, 2),
        'defensive_height': round(def_height, 2),
        'ppda': round(ppda,2),
        'width': round(offensive_width, 2),
        'xg_for': round(xg_for, 2),
        'xg_against': round(xg_against, 2)
    }

def main():
    output_file = 'statsbomb_tactical_dataset.csv'
    if os.path.exists(output_file):
        os.remove(output_file)

    print("📡 Obteniendo lista de todas las competiciones disponibles en StatsBomb...")
    competitions = sb.competitions()
    
    total_records = 0
    
    for index, row in competitions.iterrows():
        comp_id = row['competition_id']
        season_id = row['season_id']
        comp_name = row['competition_name']
        season_name = row['season_name']
        
        print(f"\n🏆 Procesando: {comp_name} - Temporada {season_name} (CompID: {comp_id}, SeasonID: {season_id})")
        
        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
            if matches.empty:
                print("   ⚠️ No hay partidos disponibles para esta temporada.")
                continue
                
            match_ids = matches['match_id'].tolist()
            print(f"   Encontrados {len(match_ids)} partidos.")
            
            dataset = []
            
            for match_id in tqdm(match_ids, desc="   Descargando Eventos", unit="partido"):
                try:
                    events = sb.events(match_id=match_id)
                    
                    if events.empty or 'team' not in events.columns:
                        continue
                        
                    teams = events['team'].unique()

                    for team in teams:
                        features = calculate_team_match_features(events, team)
                        if features:
                            features['match_id'] = match_id
                            features['competition'] = comp_name 
                            features['season'] = season_name    
                            dataset.append(features)
                except Exception:  # noqa: BLE001, S110
                    pass 

            if dataset:
                df = pd.DataFrame(dataset)
                write_header = not os.path.exists(output_file)
                df.to_csv(output_file, mode='a', header=write_header, index=False)
                total_records += len(df)
                print(f"   ✅ Lote guardado. Total acumulado: {total_records} registros.")

        except Exception as e:  # noqa: BLE001
            print(f"   ❌ Error al obtener partidos de la competición: {e}")

    print(f"\n🎉 ¡Extracción masiva completada! Total final: {total_records} registros en '{output_file}'.")

if __name__ == "__main__":
    main()