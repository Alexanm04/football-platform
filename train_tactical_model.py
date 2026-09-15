import pickle
import sys

import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split


def train_and_evaluate(name, X_train, y_train, X_val, y_val, X_test, y_test, params):
    print(f"\n Entrenando modelo para: {name} ...")
    early_stop = xgb.callback.EarlyStopping(
        rounds=20,
        metric_name='rmse',
        save_best=True
    )

    model = xgb.XGBRegressor(**params, callbacks=[early_stop])

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )

    print(f"Entrenamiento detenido en iteración: {model.best_iteration}")

    preds_train = model.predict(X_train)
    rmse_train = root_mean_squared_error(y_train, preds_train)
    r2_train = r2_score(y_train, preds_train)

    preds_test = model.predict(X_test)
    rmse_test = root_mean_squared_error(y_test, preds_test)
    r2_test = r2_score(y_test, preds_test)

    print("-" * 40)
    print(f"📊 MÉTRICAS {name.upper()}")
    print(f"RMSE Train: {rmse_train:.3f} xG  |  RMSE Test: {rmse_test:.3f} xG")
    print(f"R² Train:   {r2_train:.3f}     |  R² Test:   {r2_test:.3f}")
    
    diff_rmse = rmse_test - rmse_train
    if diff_rmse > 0.2:
        print("⚠️ ADVERTENCIA: Posible Overfitting. El modelo es mucho mejor en Train que en Test.")
    elif diff_rmse < -0.1:
         print("⚠️ ADVERTENCIA: Comportamiento inusual (Underfitting o Data Leakage). Test es mejor que Train.")
    else:
        print("✅ DIAGNÓSTICO: Modelo saludable. Aprende sin memorizar en exceso.")
    print("-" * 40)

    return model

if __name__ == "__main__":
    try:
        df = pd.read_csv('statsbomb_tactical_dataset.csv')
        print(f"Dataset cargado: {len(df)} registros encontrados.")
    except FileNotFoundError:
        print("Error: No se encontró 'statsbomb_tactical_dataset.csv'.")
        sys.exit()

    df = df.dropna()
    print(f"Limpiando nulos... Registros válidos: {len(df)}")

    features = ['defensive_height', 'ppda', 'possession', 'width']
    X = df[features]

    y_for = df['xg_for']
    y_against = df['xg_against']

    X_temp, X_test, y_for_temp, y_for_test = train_test_split(X, y_for, test_size=0.2, random_state=42)
    _, _, y_against_temp, y_against_test = train_test_split(X, y_against, test_size=0.2, random_state=42)

    X_train, X_val, y_for_train, y_for_val = train_test_split(X_temp, y_for_temp, test_size=0.25, random_state=42)
    _, _, y_against_train, y_against_val = train_test_split(X_temp, y_against_temp, test_size=0.25, random_state=42)

    print(f"División de datos -> Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

    params = {
        'objective': 'reg:squarederror',
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.9,
        'random_state': 42
    }

    model_for = train_and_evaluate(
        "xG a Favor (Ataque)", 
        X_train, y_for_train, 
        X_val, y_for_val, 
        X_test, y_for_test, 
        params
    )
    
    model_against = train_and_evaluate(
        "xG en Contra (Defensa)", 
        X_train, y_against_train, 
        X_val, y_against_val, 
        X_test, y_against_test, 
        params
    )

    models = {
        'xg_for_model': model_for,
        'xg_against_model': model_against,
        'features': features
    }

    output_filename = 'tactical_models.pkl'
    with open(output_filename, 'wb') as f:
        pickle.dump(models, f)

    print(f"\n !Modelos listos para producción guardados en '{output_filename}'!")