import os
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from src.core.rate_limit import limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/simulator", tags=["Tactical Simulator ML"])

class TacticalRequest(BaseModel):
    defensiveHeight: float = Field(
        ge=0,
        le=105,
        allow_inf_nan=False,
    )
    pressureIntensity: float = Field(
        gt=0,
        allow_inf_nan=False,
    )
    possession: float = Field(
        ge=0,
        le=100,
        allow_inf_nan=False,
    )
    width: float = Field(
        ge=0,
        le=68,
        allow_inf_nan=False,
    )

class PredictionResponse(BaseModel):
    xg_for: float
    xg_against: float
    shap_for: dict
    shap_against: dict

PROJECT_ROOT = Path(__file__).resolve().parents[4]
model_path_value = os.getenv(
    "TACTICAL_MODELS_PATH",
    str(PROJECT_ROOT / "tactical_models.pkl"),
)
MODEL_PATH = str(Path(model_path_value).expanduser())
if not Path(MODEL_PATH).is_absolute():
    MODEL_PATH = str((PROJECT_ROOT / MODEL_PATH).resolve())
xg_for_model = None
xg_against_model = None
features_order = None
explainer_for = None
explainer_against = None

if os.path.exists(MODEL_PATH):
    try: 
        with open(MODEL_PATH, "rb") as f:
            models_data = pickle.load(f)
            xg_for_model = models_data["xg_for_model"]
            xg_against_model = models_data["xg_against_model"]
            features_order = models_data["features"]
            explainer_for = shap.TreeExplainer(xg_for_model)
            explainer_against = shap.TreeExplainer(xg_against_model)
    except Exception:  # noqa: BLE001
        logger.exception("Error cargando los modelos XGBoost")
else:
    logger.warning("Archivo de modelos XGBoost no encontrado")

@router.post("/predict", response_model=PredictionResponse)
@limiter.limit("5/minute")
async def predict_xg(request: Request, tactics: TacticalRequest):
    if xg_for_model is None or features_order is None:
        raise HTTPException(
            status_code=503,
            detail="Los modelos de Machine Learning no están cargados en el servidor."
        )

    input_data = pd.DataFrame([{
        'defensive_height': tactics.defensiveHeight,
        'ppda': tactics.pressureIntensity,
        'possession': tactics.possession,
        'width': tactics.width
    }])

    input_data = input_data[features_order]

    try:
        pred_for = xg_for_model.predict(input_data)[0]
        pred_against = xg_against_model.predict(input_data)[0]

        pred_for = max(0.00, float(pred_for))
        pred_against = max(0.00, float(pred_against))

        shap_values_for = explainer_for.shap_values(input_data)[0]
        shap_values_against = explainer_against.shap_values(input_data)[0]

        base_for = float(explainer_for.expected_value[0] if isinstance(explainer_for.expected_value, np.ndarray) else explainer_for.expected_value)
        base_against = float(explainer_against.expected_value[0] if isinstance(explainer_against.expected_value, np.ndarray) else explainer_against.expected_value)

        explanation_for = {
            "base_value": round(base_for, 2),
            "features": [{"name": f, "impact": round(float(v), 3)} for f,v in zip(features_order, shap_values_for)]
        }

        explanation_against = {
                    "base_value": round(base_against, 2),
                    "features": [{"name": f, "impact": round(float(v), 3)} for f,v in zip(features_order, shap_values_against)]
                }
    except Exception:  # noqa: BLE001
        logger.exception("Error durante la inferencia táctica")
        raise HTTPException(status_code=500, detail="Error de inferencia interna.")

    return PredictionResponse(
        xg_for=round(pred_for, 2),
        xg_against=round(pred_against, 2),
        shap_for=explanation_for,
        shap_against=explanation_against
    )
