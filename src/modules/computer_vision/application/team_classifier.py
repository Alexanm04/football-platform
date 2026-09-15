from __future__ import annotations

import logging

import cv2
import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class TeamClassifier:
    """
    V22 - clasificador orientado a reconocer correctamente JUGADORES DE CAMPO.

    Principios:
      1. Los equipos se descubren principalmente por COLOR/COMPOSICION del uniforme.
      2. SigLIP NO participa en el clustering principal: una red visual generalista
         puede acercar camisetas distintas por pose/contexto y perjudicar la separacion.
      3. Los tracks raros/outliers no mueven los centroides definitivos.
      4. TODOS los tracks reciben despues una asignacion al equipo mas cercano.
         Nunca se usa la posicion en el campo para decidir TEAM_A/TEAM_B.
      5. SigLIP queda opcional y solo se puede consultar para desempatar tracks muy
         ambiguos, sin sustituir el descriptor cromatico.
    """

    def __init__(self, device: str | None = None, use_siglip: bool = False):
        self.device = device
        self.use_siglip = bool(use_siglip)
        self.last_model: dict | None = None
        self.siglip_model = None
        self.siglip_processor = None

        if self.use_siglip:
            try:
                import torch
                from PIL import Image
                from transformers import AutoProcessor, SiglipVisionModel

                model_id = "google/siglip-base-patch16-224"
                self.siglip_processor = AutoProcessor.from_pretrained(model_id)
                self.siglip_model = SiglipVisionModel.from_pretrained(model_id).to(
                    self.device or ("cuda" if torch.cuda.is_available() else "cpu")
                ).eval()
                self._torch = torch
                self._Image = Image
                logger.info("SigLIP V22 cargado como desempate opcional.")
            except Exception:
                logger.exception("SigLIP desactivado")
                self.siglip_model = None
                self.siglip_processor = None

    @staticmethod
    def _prepare_crop(crop_bgr: np.ndarray) -> np.ndarray | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if h < 18 or w < 8:
            return None

        y1, y2 = int(h * 0.12), int(h * 0.70)
        x1, x2 = int(w * 0.08), int(w * 0.92)
        torso = crop_bgr[y1:y2, x1:x2]
        if torso.size == 0 or torso.shape[0] < 8 or torso.shape[1] < 6:
            return None
        return torso

    @staticmethod
    def build_color_feature(crop_bgr: np.ndarray) -> tuple[np.ndarray | None, dict[str, float]]:
        torso = TeamClassifier._prepare_crop(crop_bgr)
        if torso is None:
            return None, {}

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
        rgb = cv2.cvtColor(torso, cv2.COLOR_BGR2RGB).astype(np.float32)

        flat_rgb = rgb.reshape(-1, 3)
        flat_hsv = hsv.reshape(-1, 3).astype(np.float32)
        flat_lab = lab.reshape(-1, 3).astype(np.float32)

        H = flat_hsv[:, 0] * 2.0
        S = flat_hsv[:, 1]
        V = flat_hsv[:, 2]

        valid = (V > 18) & (V < 250)
        green = (H >= 28) & (H <= 105) & (S > 42) & (V > 28)
        green_ratio = float(np.mean(green))
        neutral = (S < 58) & (V > 32)
        neutral_ratio = float(np.mean(neutral))

        dark = (V < 105)
        very_dark = (V < 78)
        dark_neutral = (V < 115) & (S < 95)
        dark_ratio = float(np.mean(dark))
        very_dark_ratio = float(np.mean(very_dark))
        dark_neutral_ratio = float(np.mean(dark_neutral))

        if green_ratio > 0.30:
            valid &= ~green
        if int(valid.sum()) < 20:
            valid = V > 18
        if int(valid.sum()) < 20:
            return None, {}

        c = flat_rgb[valid]
        hs = flat_hsv[valid]
        la = flat_lab[valid]

        rgb_sum = c.sum(axis=1, keepdims=True) + 1e-6
        chroma = c / rgb_sum

        def perc(a: np.ndarray, qs) -> np.ndarray:
            return np.percentile(a, qs, axis=0).reshape(-1)

        c_p = perc(chroma, (10, 25, 50, 75, 90))
        hsv_p = perc(hs, (25, 50, 75))
        lab_p = perc(la, (25, 50, 75))

        h_hist, _ = np.histogram(
            hs[:, 0], bins=24, range=(0, 180),
            weights=np.clip(hs[:, 1] / 255.0, 0, 1),
        )
        h_hist = h_hist.astype(np.float32)
        if h_hist.sum() > 0:
            h_hist /= h_hist.sum()

        rh, _ = np.histogram(chroma[:, 0], bins=16, range=(0, 1))
        gh, _ = np.histogram(chroma[:, 1], bins=16, range=(0, 1))
        bh, _ = np.histogram(chroma[:, 2], bins=16, range=(0, 1))
        rgb_hist = np.concatenate([rh, gh, bh]).astype(np.float32)
        if rgb_hist.sum() > 0:
            rgb_hist /= rgb_hist.sum()

        gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
        edge_density = float(np.mean(cv2.Canny(gray, 45, 120) > 0))

        feature = np.concatenate([
            1.00 * c_p,
            0.28 * hsv_p,
            0.10 * lab_p,
            2.00 * h_hist,
            1.35 * rgb_hist,
            np.array([0.25 * edge_density, 0.20 * neutral_ratio], dtype=np.float32),
        ]).astype(np.float32)

        if not np.isfinite(feature).all():
            return None, {}

        return feature, {
            "brightness": float(np.median(hs[:, 2])),
            "saturation": float(np.median(hs[:, 1])),
            "green_ratio": green_ratio,
            "neutral_ratio": neutral_ratio,
            "dark_ratio": dark_ratio,
            "very_dark_ratio": very_dark_ratio,
            "dark_neutral_ratio": dark_neutral_ratio,
            "edge_density": edge_density,
        }

    @staticmethod
    def _robust_scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0)
        scale = 1.4826 * mad
        scale[scale < 1e-3] = 1.0
        Z = np.clip((X - med) / scale, -5.0, 5.0)
        return Z, med, scale

    @staticmethod
    def _confidence(distances: np.ndarray, core: bool) -> float:
        d = np.sort(distances.astype(np.float32))
        if len(d) < 2:
            return 0.0
        margin = float(d[1] - d[0])
        separation = margin / max(float(d[1] + d[0]), 1e-6)
        return float(np.clip(0.70 * separation + 0.30 * (1.0 if core else 0.75), 0.0, 1.0))

    def fit_predict_scores(
        self,
        track_id_to_crops: dict[int, list[np.ndarray]],
    ) -> tuple[dict[int, int], dict[int, float], dict[int, np.ndarray]]:
        track_ids = [int(tid) for tid, crops in track_id_to_crops.items() if crops]
        if len(track_ids) < 4:
            self.last_model = None
            return {}, {}, {}

        profiles: dict[int, np.ndarray] = {}
        for tid in track_ids:
            feats = []
            for crop in track_id_to_crops[tid]:
                f, _meta = self.build_color_feature(crop)
                if f is not None:
                    feats.append(f)
            if len(feats) >= 2:
                p = np.median(np.stack(feats), axis=0).astype(np.float32)
                if np.isfinite(p).all():
                    profiles[tid] = p

        ids = list(profiles.keys())
        if len(ids) < 4:
            self.last_model = None
            return {}, {}, {}

        X = np.vstack([profiles[tid] for tid in ids]).astype(np.float32)
        Z, med, scale = self._robust_scale(X)

        km0 = KMeans(n_clusters=2, n_init=80, max_iter=600, random_state=17)
        labels0 = km0.fit_predict(Z)
        d0 = km0.transform(Z)

        keep = np.zeros(len(ids), dtype=bool)
        for c in (0, 1):
            idx = np.where(labels0 == c)[0]
            if len(idx) < 2:
                continue
            q = float(np.quantile(d0[idx, c], 0.85))
            keep[idx] = d0[idx, c] <= max(q, 1.50)

        if np.sum(labels0 == 0) < 3 or np.sum(labels0 == 1) < 3:
            keep[:] = True
        elif keep.sum() < 6:
            keep[:] = True

        km = KMeans(n_clusters=2, n_init=100, max_iter=800, random_state=23)
        km.fit(Z[keep])

        distances = km.transform(Z)
        labels = np.argmin(distances, axis=1).astype(np.int32)

        result: dict[int, int] = {}
        confidence: dict[int, float] = {}
        embeddings: dict[int, np.ndarray] = {}

        for i, tid in enumerate(ids):
            result[tid] = int(labels[i])
            confidence[tid] = self._confidence(distances[i], bool(keep[i]))
            embeddings[tid] = X[i]

        self.last_model = {
            "ids": ids,
            "profiles": X,
            "scaled": Z,
            "median": med,
            "scale": scale,
            "kmeans": km,
            "keep": keep,
            "labels": labels,
            "distances": distances,
            "cluster_counts": np.bincount(labels, minlength=2),
        }
        return result, confidence, embeddings
