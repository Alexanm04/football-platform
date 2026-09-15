from __future__ import annotations

import os
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from .team_classifier import TeamClassifier

logger = logging.getLogger(__name__)

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")


class VisionService:
    """
    VisionService V47.

    Ajustes:
      - Balón desactivado.
      - Supresión espacial de extremidades (Limb Suppression).
      - Interfaz de renderizado limpia y profesional (Broadcast UI).
    """

    PERSON_ALIASES = {
        "person", "player", "football player", "football_player",
        "goalkeeper", "referee", "official",
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        tracker: Optional[str] = None,
        device: Optional[str] = None,
        conf: float = 0.20,
        iou: float = 0.50,
        imgsz: int = 960,
        max_persons: int = 60,
        sample_every: int = 2,
        min_track_samples: int = 5,
        min_team_tracks: int = 4,
    ) -> None:
        self.device = device or ("cuda" if self._cuda_available() else "cpu")
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.max_persons = int(max_persons)
        self.sample_every = max(1, int(sample_every))
        self.min_track_samples = max(3, int(min_track_samples))
        self.min_team_tracks = max(3, int(min_team_tracks))

        project_root = Path(__file__).resolve().parents[4]
        model_path_value = model_path or os.getenv(
            "FOOTBALL_MODEL_PATH",
            str(project_root / "yolo-person-ball-v1.pt"),
        )
        model_path_obj = Path(model_path_value).expanduser()
        if not model_path_obj.is_absolute():
            model_path_obj = project_root / model_path_obj
        self.model_path = str(model_path_obj.resolve())

        tracker_value = tracker or os.getenv(
            "FOOTBALL_TRACKER_PATH",
            str(Path(__file__).resolve().parent / "botsort_football.yaml"),
        )
        tracker_path = Path(
            tracker_value
        ).expanduser()
        if not tracker_path.is_absolute():
            base_path = (
                Path(__file__).resolve().parent if tracker else project_root
            )
            tracker_path = base_path / tracker_path
        tracker_path = tracker_path.resolve()
        if not tracker_path.is_file():
            raise FileNotFoundError(f"[Vision V47] No existe el tracker: {tracker_path}")
        self.tracker = str(tracker_path)

        logger.info("VisionService configurado con device=%s", self.device)

        self.model = YOLO(self.model_path)
        if self.device == "cuda":
            try:
                self.model.to("cuda")
                self.model.fuse()
            except Exception:
                logger.exception("No se pudo preparar CUDA para VisionService")

        self.class_names = self._get_class_names()
        self.person_class_ids = self._resolve_class_ids()

        self.team_classifier = TeamClassifier(
            device=self.device,
            use_siglip=False,
        )

        self.draw_colors = {
            "TEAM_A": (50, 50, 230),   
            "TEAM_B": (230, 200, 50), 
            "GOALKEEPER": (50, 220, 50), 
            "REFEREE": (30, 30, 30),    
            "UNKNOWN": (160, 160, 160),
        }

        self._canonical_next_id = 1
        self._raw_to_canonical: dict[int, int] = {}
        self._canonical_memory: dict[int, dict[str, Any]] = {}
        self._identity_max_gap = 18

    def process_video(self, input_path: str, output_path: str, output_fps: Optional[float] = None) -> str:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"No se puede abrir el vídeo: {input_path}")

        fps = self._safe_float(cap.get(cv2.CAP_PROP_FPS), 30.0) or 30.0
        width = int(self._safe_float(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 1920) or 1920)
        height = int(self._safe_float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 1080) or 1080)
        frame_count = int(self._safe_float(cap.get(cv2.CAP_PROP_FRAME_COUNT), 0) or 0)

        logger.info(
            "Procesando vídeo width=%s height=%s fps=%.2f frames=%s",
            width,
            height,
            fps,
            frame_count,
        )
        logger.info("VisionService fase 1/2: seguimiento y observaciones")

        observations_by_frame: dict[int, list[dict[str, Any]]] = {}
        observations_by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self._reset_identity_memory()
        prev_frame_gray = None

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = self._track_frame(frame)
            obs = self._extract_person_observations(frame, result, frame_idx)
            obs = self._deduplicate_frame_observations(obs)
            if obs:
                obs.sort(key=lambda x: x["conf"], reverse=True)
                obs = obs[: self.max_persons]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            shot_cut = self._detect_shot_cut(prev_frame_gray, gray)
            if shot_cut:
                self._reset_identity_memory()
                try:
                    self.model.predictor = None
                except Exception:
                    pass
                result = self._track_frame(frame)
                obs = self._extract_person_observations(frame, result, frame_idx)
                obs = self._deduplicate_frame_observations(obs)
                if obs:
                    obs.sort(key=lambda x: x["conf"], reverse=True)
                    obs = obs[: self.max_persons]
            prev_frame_gray = gray

            obs = self._canonicalize_observations(obs, frame_idx)
            observations_by_frame[frame_idx] = obs

            if frame_idx % self.sample_every == 0:
                for item in obs:
                    if item["appearance_valid"]:
                        observations_by_track[item["track_id"]].append(item)
            frame_idx += 1

        cap.release()

        observations_by_track = {
            tid: obs
            for tid, obs in observations_by_track.items()
            if len(obs) >= self.min_track_samples
        }
        if not observations_by_track:
            raise RuntimeError("No se han encontrado personas trackeadas en el vídeo.")

        logger.info("Trayectorias estables detectadas: %s", len(observations_by_track))

        profiles = self._build_track_profiles(observations_by_track)

        if len(profiles) < self.min_team_tracks * 2:
            raise RuntimeError("No hay suficientes tracks para descubrir los dos equipos.")

        logger.info("VisionService fase 2/2: equipos y roles especiales")
        team_model = self._discover_teams(profiles)
        labels = self._classify_tracks(profiles, team_model)

        output_parent = Path(output_path).parent
        output_parent.mkdir(parents=True, exist_ok=True)
        temp_avi = output_parent / f"{Path(output_path).stem}_v47_tmp.avi"

        writer = cv2.VideoWriter(
            str(temp_avi),
            cv2.VideoWriter_fourcc(*"MJPG"),
            float(output_fps or fps),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"No se puede crear el vídeo: {temp_avi}")

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            writer.release()
            raise RuntimeError(f"No se puede reabrir el vídeo: {input_path}")

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(self._render_frame(frame, observations_by_frame.get(frame_idx, []), labels))
            frame_idx += 1

        cap.release()
        writer.release()
        self._convert_to_mp4(str(temp_avi), str(output_path))

        try:
            temp_avi.unlink(missing_ok=True)
        except OSError:
            pass

        logger.info("Vídeo procesado correctamente")
        return str(output_path)

    def _reset_identity_memory(self) -> None:
        self._canonical_next_id = 1
        self._raw_to_canonical = {}
        self._canonical_memory = {}

    @staticmethod
    def _detect_shot_cut(prev_gray: np.ndarray | None, gray: np.ndarray) -> bool:
        if prev_gray is None or gray is None or prev_gray.shape != gray.shape:
            return False
        a = cv2.resize(prev_gray, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
        b = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
        diff = float(np.mean(np.abs(a - b)))
        return diff >= 42.0

    @staticmethod
    def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        return float(inter / max(aa + ab - inter, 1e-6))

    @classmethod
    def _deduplicate_frame_observations(
        cls, obs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if len(obs) <= 1:
            return obs

        ordered = sorted(obs, key=lambda x: x.get("conf", 0.0), reverse=True)
        kept: list[dict[str, Any]] = []

        for cand in ordered:
            duplicate = False
            cb = tuple(cand["bbox"])
            for prev in kept:
                pb = tuple(prev["bbox"])
                
                iou = cls._bbox_iou(cb, pb)
                if iou >= 0.55:
                    duplicate = True
                    break

                cxa, cya = (cb[0] + cb[2]) * 0.5, (cb[1] + cb[3]) * 0.5
                pxa, pya = (pb[0] + pb[2]) * 0.5, (pb[1] + pb[3]) * 0.5
                ch, ph = max(cb[3] - cb[1], 1.0), max(pb[3] - pb[1], 1.0)
                
                area_c = max((cb[2]-cb[0])*(cb[3]-cb[1]), 1.0)
                area_p = max((pb[2]-pb[0])*(pb[3]-pb[1]), 1.0)
                size_ratio = min(area_c, area_p) / max(area_c, area_p)

                if size_ratio < 0.30:
                    small_box = cb if area_c < area_p else pb
                    big_box = pb if area_c < area_p else cb
                    
                    sb_cx, sb_cy = (small_box[0] + small_box[2]) * 0.5, (small_box[1] + small_box[3]) * 0.5
                    bb_cx, bb_cy = (big_box[0] + big_box[2]) * 0.5, (big_box[1] + big_box[3]) * 0.5
                    bb_w, bb_h = big_box[2] - big_box[0], big_box[3] - big_box[1]
                    
                    if abs(sb_cx - bb_cx) < bb_w * 0.9 and abs(sb_cy - bb_cy) < bb_h * 0.9:
                        duplicate = True
                        break

                ix1, iy1 = max(cb[0], pb[0]), max(cb[1], pb[1])
                ix2, iy2 = min(cb[2], pb[2]), min(cb[3], pb[3])
                inter_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                
                io_min = inter_area / min(area_c, area_p)
                if io_min >= 0.70:
                    duplicate = True
                    break

            if not duplicate:
                kept.append(cand)

        return kept

    def _canonicalize_observations(
        self, obs: list[dict[str, Any]], frame_idx: int
    ) -> list[dict[str, Any]]:
        if not obs:
            return []

        expired = [
            cid for cid, st in self._canonical_memory.items()
            if frame_idx - int(st["last_frame"]) > self._identity_max_gap
        ]
        for cid in expired:
            self._canonical_memory.pop(cid, None)
            for rid, mapped in list(self._raw_to_canonical.items()):
                if mapped == cid:
                    self._raw_to_canonical.pop(rid, None)

        assigned_cids: set[int] = set()
        for item in obs:
            rid = int(item["track_id"])
            cid = self._raw_to_canonical.get(rid)
            if cid is not None and cid in self._canonical_memory:
                item["track_id"] = cid
                assigned_cids.add(cid)
                self._update_identity_memory(cid, item, frame_idx)

        for item in obs:
            if int(item["track_id"]) in assigned_cids:
                continue
            rid = int(item["track_id"])
            if rid in assigned_cids:
                continue
            best_cid = None
            best_cost = 1e9
            for cid, st in self._canonical_memory.items():
                if cid in assigned_cids:
                    continue
                gap = frame_idx - int(st["last_frame"])
                if gap <= 0 or gap > self._identity_max_gap:
                    continue
                cost = self._identity_match_cost(item, st)
                if cost < best_cost:
                    best_cost = cost
                    best_cid = cid

            if best_cid is not None and best_cost <= 1.35:
                self._raw_to_canonical[rid] = best_cid
                item["track_id"] = best_cid
                assigned_cids.add(best_cid)
                self._update_identity_memory(best_cid, item, frame_idx)
            else:
                cid = self._canonical_next_id
                self._canonical_next_id += 1
                self._raw_to_canonical[rid] = cid
                item["track_id"] = cid
                assigned_cids.add(cid)
                self._update_identity_memory(cid, item, frame_idx)

        return obs

    def _update_identity_memory(self, cid: int, item: dict[str, Any], frame_idx: int) -> None:
        st = self._canonical_memory.get(cid)
        feat = np.asarray(item.get("features", np.zeros(0, dtype=np.float32)), dtype=np.float32)
        if st is None:
            self._canonical_memory[cid] = {
                "last_frame": frame_idx,
                "bbox": tuple(item["bbox"]),
                "foot": np.asarray([item["foot_x"], item["foot_y"]], dtype=np.float32),
                "height": float(item.get("height", 1.0)),
                "feature": feat.copy(),
            }
            return
        st["last_frame"] = frame_idx
        st["bbox"] = tuple(item["bbox"])
        st["foot"] = np.asarray([item["foot_x"], item["foot_y"]], dtype=np.float32)
        st["height"] = float(item.get("height", st.get("height", 1.0)))
        old = np.asarray(st.get("feature", np.zeros(0)), dtype=np.float32)
        if len(feat) and len(old) == len(feat):
            st["feature"] = (0.75 * old + 0.25 * feat).astype(np.float32)
        elif len(feat):
            st["feature"] = feat.copy()

    def _identity_match_cost(self, item: dict[str, Any], st: dict[str, Any]) -> float:
        foot = np.asarray([item["foot_x"], item["foot_y"]], dtype=np.float32)
        last_foot = np.asarray(st["foot"], dtype=np.float32)
        scale = max(float(item.get("height", 1.0)), float(st.get("height", 1.0)), 20.0)
        pos_cost = float(np.linalg.norm(foot - last_foot) / scale)

        feat = np.asarray(item.get("features", np.zeros(0)), dtype=np.float32)
        old = np.asarray(st.get("feature", np.zeros(0)), dtype=np.float32)
        feat_cost = 0.0
        if len(feat) and len(old) == len(feat):
            denom = max(float(np.sqrt(len(feat))), 1.0)
            feat_cost = float(np.linalg.norm(feat - old) / denom)

        if feat_cost > 1.25:
            return 10.0

        return 0.62 * min(pos_cost / 4.0, 2.0) + 0.38 * min(feat_cost, 2.0)

    def _track_frame(self, frame: np.ndarray) -> Any:
        kwargs = {
            "persist": True,
            "tracker": self.tracker,
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "verbose": False,
        }
        class_ids = sorted(self.person_class_ids)
        if class_ids:
            kwargs["classes"] = class_ids

        try:
            result = self.model.track(frame, device=self.device, **kwargs)
        except TypeError:
            kwargs.pop("device", None)
            result = self.model.track(frame, **kwargs)
        return result[0] if result else None

    def _extract_person_observations(
        self, frame: np.ndarray, result: Any, frame_idx: int
    ) -> list[dict[str, Any]]:
        if result is None or result.boxes is None:
            return []
        boxes = result.boxes
        if boxes.xyxy is None or boxes.cls is None or boxes.id is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
        classes = boxes.cls.cpu().numpy()
        ids = boxes.id.cpu().numpy()

        out = []
        for i, box in enumerate(xyxy):
            cid = int(classes[i])
                
            if cid not in self.person_class_ids:
                continue

            tid_f = self._safe_float(ids[i], None)
            if tid_f is None:
                continue
            tid = int(tid_f)

            x1, y1, x2, y2 = [self._safe_float(v, None) for v in box]
            if None in (x1, y1, x2, y2):
                continue

            geom = self._validate_person_geometry(x1, y1, x2, y2, frame.shape)
            if not geom["valid"]:
                continue

            appearance = self._extract_jersey_features(frame, x1, y1, x2, y2)
            out.append({
                "track_id": tid,
                "frame": frame_idx,
                "conf": self._safe_float(confs[i] if confs is not None else None, 0.0) or 0.0,
                "bbox": (x1, y1, x2, y2),
                **geom,
                **appearance,
            })
        return out

    @staticmethod
    def _validate_person_geometry(
        x1: float, y1: float, x2: float, y2: float, shape: tuple[int, ...]
    ) -> dict[str, Any]:
        h, w = shape[:2]
        x1 = max(0.0, min(w - 1.0, float(x1)))
        x2 = max(0.0, min(w - 1.0, float(x2)))
        y1 = max(0.0, min(h - 1.0, float(y1)))
        y2 = max(0.0, min(h - 1.0, float(y2)))
        bw, bh = x2 - x1, y2 - y1
        if bw <= 2 or bh <= 8:
            return {"valid": False}
        ratio = bh / max(bw, 1e-6)
        if ratio < 0.35 or ratio > 8.0:
            return {"valid": False}
        
        if bh < max(18.0, h * 0.02):
            return {"valid": False}
            
        return {
            "valid": True,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "cx": (x1 + x2) * 0.5,
            "cy": (y1 + y2) * 0.5,
            "foot_x": (x1 + x2) * 0.5,
            "foot_y": y2,
            "area": bw * bh,
            "height": bh,
            "width": bw,
            "ratio": ratio,
        }

    def _extract_jersey_features(
        self, frame: np.ndarray, x1: float, y1: float, x2: float, y2: float
    ) -> dict[str, Any]:
        h, w = frame.shape[:2]
        ix1, ix2 = max(0, int(x1)), min(w, int(x2))
        iy1, iy2 = max(0, int(y1)), min(h, int(y2))
        if ix2 <= ix1 or iy2 <= iy1:
            return self._empty_appearance()

        crop = frame[iy1:iy2, ix1:ix2]
        ch, cw = crop.shape[:2]
        if ch < 16 or cw < 8:
            return self._empty_appearance()

        feature, meta = self.team_classifier.build_color_feature(crop)
        if feature is None:
            return self._empty_appearance()

        return {
            "appearance_valid": True,
            "features": feature,
            "brightness": meta["brightness"],
            "saturation": meta["saturation"],
            "green_ratio": meta["green_ratio"],
            "neutral_ratio": meta["neutral_ratio"],
            "dark_ratio": meta["dark_ratio"],
            "very_dark_ratio": meta["very_dark_ratio"],
            "dark_neutral_ratio": meta["dark_neutral_ratio"],
            "edge_density": meta["edge_density"],
            "crop": crop.copy(),
        }

    @staticmethod
    def _empty_appearance() -> dict[str, Any]:
        return {
            "appearance_valid": False,
            "features": np.zeros(0, dtype=np.float32),
            "brightness": 0.0,
            "saturation": 0.0,
            "green_ratio": 0.0,
            "neutral_ratio": 0.0,
            "dark_ratio": 0.0,
            "very_dark_ratio": 0.0,
            "dark_neutral_ratio": 0.0,
            "edge_density": 0.0,
            "crop": None,
        }

    def _build_track_profiles(
        self, observations_by_track: dict[int, list[dict[str, Any]]]
    ) -> dict[int, dict[str, Any]]:
        profiles: dict[int, dict[str, Any]] = {}

        for tid, obs in observations_by_track.items():
            valid = [o for o in obs if o["appearance_valid"] and len(o["features"]) > 0]
            if len(valid) < self.min_track_samples:
                continue

            X = np.vstack([o["features"] for o in valid]).astype(np.float32)
            feature_median = np.median(X, axis=0)
            if not np.isfinite(feature_median).all():
                continue

            feature_mean = np.mean(X, axis=0)
            profile = (0.72 * feature_median + 0.28 * feature_mean).astype(np.float32)

            positions = np.array([[o["cx"], o["cy"]] for o in valid], dtype=np.float32)
            feet = np.array([[o["foot_x"], o["foot_y"]] for o in valid], dtype=np.float32)
            frames = np.array([int(o["frame"]) for o in valid], dtype=np.int32)
            scales = np.array([max(o["height"], 1.0) for o in valid], dtype=np.float32)
            
            if len(feet) > 1:
                delta = np.linalg.norm(np.diff(feet, axis=0), axis=1)
                norm_motion = float(np.median(delta / np.maximum(scales[1:], 1.0)))
            else:
                norm_motion = 0.0

            profiles[tid] = {
                "feature": profile,
                "samples": len(valid),
                "median_brightness": float(np.median([o["brightness"] for o in valid])),
                "median_saturation": float(np.median([o["saturation"] for o in valid])),
                "median_green": float(np.median([o["green_ratio"] for o in valid])),
                "median_neutral": float(np.median([o["neutral_ratio"] for o in valid])),
                "median_dark": float(np.median([o["dark_ratio"] for o in valid])),
                "median_very_dark": float(np.median([o["very_dark_ratio"] for o in valid])),
                "median_dark_neutral": float(np.median([o["dark_neutral_ratio"] for o in valid])),
                "median_edge": float(np.median([o["edge_density"] for o in valid])),
                "positions": positions,
                "feet": feet,
                "frames": frames,
                "scales": scales,
                "crops": [o["crop"] for o in valid if o["crop"] is not None],
                "norm_motion": norm_motion,
            }

        return profiles

    @staticmethod
    def _early_referee_score(profile: dict[str, Any]) -> float:
        frames = np.asarray(profile.get("frames", np.zeros(0)), dtype=np.int32)
        crops = profile.get("crops", [])
        if len(frames) == 0 or not crops:
            return 0.0

        order = np.argsort(frames)
        n = min(len(order), 18)
        vals = []
        for idx in order[:n]:
            crop = crops[int(idx)] if int(idx) < len(crops) else None
            if crop is None or getattr(crop, "size", 0) == 0:
                continue
            cg, cc, cd, bg, bc, bd = VisionService._shirt_region_stats(crop)
            dark = float(np.clip(cd, 0.0, 1.0))
            contrast = float(np.clip(cd - bd + 0.10, 0.0, 1.0))
            low_chroma = float(np.clip(1.0 - cc, 0.0, 1.0))
            vals.append(0.58 * dark + 0.30 * contrast + 0.12 * low_chroma)

        if not vals:
            return 0.0
        a = np.asarray(vals, dtype=np.float32)
        return float(np.clip(0.50 * np.quantile(a, 0.50) +
                             0.50 * np.quantile(a, 0.80), 0.0, 1.0))

    @staticmethod
    def _pre_referee_score(profile: dict[str, Any]) -> float:
        shirt = float(np.clip(VisionService._referee_shirt_score(profile), 0.0, 1.0))
        dark_neutral = float(np.clip(profile.get("median_dark_neutral", 0.0), 0.0, 1.0))
        neutral = float(np.clip(profile.get("median_neutral", 0.0), 0.0, 1.0))
        sat = float(np.clip(profile.get("median_saturation", 0.0) / 255.0, 0.0, 1.0))
        early = float(np.clip(VisionService._early_referee_score(profile), 0.0, 1.0))

        score = (
            0.45 * shirt
            + 0.18 * dark_neutral
            + 0.10 * neutral
            + 0.10 * (1.0 - sat)
            + 0.17 * early
        )
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _pre_goalkeeper_score(profile: dict[str, Any]) -> float:
        shirt = float(np.clip(VisionService._goalkeeper_shirt_score(profile), 0.0, 1.0))
        green = float(np.clip(profile.get("median_green", 0.0) / 0.30, 0.0, 1.0))
        sat = float(np.clip(profile.get("median_saturation", 0.0) / 255.0, 0.0, 1.0))
        return float(np.clip(0.60 * shirt + 0.22 * green + 0.18 * sat, 0.0, 1.0))

    def _discover_teams(self, profiles: dict[int, dict[str, Any]]) -> dict[str, Any]:
        all_ids = [tid for tid in profiles if profiles[tid].get("crops")]
        if len(all_ids) < self.min_team_tracks * 2:
            raise RuntimeError("No hay suficientes tracks con apariencia para descubrir los equipos.")

        ref_ranked = sorted(
            ((tid, self._pre_referee_score(profiles[tid])) for tid in all_ids),
            key=lambda x: x[1],
            reverse=True,
        )
        exclude: set[int] = set()

        for tid, score in ref_ranked:
            if score >= 0.38 or self._referee_shirt_score(profiles[tid]) >= 0.35:
                exclude.add(tid)

        gk_ranked = sorted(
            (
                (tid, self._pre_goalkeeper_score(profiles[tid]))
                for tid in all_ids
                if tid not in exclude
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        if gk_ranked:
            tid, score = gk_ranked[0]
            if score >= 0.65:
                exclude.add(tid)

        if len(all_ids) - len(exclude) < self.min_team_tracks * 2:
            exclude = set()

        fit_ids = [tid for tid in all_ids if tid not in exclude]
        crop_map = {tid: profiles[tid]["crops"] for tid in fit_ids}

        labels, _confidence, _embeddings = self.team_classifier.fit_predict_scores(crop_map)
        model = getattr(self.team_classifier, "last_model", None)
        if model is None or len(labels) < self.min_team_tracks * 2:
            raise RuntimeError(f"El clasificador no ha podido formar dos equipos fiables. Tracks validos={len(labels)}")

        model = dict(model)
        ids = list(model["ids"])
        med = np.asarray(model["median"], dtype=np.float32)
        scale = np.asarray(model["scale"], dtype=np.float32)
        km = model["kmeans"]

        all_model_ids = list(ids)
        all_profiles = [np.asarray(model["profiles"][i], dtype=np.float32) for i in range(len(ids))]
        all_scaled = [np.asarray(model["scaled"][i], dtype=np.float32) for i in range(len(ids))]
        all_labels = [int(model["labels"][i]) for i in range(len(ids))]
        all_distances = [np.asarray(model["distances"][i], dtype=np.float32) for i in range(len(ids))]
        all_keep = [bool(model["keep"][i]) for i in range(len(ids))]

        for tid in exclude:
            f = np.asarray(profiles[tid]["feature"], dtype=np.float32)
            z = np.clip((f - med) / np.maximum(scale, 1e-6), -5.0, 5.0)
            d = km.transform(z.reshape(1, -1))[0].astype(np.float32)
            c = int(np.argmin(d))
            all_model_ids.append(tid)
            all_profiles.append(f)
            all_scaled.append(z)
            all_labels.append(c)
            all_distances.append(d)
            all_keep.append(False)

        model["ids"] = all_model_ids
        model["profiles"] = np.vstack(all_profiles).astype(np.float32)
        model["scaled"] = np.vstack(all_scaled).astype(np.float32)
        model["labels"] = np.asarray(all_labels, dtype=np.int32)
        model["distances"] = np.vstack(all_distances).astype(np.float32)
        model["keep"] = np.asarray(all_keep, dtype=bool)

        for tid in all_model_ids:
            p = profiles[tid]
            ids_idx = model["ids"].index(tid)
            votes = []
            for crop in p.get("crops", []):
                f, _meta = self.team_classifier.build_color_feature(crop)
                if f is None:
                    continue
                z = np.clip((np.asarray(f, dtype=np.float32) - med) / np.maximum(scale, 1e-6), -5.0, 5.0)
                d = km.transform(z.reshape(1, -1))[0]
                votes.append(int(np.argmin(d)))
            if votes:
                counts = np.bincount(np.asarray(votes, dtype=np.int32), minlength=2)
                vote_label = int(np.argmax(counts))
                vote_ratio = float(counts[vote_label] / len(votes))
                
                current = int(model["labels"][ids_idx])
                if vote_label != current and vote_ratio >= 0.68:
                    model["labels"][ids_idx] = vote_label

        proto = {}
        for c in (0, 1):
            idx = [
                i for i, lab in enumerate(model["labels"])
                if bool(model["keep"][i]) and int(lab) == c
            ]
            if idx:
                proto[c] = np.median(model["scaled"][idx], axis=0).astype(np.float32)

        if 0 in proto and 1 in proto:
            for i, tid in enumerate(model["ids"]):
                d0 = float(np.linalg.norm(model["scaled"][i] - proto[0]) / max(np.sqrt(model["scaled"].shape[1]), 1.0))
                d1 = float(np.linalg.norm(model["scaled"][i] - proto[1]) / max(np.sqrt(model["scaled"].shape[1]), 1.0))
                current = int(model["labels"][i])
                other = 1 - current
                dc = d0 if current == 0 else d1
                do = d1 if current == 0 else d0
                if do + 0.08 < dc:
                    model["labels"][i] = other

        model["distances"] = km.transform(model["scaled"]).astype(np.float32)
        counts = np.bincount(model["labels"], minlength=2)
        if counts.min() < self.min_team_tracks:
            raise RuntimeError(f"El descubrimiento de equipos no es fiable: tamaños={counts.tolist()}")

        self.team_classifier.last_model = model

        confidence2 = {}
        for i, tid in enumerate(model["ids"]):
            confidence2[tid] = float(
                self._player_team_confidence(model["distances"][i], bool(model["keep"][i]))
            )

        id_to_i = {tid: i for i, tid in enumerate(model["ids"])}
        for tid in profiles:
            i = id_to_i.get(tid)
            if i is None:
                continue
            profiles[tid]["team_cluster"] = int(model["labels"][i])
            profiles[tid]["team_distance"] = float(model["distances"][i, model["labels"][i]])
            profiles[tid]["team_visual_conf"] = float(confidence2.get(tid, 0.0))
            profiles[tid]["team_core"] = bool(model["keep"][i])

        return model

    def _classify_tracks(
        self, profiles: dict[int, dict[str, Any]], model: dict[str, Any]
    ) -> dict[int, dict[str, Any]]:
        ids = model["ids"]
        id_to_i = {tid: i for i, tid in enumerate(ids)}
        result: dict[int, dict[str, Any]] = {}

        for tid in profiles:
            i = id_to_i.get(tid)
            if i is None:
                continue

            cluster = int(model["labels"][i])
            dist = float(model["distances"][i, cluster])
            conf = float(self._player_team_confidence(model["distances"][i], bool(model["keep"][i])))
            team = "TEAM_A" if cluster == 0 else "TEAM_B"

            result[tid] = {
                "role": "PLAYER",
                "team": team,
                "display": team,
                "confidence": conf,
                "team_distance": dist,
            }

        specials = self._special_scores(profiles, model)

        gk_aliases_all = self._select_optional_goalkeepers(profiles=profiles, specials=specials)

        ref_anchor, ref_aliases = self._select_special_identity(
            profiles=profiles,
            specials=specials,
            role="REFEREE",
            forbidden=set(gk_aliases_all),
        )

        if ref_anchor is not None:
            chain = {ref_anchor}
            changed = True
            while changed:
                changed = False
                for tid in list(ref_aliases):
                    if tid in chain or tid in gk_aliases_all:
                        continue
                    for aid in list(chain):
                        if self._is_same_referee_fragment(profiles, specials, aid, tid):
                            chain.add(tid)
                            changed = True
                            break
            ref_aliases = chain
        else:
            ref_aliases = set()

        if ref_anchor is None:
            remaining = [
                (tid, float(vals["referee_score_final"]))
                for tid, vals in specials.items()
                if tid not in gk_aliases_all
            ]
            remaining.sort(key=lambda x: x[1], reverse=True)

            if remaining:
                top_tid, top_score = remaining[0]
                second_score = remaining[1][1] if len(remaining) > 1 else 0.0
                top = specials[top_tid]

                referee_rank_ok = (
                    top_score >= 0.50
                    and (top_score - second_score) >= 0.05
                    and top["referee_shirt"] >= 0.40
                )

                if referee_rank_ok:
                    ref_anchor = top_tid
                    ref_aliases = {top_tid}
                    logger.info(
                        "Desempate alternativo de árbitro seleccionado: puntuación=%.2f",
                        top_score,
                    )

        for tid, sp in specials.items():
            if tid in gk_aliases_all:
                continue
            if sp.get("referee_shirt", 0.0) >= 0.45 and sp.get("pre_referee", 0.0) >= 0.42:
                ref_aliases.add(tid)

        gk_canonical = min(gk_aliases_all) if gk_aliases_all else None
        for tid in gk_aliases_all:
            if tid in result:
                result[tid] = {
                    "role": "GOALKEEPER",
                    "team": "SPECIAL",
                    "display": "GOALKEEPER",
                    "confidence": float(np.clip(specials[tid]["gk_score"], 0.0, 1.0)),
                    "team_distance": result[tid]["team_distance"],
                    "canonical_id": int(gk_canonical if gk_canonical is not None else tid),
                }

        for tid in ref_aliases:
            if tid in result and tid not in gk_aliases_all:
                result[tid] = {
                    "role": "REFEREE",
                    "team": "SPECIAL",
                    "display": "REFEREE",
                    "confidence": float(np.clip(specials[tid]["referee_score"], 0.0, 1.0)),
                    "team_distance": result[tid]["team_distance"],
                    "canonical_id": int(ref_anchor if ref_anchor is not None else tid),
                }

        return result

    @staticmethod
    def _shirt_region_stats(crop: np.ndarray) -> tuple[float, float, float, float, float, float]:
        h, w = crop.shape[:2]
        if h < 18 or w < 8:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        y1, y2 = int(h * 0.12), int(h * 0.68)
        x1, x2 = int(w * 0.10), int(w * 0.90)
        torso = crop[y1:y2, x1:x2]
        if torso.size == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        th, tw = torso.shape[:2]
        cy1, cy2 = int(th * 0.08), int(th * 0.66)
        cx1, cx2 = int(tw * 0.25), int(tw * 0.75)
        center = torso[cy1:cy2, cx1:cx2]
        if center.size == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        mask_border = np.ones((th, tw), dtype=bool)
        mask_border[cy1:cy2, cx1:cx2] = False
        border = torso[mask_border]
        if border.size == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        border = border.reshape(-1, 1, 3)

        def feats(img: np.ndarray) -> tuple[float, float, float]:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            H = hsv[:, :, 0].astype(np.float32) * 2.0
            S = hsv[:, :, 1].astype(np.float32)
            V = hsv[:, :, 2].astype(np.float32)
            green = np.mean(
                (H >= 32.0) & (H <= 105.0) &
                (S >= 50.0) & (V >= 32.0)
            )
            colorful = np.mean((S >= 72.0) & (V >= 35.0))
            dark_neutral = np.mean((V < 132.0) & (S < 125.0))
            return float(green), float(colorful), float(dark_neutral)

        cg, cc, cd = feats(center)
        bg, bc, bd = feats(border)
        return cg, cc, cd, bg, bc, bd

    @staticmethod
    def _referee_shirt_score(profile: dict[str, Any]) -> float:
        crops = profile.get("crops", [])
        if not crops:
            return 0.0

        per_frame: list[float] = []
        for crop in crops:
            if crop is None or crop.size == 0:
                continue
            
            cg, cc, cd, bg, bc, bd = VisionService._shirt_region_stats(crop)

            h, w = crop.shape[:2]
            pants_crop = crop[int(h * 0.60):int(h * 0.95), int(w * 0.25):int(w * 0.75)]
            pants_dark = 0.0
            if pants_crop.size > 0:
                hsv_p = cv2.cvtColor(pants_crop, cv2.COLOR_BGR2HSV)
                pants_dark = float(np.mean(hsv_p[:, :, 2] < 110.0))

            dark_center = float(np.clip(cd, 0.0, 1.0))
            dark_contrast = float(np.clip(dark_center - bd + 0.10, 0.0, 1.0))
            low_chroma = float(np.clip(1.0 - cc, 0.0, 1.0))

            uniform_darkness = 0.60 * dark_center + 0.40 * pants_dark

            score = (
                0.55 * uniform_darkness
                + 0.25 * dark_contrast
                + 0.20 * low_chroma
            )
            per_frame.append(float(np.clip(score, 0.0, 1.0)))

        if not per_frame:
            return 0.0

        vals = np.asarray(per_frame, dtype=np.float32)
        q60 = float(np.quantile(vals, 0.60))
        q80 = float(np.quantile(vals, 0.80))
        return float(np.clip(0.65 * q60 + 0.35 * q80, 0.0, 1.0))

    @staticmethod
    def _goalkeeper_shirt_score(profile: dict[str, Any]) -> float:
        crops = profile.get("crops", [])
        if not crops:
            return 0.0

        per_frame: list[float] = []
        for crop in crops:
            if crop is None or crop.size == 0:
                continue
            cg, cc, cd, bg, bc, bd = VisionService._shirt_region_stats(crop)

            green_center = float(np.clip(cg, 0.0, 1.0))
            color_center = float(np.clip(cc, 0.0, 1.0))
            green_contrast = float(np.clip(cg - bg + 0.12, 0.0, 1.0))
            color_contrast = float(np.clip(cc - bc + 0.08, 0.0, 1.0))

            score = (
                0.38 * green_center
                + 0.20 * green_contrast
                + 0.27 * color_center
                + 0.15 * color_contrast
            )
            per_frame.append(float(np.clip(score, 0.0, 1.0)))

        if not per_frame:
            return 0.0

        vals = np.asarray(per_frame, dtype=np.float32)
        q60 = float(np.quantile(vals, 0.60))
        q82 = float(np.quantile(vals, 0.82))
        return float(np.clip(0.68 * q60 + 0.32 * q82, 0.0, 1.0))

    @staticmethod
    def _player_team_confidence(distances: np.ndarray, core: bool) -> float:
        d = np.sort(np.asarray(distances, dtype=np.float32))
        if len(d) < 2:
            return 0.40
        best = float(d[0])
        second = float(d[1])
        margin = max(0.0, second - best)
        separation = margin / max(second + best, 1e-6)
        base = 0.92 if core else 0.78
        return float(np.clip(0.55 * separation + 0.45 * base, 0.45, 0.99))

    def _build_frame_team_centroids(
        self,
        profiles: dict[int, dict[str, Any]],
    ) -> dict[int, dict[int, np.ndarray]]:
        frame_points: dict[int, dict[int, list[np.ndarray]]] = defaultdict(
            lambda: {0: [], 1: []}
        )
        for tid, p in profiles.items():
            if not p.get("team_core", False):
                continue
            c = int(p.get("team_cluster", -1))
            if c not in (0, 1):
                continue
            for frame_no, foot in zip(p["frames"], p["feet"]):
                frame_points[int(frame_no)][c].append(
                    np.asarray(foot, dtype=np.float32)
                )
        out: dict[int, dict[int, np.ndarray]] = {}
        for frame_no, groups in frame_points.items():
            item: dict[int, np.ndarray] = {}
            for c in (0, 1):
                pts = groups[c]
                if pts:
                    item[c] = np.median(np.stack(pts), axis=0)
            if item:
                out[frame_no] = item
        return out

    def _track_spatial_stats(
        self,
        profile: dict[str, Any],
        frame_centroids: dict[int, dict[int, np.ndarray]],
    ) -> dict[str, float]:
        ts: list[float] = []

        for frame_no, foot in zip(profile["frames"], profile["feet"]):
            cent = frame_centroids.get(int(frame_no), {})
            if 0 not in cent or 1 not in cent:
                continue

            c0 = cent[0].astype(np.float32)
            c1 = cent[1].astype(np.float32)
            axis = c1 - c0
            sep = float(np.linalg.norm(axis))

            if sep < 35.0:
                continue

            u = axis / sep
            t = float(np.dot(foot.astype(np.float32) - c0, u) / sep)
            ts.append(t)

        if not ts:
            return {
                "central_ratio": 0.0,
                "deep_ratio": 0.0,
                "side_consistency": 0.0,
            }

        t = np.asarray(ts, dtype=np.float32)
        central_ratio = float(np.mean((t >= 0.32) & (t <= 0.68)))
        deep_ratio = float(np.mean((t <= 0.16) | (t >= 0.84)))

        side0 = float(np.mean(t <= 0.28))
        side1 = float(np.mean(t >= 0.72))
        side_consistency = max(side0, side1)

        return {
            "central_ratio": central_ratio,
            "deep_ratio": deep_ratio,
            "side_consistency": side_consistency,
        }

    def _special_scores(
        self,
        profiles: dict[int, dict[str, Any]],
        model: dict[str, Any],
    ) -> dict[int, dict[str, float]]:
        frame_centroids = self._build_frame_team_centroids(profiles)

        distances = np.asarray(model["distances"], dtype=np.float32)
        keep = np.asarray(model["keep"], dtype=bool)
        labels = np.asarray(model["labels"], dtype=np.int32)
        id_to_i = {tid: i for i, tid in enumerate(model["ids"])}

        core_radii = np.ones(2, dtype=np.float32)
        for c in (0, 1):
            idx = np.where(keep & (labels == c))[0]
            if len(idx):
                vals = distances[idx, c]
                med = float(np.median(vals))
                q = float(np.quantile(vals, 0.75))
                core_radii[c] = max(0.5, q, med * 1.15)

        raw: dict[int, dict[str, float]] = {}

        for tid, p in profiles.items():
            cluster = int(p.get("team_cluster", 0))
            i = id_to_i[tid]
            d_assigned = float(distances[i, cluster])
            d_other = float(distances[i, 1 - cluster])
            d_min = min(d_assigned, d_other)
            radius = float(core_radii[cluster])

            visual_anomaly = float(np.clip((d_assigned / max(radius, 1e-6) - 1.0) / 1.60, 0.0, 1.0))
            dual_outlier = float(np.clip((d_min / max(float(np.mean(core_radii)), 1e-6) - 0.85) / 1.45, 0.0, 1.0))

            spatial = self._track_spatial_stats(p, frame_centroids)
            central = spatial["central_ratio"]
            deep = spatial["deep_ratio"]
            side_consistency = spatial["side_consistency"]

            sat = float(np.clip(p["median_saturation"] / 255.0, 0.0, 1.0))
            neutral = float(np.clip(p["median_neutral"], 0.0, 1.0))
            green = float(np.clip(p["median_green"] / 0.30, 0.0, 1.0))
            dark = float(np.clip(p["median_dark"], 0.0, 1.0))
            very_dark = float(np.clip(p["median_very_dark"], 0.0, 1.0))
            dark_neutral = float(np.clip(p["median_dark_neutral"], 0.0, 1.0))
            motion = float(np.clip(p["norm_motion"] / 0.50, 0.0, 1.0))

            referee_appearance = float(
                np.clip(
                    0.38 * dark_neutral + 0.27 * dark + 0.20 * very_dark
                    + 0.10 * neutral + 0.05 * (1.0 - sat), 0.0, 1.0
                )
            )

            referee_shirt = self._referee_shirt_score(p)
            early_referee = self._early_referee_score(p)
            pre_referee = self._pre_referee_score(p)

            goalkeeper_shirt = self._goalkeeper_shirt_score(p)
            team_outlier = 0.0
            core_idx = np.where(keep & (labels == cluster))[0]
            if len(core_idx) >= 3:
                core_d = distances[core_idx, cluster]
                team_outlier = float(np.mean(core_d <= d_assigned))

            gk_signature = float(np.clip(
                0.55 * goalkeeper_shirt + 0.25 * team_outlier
                + 0.12 * dual_outlier + 0.08 * visual_anomaly, 0.0, 1.0
            ))

            gk_score = (
                0.42 * goalkeeper_shirt + 0.22 * team_outlier + 0.16 * deep
                + 0.08 * side_consistency + 0.08 * dual_outlier + 0.04 * motion
            )

            if p.get("team_core", False):
                if goalkeeper_shirt >= 0.46 and team_outlier >= 0.68 and (deep >= 0.32 or side_consistency >= 0.52):
                    gk_score *= 0.98
                elif goalkeeper_shirt >= 0.38 and team_outlier >= 0.62:
                    gk_score *= 0.84
                elif green < 0.35 and team_outlier < 0.55:
                    gk_score *= 0.24
                else:
                    gk_score *= 0.50

            referee_score = (
                0.45 * referee_shirt
                + 0.18 * referee_appearance
                + 0.18 * pre_referee
                + 0.09 * early_referee
                + 0.06 * central
                + 0.04 * dual_outlier
            )

            if p.get("team_core", False):
                if referee_shirt < 0.42 and referee_appearance < 0.50 and dual_outlier < 0.32:
                    referee_score *= 0.20
                elif referee_shirt < 0.52 and referee_appearance < 0.58:
                    referee_score *= 0.55
                else:
                    referee_score *= 0.98

            raw[tid] = {
                "gk_score": float(np.clip(gk_score, 0.0, 1.0)),
                "referee_score": float(np.clip(referee_score, 0.0, 1.0)),
                "visual_anomaly": visual_anomaly,
                "dual_outlier": dual_outlier,
                "central_ratio": central,
                "deep_ratio": deep,
                "side_consistency": side_consistency,
                "motion": motion,
                "referee_appearance": referee_appearance,
                "referee_shirt": referee_shirt,
                "pre_referee": pre_referee,
                "early_referee": early_referee,
                "goalkeeper_shirt": goalkeeper_shirt,
                "team_outlier": team_outlier,
                "gk_signature": gk_signature,
                "d_min": d_min,
                "d_assigned": d_assigned,
            }

        ids = list(raw.keys())
        for tid in ids:
            r = raw[tid]
            r["ref_rank_appearance"] = float(np.mean([raw[x]["referee_appearance"] <= r["referee_appearance"] for x in ids]))
            r["ref_rank_central"] = float(np.mean([raw[x]["central_ratio"] <= r["central_ratio"] for x in ids]))
            r["ref_rank_outlier"] = float(np.mean([raw[x]["dual_outlier"] <= r["dual_outlier"] for x in ids]))
            r["ref_rank_shirt"] = float(np.mean([raw[x]["referee_shirt"] <= r["referee_shirt"] for x in ids]))

            relative = (
                0.50 * r["ref_rank_shirt"]
                + 0.22 * r["ref_rank_appearance"]
                + 0.16 * r["ref_rank_central"]
                + 0.12 * r["ref_rank_outlier"]
            )
            r["referee_relative"] = float(np.clip(relative, 0.0, 1.0))
            r["referee_score_final"] = float(np.clip(0.60 * r["referee_score"] + 0.40 * relative, 0.0, 1.0))

        return raw

    @staticmethod
    def _feature_distance(
        a: np.ndarray,
        b: np.ndarray,
        median: np.ndarray,
        scale: np.ndarray,
    ) -> float:
        za = np.clip((a - median) / scale, -5.0, 5.0)
        zb = np.clip((b - median) / scale, -5.0, 5.0)
        return float(np.linalg.norm(za - zb) / max(np.sqrt(len(za)), 1.0))

    @staticmethod
    def _frame_overlap_ratio(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) == 0 or len(b) == 0:
            return 0.0
        sa = set(int(x) for x in a.tolist())
        sb = set(int(x) for x in b.tolist())
        inter = len(sa.intersection(sb))
        return float(inter / max(1, min(len(sa), len(sb))))

    @staticmethod
    def _temporal_fragment_distance(
        anchor: dict[str, Any],
        candidate: dict[str, Any],
    ) -> tuple[int, float]:
        af = np.asarray(anchor["frames"], dtype=np.int32)
        bf = np.asarray(candidate["frames"], dtype=np.int32)
        if len(af) == 0 or len(bf) == 0:
            return 10_000, 1e9

        if int(af[-1]) <= int(bf[0]):
            end_a = anchor["feet"][-min(5, len(anchor["feet"])):]
            start_b = candidate["feet"][:min(5, len(candidate["feet"]))]
            gap = max(0, int(bf[0]) - int(af[-1]) - 1)
        elif int(bf[-1]) <= int(af[0]):
            end_a = candidate["feet"][-min(5, len(candidate["feet"])):]
            start_b = anchor["feet"][:min(5, len(anchor["feet"]))]
            gap = max(0, int(af[0]) - int(bf[-1]) - 1)
        else:
            return 0, 0.0

        a = np.asarray(end_a, dtype=np.float32)
        b = np.asarray(start_b, dtype=np.float32)
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        min_d = float(np.min(d)) if d.size else 1e9

        scale_a = float(np.median(anchor.get("scales", np.array([1.0]))))
        scale_b = float(np.median(candidate.get("scales", np.array([1.0]))))
        scale = max(1.0, 0.5 * (scale_a + scale_b))

        return gap, min_d / scale

    def _select_optional_goalkeepers(
        self,
        profiles: dict[int, dict[str, Any]],
        specials: dict[int, dict[str, float]],
    ) -> set[int]:
        ranked = sorted(specials.items(), key=lambda kv: float(kv[1]["gk_score"]), reverse=True)

        def strong_candidate(tid: int, minimum: float) -> bool:
            s = specials[tid]
            return (
                s["gk_score"] >= minimum
                and s["goalkeeper_shirt"] >= 0.40
                and s["gk_signature"] >= 0.46
                and s["team_outlier"] >= 0.52
                and (s["deep_ratio"] >= 0.28 or s["side_consistency"] >= 0.48)
            )

        accepted: set[int] = set()
        best_score = 0.0

        for tid, _ in ranked:
            if strong_candidate(tid, 0.55):
                accepted.add(tid)
                best_score = float(specials[tid]["gk_score"])
                logger.info("Portero principal detectado: puntuación=%.2f", best_score)
                break

        if not accepted:
            return accepted

        for tid, _ in ranked:
            if tid in accepted:
                continue

            s = specials[tid]
            cluster = int(profiles[tid].get("team_cluster", -1))

            if not strong_candidate(tid, 0.49):
                continue
                
            if float(s["gk_score"]) < max(0.52, best_score * 0.68):
                continue
                
            if float(s["team_outlier"]) < 0.55:
                continue

            conflict = False
            for acc_tid in accepted:
                acc_cluster = int(profiles[acc_tid].get("team_cluster", -1))
                overlap = self._frame_overlap_ratio(profiles[acc_tid]["frames"], profiles[tid]["frames"])
                
                if cluster == acc_cluster and overlap > 0.05:
                    conflict = True
                    break
                if overlap > 0.30:
                    conflict = True
                    break

            if not conflict:
                accepted.add(tid)
                logger.info(
                    "Portero adicional detectado: puntuación=%.2f",
                    s["gk_score"],
                )

        return accepted

    @staticmethod
    def _is_same_referee_fragment(
        profiles: dict[int, dict[str, Any]],
        specials: dict[int, dict[str, float]],
        anchor_tid: int,
        candidate_tid: int,
    ) -> bool:
        if anchor_tid == candidate_tid:
            return True

        a = profiles.get(anchor_tid)
        b = profiles.get(candidate_tid)
        if a is None or b is None:
            return False

        overlap = VisionService._frame_overlap_ratio(a["frames"], b["frames"])
        if overlap > 0.0:
            return False

        gap, spatial_gap = VisionService._temporal_fragment_distance(a, b)
        if gap > 75 or spatial_gap > 8.5:
            return False

        s = specials.get(candidate_tid, {})
        sa = specials.get(anchor_tid, {})

        shirt = float(s.get("referee_shirt", 0.0))
        anchor_shirt = float(sa.get("referee_shirt", 0.0))
        pre_ref = float(s.get("pre_referee", 0.0))
        anchor_pre_ref = float(sa.get("pre_referee", 0.0))
        appearance = float(s.get("referee_appearance", 0.0))
        anchor_appearance = float(sa.get("referee_appearance", 0.0))

        shirt_delta = abs(shirt - anchor_shirt)
        appearance_delta = abs(appearance - anchor_appearance)
        pre_delta = abs(pre_ref - anchor_pre_ref)

        visual_ok = (
            shirt >= 0.30
            and pre_ref >= 0.30
            and shirt_delta <= 0.28
            and appearance_delta <= 0.23
            and pre_delta <= 0.28
        )

        if bool(b.get("team_core", False)):
            if pre_ref < 0.36 and shirt < 0.40:
                return False

        return visual_ok

    def _select_special_identity(
        self,
        profiles: dict[int, dict[str, Any]],
        specials: dict[int, dict[str, float]],
        role: str,
        forbidden: set[int] | None = None,
        allowed_clusters: set[int] | None = None,
        preferred_anchor: int | None = None,
    ) -> tuple[int | None, set[int]]:
        forbidden = forbidden or set()
        allowed_clusters = allowed_clusters or {0, 1}

        score_key = "gk_score" if role == "GOALKEEPER" else "referee_score_final"
        anchor_threshold = 0.60 if role == "GOALKEEPER" else 0.48

        candidates = [
            (tid, float(vals[score_key]))
            for tid, vals in specials.items()
            if tid not in forbidden
            and (role != "GOALKEEPER" or int(profiles[tid].get("team_cluster", -1)) in allowed_clusters)
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

        if not candidates:
            return None, set()

        if role == "REFEREE" and preferred_anchor in {tid for tid, _ in candidates}:
            pa = specials[int(preferred_anchor)]
            if (pa.get("referee_shirt", 0.0) >= 0.42 and pa.get("pre_referee", 0.0) >= 0.42 and pa.get("early_referee", 0.0) >= 0.35):
                anchor_tid = int(preferred_anchor)
                anchor_score = float(specials[anchor_tid][score_key])
            else:
                anchor_tid, anchor_score = candidates[0]
        else:
            anchor_tid, anchor_score = candidates[0]

        if anchor_score < anchor_threshold:
            if role == "GOALKEEPER":
                strong_gk = [
                    (tid, float(vals[score_key]))
                    for tid, vals in candidates
                    if specials[tid].get("gk_signature", 0.0) >= 0.50
                    and specials[tid].get("goalkeeper_shirt", 0.0) >= 0.38
                    and specials[tid].get("team_outlier", 0.0) >= 0.58
                    and (specials[tid].get("deep_ratio", 0.0) >= 0.40 or specials[tid].get("side_consistency", 0.0) >= 0.58)
                ]
                if strong_gk:
                    strong_gk.sort(key=lambda x: x[1], reverse=True)
                    anchor_tid, anchor_score = strong_gk[0]
                else:
                    return None, set()
            else:
                strong_ref = [
                    (tid, float(vals[score_key]))
                    for tid, vals in candidates
                    if specials[tid].get("early_referee", 0.0) >= 0.48
                    and specials[tid].get("referee_shirt", 0.0) >= 0.34
                ]
                if strong_ref:
                    strong_ref.sort(key=lambda x: (specials[x[0]].get("early_referee", 0.0), x[1]), reverse=True)
                    anchor_tid, anchor_score = strong_ref[0]
                else:
                    return None, set()

        if role == "REFEREE":
            a = specials[anchor_tid]
            if (a["referee_shirt"] < 0.40 and a["referee_appearance"] < 0.42 and a["dual_outlier"] < 0.18 and a["central_ratio"] < 0.22):
                return None, set()

        aliases: set[int] = {anchor_tid}
        anchor = profiles[anchor_tid]

        possible_fragments = []
        for tid, score in candidates:
            if tid in forbidden or tid == anchor_tid:
                continue

            p = profiles[tid]
            overlap = self._frame_overlap_ratio(anchor["frames"], p["frames"])
            if overlap > 0.0:
                continue

            gap, spatial_gap = self._temporal_fragment_distance(anchor, p)
            feat_dist = self._feature_distance(
                np.asarray(anchor["feature"], dtype=np.float32),
                np.asarray(p["feature"], dtype=np.float32),
                np.asarray(self.team_classifier.last_model["median"], dtype=np.float32),
                np.asarray(self.team_classifier.last_model["scale"], dtype=np.float32),
            )

            if role == "GOALKEEPER":
                max_feat_dist = 1.05
                spatial_limit = 5.5 if gap <= 20 else 7.5
            else:
                max_feat_dist = 1.10
                spatial_limit = 8.0 if gap <= 20 else 10.0

            sequential_ok = gap <= (60 if role == "REFEREE" else 60)
            appearance_ok = feat_dist <= max_feat_dist
            spatial_ok = spatial_gap <= spatial_limit

            if role == "REFEREE":
                rs = specials[tid]
                appearance_ok = appearance_ok and (
                    rs.get("referee_shirt", 0.0) >= 0.30
                    and rs.get("pre_referee", 0.0) >= 0.30
                    and abs(rs.get("referee_shirt", 0.0) - specials[anchor_tid].get("referee_shirt", 0.0)) <= 0.28
                    and abs(rs.get("referee_appearance", 0.0) - specials[anchor_tid].get("referee_appearance", 0.0)) <= 0.23
                    and abs(rs.get("pre_referee", 0.0) - specials[anchor_tid].get("pre_referee", 0.0)) <= 0.28
                    and not (p.get("team_core", False) and rs.get("pre_referee", 0.0) < 0.36 and rs.get("referee_shirt", 0.0) < 0.40)
                )

            if role == "GOALKEEPER":
                shirt_delta = abs(specials[tid]["goalkeeper_shirt"] - specials[anchor_tid]["goalkeeper_shirt"])
                green_delta = abs(profiles[tid]["median_green"] - profiles[anchor_tid]["median_green"])
                sat_delta = abs(profiles[tid]["median_saturation"] - profiles[anchor_tid]["median_saturation"])

                appearance_ok = appearance_ok and (
                    (shirt_delta <= 0.24 or (shirt_delta <= 0.34 and green_delta <= 0.16))
                    and sat_delta <= 55.0
                    and (specials[tid]["gk_score"] >= 0.46 or (specials[tid]["gk_signature"] >= 0.50 and specials[tid].get("team_outlier", 0.0) >= 0.62))
                )
                spatial_ok = spatial_ok and (specials[tid]["deep_ratio"] >= 0.32 and specials[tid]["side_consistency"] >= 0.44)

            strong_visual = (
                feat_dist <= (max_feat_dist * 0.72)
                and (
                    specials[tid][score_key] >= anchor_score * 0.65
                    or (role == "REFEREE" and specials[tid].get("referee_shirt", 0.0) >= 0.30 and specials[tid].get("pre_referee", 0.0) >= 0.30)
                )
            )

            if sequential_ok and appearance_ok and spatial_ok:
                possible_fragments.append((tid, feat_dist, gap, spatial_gap, strong_visual))
            elif (role == "REFEREE" and strong_visual and gap <= 90 and specials[tid].get("referee_shirt", 0.0) >= 0.30 and specials[tid].get("pre_referee", 0.0) >= 0.30):
                possible_fragments.append((tid, feat_dist, gap, spatial_gap, strong_visual))

        possible_fragments.sort(key=lambda x: (not x[4], x[2], x[3], x[1]))

        for tid, feat_dist, gap, spatial_gap, strong_visual in possible_fragments:
            conflict = False
            for aid in aliases:
                ov = self._frame_overlap_ratio(profiles[aid]["frames"], profiles[tid]["frames"])
                if ov > 0.15:
                    conflict = True
                    break
            if not conflict:
                aliases.add(tid)

        logger.info(
            "Rol %s clasificado: puntuación=%.2f fragmentos=%s",
            role,
            anchor_score,
            len(aliases),
        )
        return anchor_tid, aliases

    def _render_frame(
        self,
        frame: np.ndarray,
        observations: list[dict[str, Any]],
        labels: dict[int, dict[str, Any]],
    ) -> np.ndarray:
        out = frame.copy()
        
        observations.sort(
            key=lambda obs: (obs["bbox"][2] - obs["bbox"][0]) * (obs["bbox"][3] - obs["bbox"][1]), 
            reverse=True
        )
        
        observations = self._deduplicate_frame_observations(observations)
        
        seen_canonical_ids: set[int] = set()
        seen_roles: dict[str, int] = defaultdict(int)
        drawn_centers: list[tuple[int, int]] = []

        for obs in observations:
            tid = obs["track_id"]
            info = labels.get(tid)
            if info is None:
                continue

            canonical_id = int(info.get("canonical_id", tid))
            role = info.get("role", "PLAYER")

            if canonical_id in seen_canonical_ids:
                continue
                
            if role == "REFEREE" and seen_roles["REFEREE"] >= 1:
                continue
            if role == "GOALKEEPER" and seen_roles["GOALKEEPER"] >= 2:
                continue

            x1, y1, x2, y2 = [int(round(v)) for v in obs["bbox"]]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            is_visual_duplicate = False
            for (dcx, dcy) in drawn_centers:
                if np.hypot(cx - dcx, cy - dcy) < 50:
                    is_visual_duplicate = True
                    break
            
            if is_visual_duplicate:
                continue

            drawn_centers.append((cx, cy))
            seen_canonical_ids.add(canonical_id)
            seen_roles[role] += 1

            display = info["display"]
            color = self.draw_colors.get(display, self.draw_colors["UNKNOWN"])
            
            if display == "TEAM_A":
                final_label = "TEAM A"
            elif display == "TEAM_B":
                final_label = "TEAM B"
            elif display == "GOALKEEPER":
                final_label = "GK"
            elif display == "REFEREE":
                final_label = "REF"
            else:
                final_label = display

            w = x2 - x1
            h = y2 - y1
            ellipse_w = min(w, int(h * 0.9), 70) 

            cv2.ellipse(out, (cx, y2), (int(ellipse_w / 1.8), int(ellipse_w / 5)), 0, 0, 360, color, -1)
            cv2.ellipse(out, (cx, y2), (int(ellipse_w / 1.8), int(ellipse_w / 5)), 0, 0, 360, (255, 255, 255), 1)

            (tw, th), _ = cv2.getTextSize(final_label, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
            pad_x, pad_y = 6, 4
            
            bg_x1 = cx - int(tw / 2) - pad_x
            bg_y1 = y1 - th - pad_y * 2 - 5
            bg_x2 = cx + int(tw / 2) + pad_x
            bg_y2 = y1 - 5
            
            overlay = out.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)
            
            cv2.line(out, (bg_x1, bg_y2), (bg_x2, bg_y2), color, 2)
            
            cv2.putText(
                out,
                final_label,
                (cx - int(tw / 2), y1 - pad_y - 5),
                cv2.FONT_HERSHEY_DUPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            
        return out

    def _get_class_names(self) -> dict[int, str]:
        names = self.model.names
        if isinstance(names, dict):
            return {int(k): str(v).strip().lower() for k, v in names.items()}
        if isinstance(names, (list, tuple)):
            return {i: str(v).strip().lower() for i, v in enumerate(names)}
        return {}

    @staticmethod
    def _normalize_name(name: str) -> str:
        return str(name).strip().lower().replace("-", "_")

    def _resolve_class_ids(self) -> set[int]:
        person_ids = set()
        for cid, name in self.class_names.items():
            n = self._normalize_name(name)
            if n in self.PERSON_ALIASES:
                person_ids.add(cid)
        if not person_ids and self.class_names.get(0) == "person":
            person_ids.add(0)
        return person_ids

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @staticmethod
    def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        if value is None:
            return default
        try:
            out = float(value)
        except (TypeError, ValueError):
            return default
        return out if np.isfinite(out) else default

    def _convert_to_mp4(self, input_avi: str, output_mp4: str) -> None:
        import shutil
        import subprocess

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(f"ffmpeg no está disponible. AVI: {input_avi}")

        cmd = [
            ffmpeg, "-y", "-i", input_avi,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", output_mp4,
        ]
        completed = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if completed.returncode != 0:
            raise RuntimeError("ffmpeg ha fallado:\n" + completed.stderr[-4000:])