"""Standalone football detector test for FootVision.

Purpose: evaluate goalkeeper/referee/player detection on a reference clip
WITHOUT tracking, ReID, KMeans, HSV heuristics, or PRTReID.

Model: gianpaj/football-players-detection-1, YOLOv8x fine-tuned for
ball/goalkeeper/player/referee detection.
"""

from pathlib import Path

import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


MODEL_REPO = "gianpaj/football-players-detection-1"
MODEL_FILE = "weights/best.pt"

INPUT_VIDEO = "finalrecortada.mp4"
OUTPUT_VIDEO = "resultado.mp4"

CONFIDENCE = 0.20
IOU = 0.50
IMGSZ = 1280
DEVICE = 0  

COLORS = {
    "player": (255, 80, 80),
    "goalkeeper": (0, 220, 255),
    "referee": (255, 0, 180),
    "ball": (255, 255, 255),
}


def main() -> None:
    input_path = Path(INPUT_VIDEO)
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el vídeo: {input_path}")

    print("Descargando/localizando modelo...")
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )

    print(f"Modelo: {model_path}")
    model = YOLO(model_path)

    print("Clases del modelo:")
    print(model.names)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("No se pudo abrir VideoWriter")

    frame_idx = 0
    counts = {name: 0 for name in COLORS}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(
                source=frame,
                conf=CONFIDENCE,
                iou=IOU,
                imgsz=IMGSZ,
                device=DEVICE,
                verbose=False,
            )
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()

                for box, cls_id, conf in zip(boxes, classes, confs):
                    name = str(model.names[int(cls_id)]).strip().lower()
                    if name not in COLORS:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    color = COLORS[name]
                    counts[name] += 1

                    if name == "ball":
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        radius = max(3, int(max(x2 - x1, y2 - y1) / 2))
                        cv2.circle(frame, (cx, cy), radius, color, 2)
                        label = f"BALL {conf:.2f}"
                        text_x, text_y = x1, max(18, y1 - 6)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{name.upper()} {conf:.2f}"
                        text_x, text_y = x1, max(18, y1 - 6)

                    cv2.putText(
                        frame,
                        label,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            cv2.putText(
                frame,
                f"Frame {frame_idx}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame)
            frame_idx += 1

            if frame_idx % 30 == 0:
                print(f"Procesados {frame_idx} frames...")

    finally:
        cap.release()
        writer.release()

    print("\n=== RESULTADO ===")
    print(f"Frames: {frame_idx}")
    for name, count in counts.items():
        print(f"{name:12s}: {count}")
    print(f"Vídeo: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
