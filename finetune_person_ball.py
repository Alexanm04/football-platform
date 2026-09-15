from ultralytics import YOLO

model = YOLO("yolo-football-player-detection.pt")
model.train(
    data="./football-dataset-merged/data.yaml",
    epochs=30,
    imgsz=960,
    batch=4,
    patience=15,
    augment=True,
    cache=True,
    workers=4,
)