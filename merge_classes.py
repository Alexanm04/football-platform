import shutil
from pathlib import Path
import yaml

DATASET_DIR = Path("./football-dataset/data")
OUT_DIR = Path("./football-dataset-merged")

data_yaml = yaml.safe_load((DATASET_DIR / "data.yaml").read_text())
names = data_yaml["names"]  
ball_ids = {i for i, n in enumerate(names) if n.lower() == "ball"}
person_ids = {i for i, n in enumerate(names) if n.lower() in {"player", "goalkeeper", "referee"}}

remap = {}
for i in person_ids:
    remap[i] = 0
for i in ball_ids:
    remap[i] = 1

for split in ("train", "valid", "test"):
    src_labels = DATASET_DIR / split / "labels"
    if not src_labels.exists():
        continue
    dst_images = OUT_DIR / split / "images"
    dst_labels = OUT_DIR / split / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    for img in (DATASET_DIR / split / "images").glob("*"):
        shutil.copy(img, dst_images / img.name)

    for lbl_file in src_labels.glob("*.txt"):
        lines_out = []
        for line in lbl_file.read_text().splitlines():
            if not line.strip():
                continue
            cls, *coords = line.split()
            new_cls = remap.get(int(cls))
            if new_cls is None:
                continue
            lines_out.append(" ".join([str(new_cls), *coords]))
        (dst_labels / lbl_file.name).write_text("\n".join(lines_out))

new_yaml = {
    "path": str(OUT_DIR.resolve()),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "names": {0: "person", 1: "ball"},
}
(OUT_DIR / "data.yaml").write_text(yaml.dump(new_yaml, sort_keys=False))
print("Listo:", OUT_DIR / "data.yaml")
