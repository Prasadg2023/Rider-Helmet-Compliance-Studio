from pathlib import Path
import random

import cv2
import pandas as pd


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "train_labels.csv"
IMAGE_DIR = ROOT / "JPEGImages"
OUTPUT_DIR = ROOT / "dataset"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

CLASS_MAP = {
    "red": 0,
    "white": 0,
    "blue": 0,
    "yellow": 0,
    "none": 1,
}


def ensure_dirs() -> None:
    for split in ["train", "val"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def find_image(base: str) -> tuple[Path | None, str | None]:
    for ext in [".jpg", ".jpeg", ".png"]:
        path = IMAGE_DIR / f"{base}{ext}"
        if path.exists():
            return path, path.name
    return None, None


def build_records() -> list[tuple[Path, str, list[str]]]:
    df = pd.read_csv(CSV_PATH)
    records = []
    for _, row in df.iterrows():
        base = str(row["ID"]).split(".")[0]
        labels = str(row["Label"]).split()
        img_path, img_name = find_image(base)
        if img_path is None or img_name is None:
            continue
        records.append((img_path, img_name, labels))
    random.Random(RANDOM_SEED).shuffle(records)
    return records


def yolo_line(box_tokens: list[str], width: int, height: int) -> str | None:
    if len(box_tokens) != 5:
        return None
    try:
        x1, y1, x2, y2 = map(float, box_tokens[:4])
    except ValueError:
        return None
    cls_name = box_tokens[4]
    if cls_name not in CLASS_MAP:
        return None

    cls_id = CLASS_MAP[cls_name]
    x1 = max(0.0, min(x1, width))
    x2 = max(0.0, min(x2, width))
    y1 = max(0.0, min(y1, height))
    y2 = max(0.0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None

    xc = ((x1 + x2) / 2.0) / width
    yc = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def process(records: list[tuple[Path, str, list[str]]], split: str) -> None:
    for img_path, img_name, labels in records:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        height, width = img.shape[:2]
        output_image = OUTPUT_DIR / "images" / split / img_name
        output_label = OUTPUT_DIR / "labels" / split / f"{Path(img_name).stem}.txt"
        cv2.imwrite(str(output_image), img)

        lines = []
        for i in range(0, len(labels), 5):
            line = yolo_line(labels[i : i + 5], width, height)
            if line is not None:
                lines.append(line)
        output_label.write_text("\n".join(lines), encoding="utf-8")


def write_data_yaml() -> None:
    yaml_text = "\n".join(
        [
            f"path: {OUTPUT_DIR.as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: rider_with_helmet",
            "  1: rider_without_helmet",
            "",
        ]
    )
    (OUTPUT_DIR / "data.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    records = build_records()
    split_idx = int(len(records) * TRAIN_RATIO)
    process(records[:split_idx], "train")
    process(records[split_idx:], "val")
    write_data_yaml()
    print("Dataset ready")


if __name__ == "__main__":
    main()
