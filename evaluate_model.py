from pathlib import Path
import random

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "data.yaml"
RUNS_DIR = ROOT / "runs" / "detect"
VAL_IMAGE_DIR = ROOT / "dataset" / "images" / "val"
OUTPUT_DIR = ROOT / "runs" / "detect" / "inspection"


def latest_model_path() -> Path:
    candidates = sorted(RUNS_DIR.glob("helmet_model*/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError("No trained best.pt model found in runs/detect.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> None:
    model_path = latest_model_path()
    model = YOLO(str(model_path))

    print(f"Using model: {model_path}")
    print("Task: rider helmet compliance detection")
    print("Classes: 0=rider_with_helmet, 1=rider_without_helmet")
    metrics = model.val(data=str(DATA_PATH), imgsz=416, batch=4, workers=0, device="cpu")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    val_images = sorted([path for path in VAL_IMAGE_DIR.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    sample_images = random.Random(42).sample(val_images, k=min(5, len(val_images)))
    results = model.predict([str(path) for path in sample_images], conf=0.4, save=True, project=str(OUTPUT_DIR), name="samples")

    print("\nSaved sample predictions:")
    for path, result in zip(sample_images, results):
        detections = 0 if result.boxes is None else len(result.boxes)
        print(f"- {path.name}: {detections} detections")


if __name__ == "__main__":
    main()
