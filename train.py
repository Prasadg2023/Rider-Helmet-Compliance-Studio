from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "data.yaml"
BASE_MODEL = ROOT / "yolov8n.pt"


model = YOLO(str(BASE_MODEL))

model.train(
    data=str(DATA_PATH),
    epochs=10,
    imgsz=416,
    batch=4,
    workers=0,
    patience=7,
    pretrained=True,
    device="cpu",
    cos_lr=True,
    close_mosaic=5,
    amp=False,
    name="helmet_model_cpu",
)
