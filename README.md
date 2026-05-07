# Rider Helmet Compliance Detection

A CPU-friendly YOLOv8 project for detecting rider helmet compliance in traffic-style images.

## What This Project Detects

This model is trained for:

- `rider_with_helmet`
- `rider_without_helmet`

It is intended for rider or head scenes, not standalone helmet product photos.

## Project Files

- `app.py`: Streamlit dashboard for image, sample, and webcam-based prediction
- `train.py`: YOLOv8 training script tuned for CPU use
- `prepare_dataset.py`: Converts the source CSV labels into YOLO dataset format
- `evaluate_model.py`: Validates the latest trained model and saves sample predictions
- `dataset/data.yaml`: Dataset config for YOLO training and validation

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Prepare Dataset

Make sure these exist first:

- `train_labels.csv`
- `JPEGImages/`

Then run:

```powershell
python prepare_dataset.py
```

## Train

```powershell
python train.py
```

The best model weights will be saved under `runs/detect/`.

## Evaluate

```powershell
python evaluate_model.py
```

## Run The App

```powershell
streamlit run app.py
```

## Notes

- This repository should not usually include `venv/`, `runs/`, or large dataset folders.
- If you want to share trained weights, upload the chosen `.pt` file separately or through a release.

## Live demo : http://rider-helmet-compliance-studio-65qxnuvlngunjs3c7m7tgv.streamlit.app/

## Author
**Prasad Karade**
### Eamil : karadeprasad023@gmail.com
### GitHub : https://github.com/Prasadg2023
### LinkedIn : https://www.linkedin.com/in/prasad-karade-641799399/

