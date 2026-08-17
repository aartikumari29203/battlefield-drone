from ultralytics import YOLO
from pathlib import Path


def main():
    model = YOLO(
        "runs/detect/results/visdrone_train/weights/best.pt"
    )

    val_images = Path(
        "datasets/VisDrone/VisDrone-DET/val/images"
    )

    model.predict(
        source=str(val_images),
        conf=0.25,
        save=True,
        project="results",
        name="visdrone_predictions"
    )

    print("VisDrone final testing completed.")


if __name__ == "__main__":
    main()