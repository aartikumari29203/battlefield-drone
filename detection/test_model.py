from ultralytics import YOLO
from pathlib import Path


def main():
    model = YOLO(
        "runs/detect/results/kiit_test-2/weights/best.pt"
    )

    test_images = Path(
        "datasets/KIIT-MiTA/KIIT-MiTA/test/images"
    )

    model.predict(
        source=str(test_images),
        conf=0.25,
        save=True,
        project="results",
        name="kiit_predictions"
    )

    print("KIIT-MiTA testing completed.")
    print("Predictions saved to results/kiit_predictions")


if __name__ == "__main__":
    main()