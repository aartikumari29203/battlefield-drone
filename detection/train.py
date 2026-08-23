from ultralytics import YOLO


def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data="datasets/VisDrone/VisDrone-DET/data.yaml",
        epochs=30,
        imgsz=640,
        batch=4,
        project="results",
        name="visdrone_train"
    )

    print("VisDrone training completed.")


if __name__ == "__main__":
    main()