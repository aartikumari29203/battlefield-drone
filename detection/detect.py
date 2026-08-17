from ultralytics import YOLO


def main():
    model = YOLO("yolo11n.pt")

    print("YOLO11 model loaded successfully.")


if __name__ == "__main__":
    main()