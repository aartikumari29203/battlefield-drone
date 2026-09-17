from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "kiit_mita_best.pt"


def main():
    if not MODEL_PATH.exists():
        print("❌ KIIT-MiTA model not found:")
        print(MODEL_PATH)
        return

    model = YOLO(str(MODEL_PATH))

    print("✅ KIIT-MiTA YOLO model loaded!")
    print("Classes:")
    print(model.names)


if __name__ == "__main__":
    main()