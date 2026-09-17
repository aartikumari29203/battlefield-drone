from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "kiit_mita_best.pt"

DATA_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "KIIT-MiTA"
    / "KIIT-MiTA.yml"
)


def main():

    print("=" * 60)
    print("KIIT-MiTA MODEL VALIDATION")
    print("=" * 60)

    print()
    print(f"Model : {MODEL_PATH}")
    print(f"Data  : {DATA_PATH}")
    print()

    if not MODEL_PATH.exists():
        print("❌ Model not found.")
        return

    if not DATA_PATH.exists():
        print("❌ KIIT-MiTA.yml not found.")
        return

    print("Loading model...")

    model = YOLO(str(MODEL_PATH))

    print("✅ Model loaded successfully!")

    print()
    print("Model classes:")
    print(model.names)

    print()
    print("Running validation...")
    print()

    results = model.val(
        data=str(DATA_PATH),
        split="val",
        verbose=True,
    )

    print()
    print("=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)

    print()
    print(f"mAP50    : {results.box.map50:.4f}")
    print(f"mAP50-95 : {results.box.map:.4f}")

    print()
    print("Per-class results:")

    for class_id, class_name in model.names.items():

        if class_id < len(results.box.ap50):

            print(
                f"{class_name:20s} "
                f"AP50: {results.box.ap50[class_id]:.4f}"
            )


if __name__ == "__main__":
    main()