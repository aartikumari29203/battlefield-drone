from pathlib import Path
from collections import Counter

import cv2
from ultralytics import YOLO


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "kiit_mita_best.pt"

IMAGE_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "KIIT-MiTA"
    / "test"
    / "images"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "detections"
)


# =========================================================
# SETTINGS
# =========================================================

CONFIDENCE_THRESHOLD = 0.25


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("KIIT-MiTA DETECTION VERIFICATION")
    print("=" * 60)

    # -----------------------------------------------------
    # CHECK PATHS
    # -----------------------------------------------------

    if not MODEL_PATH.exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        return

    if not IMAGE_DIR.exists():
        print(f"❌ Input folder not found: {IMAGE_DIR}")
        return

    if not OUTPUT_DIR.exists():
        print(f"❌ Output folder not found: {OUTPUT_DIR}")
        return

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    print("\nLoading model...")

    model = YOLO(str(MODEL_PATH))

    print("✅ Model loaded")
    print(f"Classes: {model.names}")

    # -----------------------------------------------------
    # GET INPUT IMAGES
    # -----------------------------------------------------

    image_paths = []

    for extension in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(IMAGE_DIR.glob(extension))

    image_paths = sorted(image_paths)

    print(f"\nInput images : {len(image_paths)}")

    # -----------------------------------------------------
    # GET OUTPUT IMAGES
    # -----------------------------------------------------

    output_paths = []

    for extension in ("*.jpg", "*.jpeg", "*.png"):
        output_paths.extend(OUTPUT_DIR.glob(extension))

    output_paths = sorted(output_paths)

    print(f"Output images: {len(output_paths)}")

    # -----------------------------------------------------
    # BASIC FILE COUNT CHECK
    # -----------------------------------------------------

    if len(image_paths) != len(output_paths):
        print("\n⚠️ WARNING:")
        print("Input and output image counts do not match.")

    else:
        print("\n✅ All input images have corresponding output files.")

    # -----------------------------------------------------
    # RUN VERIFICATION
    # -----------------------------------------------------

    class_counts = Counter()

    confidence_values = []

    images_with_detections = 0
    images_without_detections = 0

    total_detections = 0

    low_confidence_detections = 0

    # -----------------------------------------------------
    # PROCESS IMAGES
    # -----------------------------------------------------

    print("\nRunning verification...\n")

    for index, image_path in enumerate(image_paths, start=1):

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"⚠️ Could not read: {image_path.name}")
            continue

        results = model(
            image,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        image_detection_count = 0

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                confidence = float(box.conf[0])

                class_id = int(box.cls[0])

                class_name = model.names[class_id]

                class_counts[class_name] += 1

                confidence_values.append(confidence)

                total_detections += 1

                image_detection_count += 1

                if confidence < 0.40:
                    low_confidence_detections += 1

        # -------------------------------------------------
        # IMAGE RESULT
        # -------------------------------------------------

        if image_detection_count > 0:
            images_with_detections += 1
        else:
            images_without_detections += 1

        # Progress
        if index % 10 == 0 or index == len(image_paths):
            print(
                f"Processed {index}/{len(image_paths)} images"
            )

    # =====================================================
    # RESULTS
    # =====================================================

    print("\n")
    print("=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)

    print(f"\nTotal input images       : {len(image_paths)}")
    print(f"Total output images      : {len(output_paths)}")

    print(
        f"Images with detections  : "
        f"{images_with_detections}"
    )

    print(
        f"Images without detections: "
        f"{images_without_detections}"
    )

    print(
        f"\nTotal detections        : "
        f"{total_detections}"
    )

    # -----------------------------------------------------
    # CLASS COUNTS
    # -----------------------------------------------------

    print("\nDetections by class:")
    print("-" * 40)

    for class_id, class_name in model.names.items():

        count = class_counts[class_name]

        print(
            f"{class_name:22s}: {count}"
        )

    # -----------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------

    if confidence_values:

        average_confidence = (
            sum(confidence_values)
            / len(confidence_values)
        )

        highest_confidence = max(
            confidence_values
        )

        lowest_confidence = min(
            confidence_values
        )

        print("\nConfidence statistics:")
        print("-" * 40)

        print(
            f"Average confidence : "
            f"{average_confidence:.3f}"
        )

        print(
            f"Highest confidence : "
            f"{highest_confidence:.3f}"
        )

        print(
            f"Lowest confidence  : "
            f"{lowest_confidence:.3f}"
        )

        print(
            f"Below 0.40         : "
            f"{low_confidence_detections}"
        )

    else:

        print("\n⚠️ No detections were produced.")

    # =====================================================
    # FINAL STATUS
    # =====================================================

    print("\n")
    print("=" * 60)

    if (
        len(image_paths) == len(output_paths)
        and total_detections > 0
    ):

        print("✅ DETECTION PIPELINE IS WORKING")

    elif total_detections == 0:

        print("⚠️ NO DETECTIONS FOUND")

    else:

        print("⚠️ CHECK OUTPUT FILES")

    print("=" * 60)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()