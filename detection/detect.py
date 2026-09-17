from pathlib import Path

import cv2
from ultralytics import YOLO


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "kiit_mita_best.pt"
)

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
# CLASS COLORS
# =========================================================

CLASS_COLORS = {
    "artillary": (255, 165, 0),
    "missile": (0, 0, 255),
    "radar": (255, 0, 255),
    "m. rocket launcher": (0, 255, 255),
    "soldier": (0, 165, 255),
    "tank": (255, 0, 0),
    "vehicle": (0, 255, 0),
}


# =========================================================
# NORMALIZE CLASS NAME
# =========================================================

def normalize_class_name(class_name):
    return str(class_name).strip().lower()


# =========================================================
# GET IMAGE PATHS
# =========================================================

def get_image_paths():
    image_paths = []

    for extension in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(
            IMAGE_DIR.glob(extension)
        )

    return sorted(image_paths)


# =========================================================
# DRAW DETECTION
# =========================================================

def draw_detection(
    image,
    bbox,
    class_name,
    confidence,
):
    x1, y1, x2, y2 = map(int, bbox)

    normalized_class = normalize_class_name(
        class_name
    )

    color = CLASS_COLORS.get(
        normalized_class,
        (0, 255, 0),
    )

    # Bounding box
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        3,
    )

    # Label
    label = (
        f"{class_name} "
        f"{confidence:.2f}"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    (
        text_width,
        text_height,
    ), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness,
    )

    label_x = x1
    label_y = y1 - 8

    if label_y - text_height < 0:
        label_y = (
            y1
            + text_height
            + 8
        )

    # Label background
    cv2.rectangle(
        image,
        (
            label_x,
            label_y
            - text_height
            - 8,
        ),
        (
            label_x
            + text_width
            + 8,
            label_y
            + baseline,
        ),
        color,
        -1,
    )

    # Label text
    cv2.putText(
        image,
        label,
        (
            label_x + 4,
            label_y,
        ),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


# =========================================================
# RUN DETECTION
# =========================================================

def run_detection(model, image_path):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        print(
            f"⚠️ Could not read: {image_path.name}"
        )
        return None, []

    results = model(
        image,
        conf=0.25,
        verbose=False,
    )

    detections = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            confidence = float(
                box.conf[0]
            )

            class_id = int(
                box.cls[0]
            )

            class_name = model.names[
                class_id
            ]

            detections.append(
                {
                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2,
                    ],
                    "confidence": confidence,
                    "class": class_name,
                }
            )

    # Draw detections
    annotated = image.copy()

    for detection in detections:

        draw_detection(
            annotated,
            detection["bbox"],
            detection["class"],
            detection["confidence"],
        )

    return annotated, detections


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("KIIT-MiTA OBJECT DETECTION")
    print("=" * 60)

    # -----------------------------------------------------
    # CHECK MODEL
    # -----------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"KIIT-MiTA model not found:\n"
            f"{MODEL_PATH}"
        )

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    model = YOLO(
        str(MODEL_PATH)
    )

    print(
        "\n✅ KIIT-MiTA YOLO model loaded!"
    )

    print("\nClasses:")
    print(model.names)

    # -----------------------------------------------------
    # CHECK IMAGE DIRECTORY
    # -----------------------------------------------------

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found:\n"
            f"{IMAGE_DIR}"
        )

    image_paths = get_image_paths()

    if not image_paths:
        raise FileNotFoundError(
            f"No test images found in:\n"
            f"{IMAGE_DIR}"
        )

    print(
        f"\n📷 Found {len(image_paths)} test image(s)."
    )

    # -----------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"💾 Output directory:\n{OUTPUT_DIR}"
    )

    # -----------------------------------------------------
    # PROCESS IMAGES
    # -----------------------------------------------------

    total_detections = 0

    for image_number, image_path in enumerate(
        image_paths,
        start=1,
    ):

        print(
            f"\n[{image_number}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        annotated, detections = run_detection(
            model,
            image_path,
        )

        if annotated is None:
            continue

        # Save result
        output_path = (
            OUTPUT_DIR
            / image_path.name
        )

        cv2.imwrite(
            str(output_path),
            annotated,
        )

        print(
            f"   Detections: {len(detections)}"
        )

        for detection in detections:

            print(
                f"   - "
                f"{detection['class']} "
                f"({detection['confidence']:.2f})"
            )

        print(
            f"   ✅ Saved: {output_path}"
        )

        total_detections += len(
            detections
        )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("DETECTION COMPLETE")
    print("=" * 60)

    print(
        f"Images processed: {len(image_paths)}"
    )

    print(
        f"Total detections: {total_detections}"
    )

    print(
        f"Results saved to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()