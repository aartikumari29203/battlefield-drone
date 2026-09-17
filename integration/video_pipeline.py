import sys
from pathlib import Path

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from ultralytics import YOLO

from telemetry import get_simulated_telemetry
from tracking.tracker import ObjectTracker
from geolocation.geolocation import (
    bbox_center,
    calculate_target_location,
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
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
    / "tracking"
)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():

    print("=" * 60)
    print("KIIT-MiTA YOLO + DeepSORT + GEOLOCATION")
    print("IMAGE SEQUENCE PIPELINE")
    print("=" * 60)

    # -----------------------------------------------------
    # Check paths
    # -----------------------------------------------------
    if not MODEL_PATH.exists():
        print(f"❌ Model not found:")
        print(MODEL_PATH)
        return

    if not IMAGE_DIR.exists():
        print(f"❌ Image directory not found:")
        print(IMAGE_DIR)
        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Load YOLO
    # -----------------------------------------------------
    print("\nLoading YOLO model...")

    model = YOLO(str(MODEL_PATH))

    print("✅ YOLO model loaded.")
    print(f"Classes: {model.names}")

    # -----------------------------------------------------
    # Initialize DeepSORT
    # -----------------------------------------------------
    tracker = ObjectTracker()

    print("✅ DeepSORT tracker initialized.")

    # -----------------------------------------------------
    # Telemetry
    # -----------------------------------------------------
    telemetry = get_simulated_telemetry()

    print("\nDrone telemetry:")
    print(f"Latitude  : {telemetry.latitude}")
    print(f"Longitude : {telemetry.longitude}")
    print(f"Altitude  : {telemetry.altitude} m")
    print(f"Yaw       : {telemetry.yaw}°")
    print(f"Pitch     : {telemetry.pitch}°")
    print(f"Roll      : {telemetry.roll}°")

    # -----------------------------------------------------
    # Find images
    # -----------------------------------------------------
    image_paths = sorted(
        [
            p
            for p in IMAGE_DIR.iterdir()
            if p.suffix.lower()
            in [".jpg", ".jpeg", ".png"]
        ]
    )

    if not image_paths:
        print("\n❌ No images found.")
        print(IMAGE_DIR)
        return

    print("\nInput directory:")
    print(IMAGE_DIR)

    print(f"\nImages found: {len(image_paths)}")

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    # -----------------------------------------------------
    # Counters
    # -----------------------------------------------------
    total_detections = 0
    total_tracks = 0
    processed_images = 0

    # -----------------------------------------------------
    # Process images
    # -----------------------------------------------------
    for image_number, image_path in enumerate(
        image_paths,
        start=1,
    ):

        print(
            f"\nImage {image_number}/{len(image_paths)}: "
            f"{image_path.name}"
        )

        # -------------------------------------------------
        # Read image
        # -------------------------------------------------
        frame = cv2.imread(str(image_path))

        if frame is None:
            print("❌ Could not read image.")
            continue

        image_height, image_width = frame.shape[:2]

        # -------------------------------------------------
        # YOLO detection
        # -------------------------------------------------
        results = model(
            frame,
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
                    (
                        [x1, y1, x2, y2],
                        confidence,
                        class_name,
                    )
                )

        total_detections += len(detections)

        print(
            f"YOLO detections: {len(detections)}"
        )

        # -------------------------------------------------
        # DeepSORT tracking
        # -------------------------------------------------
        tracks = tracker.update(
            detections,
            frame,
        )

        total_tracks += len(tracks)

        print(
            f"Confirmed tracks: {len(tracks)}"
        )

        # -------------------------------------------------
        # Draw results
        # -------------------------------------------------
        display_frame = frame.copy()

        for track in tracks:

            track_id = track["id"]
            class_name = track["class"]
            bbox = track["bbox"]

            x1, y1, x2, y2 = map(
                int,
                bbox,
            )

            # ---------------------------------------------
            # Bounding-box center
            # ---------------------------------------------
            center_x, center_y = bbox_center(
                bbox
            )

            center_x_int = int(center_x)
            center_y_int = int(center_y)

            # ---------------------------------------------
            # Geolocation
            # ---------------------------------------------
            latitude, longitude = (
                calculate_target_location(
                    drone_lat=telemetry.latitude,
                    drone_lon=telemetry.longitude,
                    altitude=telemetry.altitude,
                    image_width=image_width,
                    image_height=image_height,
                    target_x=center_x,
                    target_y=center_y,
                )
            )

            # ---------------------------------------------
            # Bounding box
            # ---------------------------------------------
            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            # ---------------------------------------------
            # Label
            # ---------------------------------------------
            label = (
                f"ID {track_id} | "
                f"{class_name}"
            )

            cv2.putText(
                display_frame,
                label,
                (
                    x1,
                    max(y1 - 10, 20),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # ---------------------------------------------
            # Center point
            # ---------------------------------------------
            cv2.circle(
                display_frame,
                (
                    center_x_int,
                    center_y_int,
                ),
                5,
                (0, 0, 255),
                -1,
            )

            # ---------------------------------------------
            # Geolocation text
            # ---------------------------------------------
            geo_text = (
                f"{latitude:.6f}, "
                f"{longitude:.6f}"
            )

            cv2.putText(
                display_frame,
                geo_text,
                (
                    x1,
                    min(y2 + 20, image_height - 10),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

            # ---------------------------------------------
            # Console output
            # ---------------------------------------------
            print(
                f"  Track {track_id} | "
                f"{class_name} | "
                f"Location: "
                f"{latitude:.6f}, "
                f"{longitude:.6f}"
            )

        # -------------------------------------------------
        # Telemetry overlay
        # -------------------------------------------------
        telemetry_text = [
            f"LAT: {telemetry.latitude:.6f}",
            f"LON: {telemetry.longitude:.6f}",
            f"ALT: {telemetry.altitude:.1f} m",
            f"YAW: {telemetry.yaw:.1f}",
        ]

        y_position = 30

        for text in telemetry_text:

            cv2.putText(
                display_frame,
                text,
                (
                    20,
                    y_position,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            y_position += 25

        # -------------------------------------------------
        # Image number
        # -------------------------------------------------
        cv2.putText(
            display_frame,
            f"Image: {image_number}/{len(image_paths)}",
            (
                20,
                image_height - 20,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # -------------------------------------------------
        # Save annotated image
        # -------------------------------------------------
        output_path = (
            OUTPUT_DIR
            / image_path.name
        )

        cv2.imwrite(
            str(output_path),
            display_frame,
        )

        processed_images += 1

        # -------------------------------------------------
        # Progress
        # -------------------------------------------------
        if image_number % 10 == 0:
            print(
                f"Progress: "
                f"{image_number}/{len(image_paths)}"
            )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    print(
        f"Images processed   : "
        f"{processed_images}"
    )

    print(
        f"Total detections   : "
        f"{total_detections}"
    )

    print(
        f"Total track outputs : "
        f"{total_tracks}"
    )

    print(
        f"Annotated images    : "
        f"{OUTPUT_DIR}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()