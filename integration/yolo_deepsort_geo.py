import sys
from pathlib import Path
import csv


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# IMPORTS
# =========================================================

import cv2
from ultralytics import YOLO

from telemetry import get_simulated_telemetry
from tracking.tracker import ObjectTracker
from geolocation.geolocation import (
    bbox_center,
    calculate_target_location,
)


# =========================================================
# PATHS
# =========================================================

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
    / "yolo_deepsort_geo"
)

CSV_PATH = (
    OUTPUT_DIR
    / "tracking_results.csv"
)


# =========================================================
# SETTINGS
# =========================================================

CONFIDENCE = 0.25

SHOW_IMAGES = False

DISPLAY_DELAY = 300


# =========================================================
# CSV HEADER
# =========================================================

CSV_HEADER = [
    "frame",
    "image",
    "track_id",
    "class",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "drone_latitude",
    "drone_longitude",
    "altitude_m",
    "yaw_deg",
    "pitch_deg",
    "roll_deg",
    "target_latitude",
    "target_longitude",
]


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("KIIT-MiTA YOLO + DeepSORT + GEOLOCATION")
    print("=" * 70)

    # =====================================================
    # CHECK MODEL
    # =====================================================

    print("\nModel:")
    print(MODEL_PATH)

    if not MODEL_PATH.exists():
        print("\n❌ Model file not found.")
        print(f"Expected: {MODEL_PATH}")
        return

    # =====================================================
    # CHECK IMAGE DIRECTORY
    # =====================================================

    print("\nImage directory:")
    print(IMAGE_DIR)

    if not IMAGE_DIR.exists():
        print("\n❌ Image directory not found.")
        print(f"Expected: {IMAGE_DIR}")
        return

    # =====================================================
    # LOAD YOLO
    # =====================================================

    print("\nLoading YOLO model...")

    try:
        model = YOLO(str(MODEL_PATH))

    except Exception as error:
        print("\n❌ Failed to load YOLO model.")
        print(error)
        return

    print("✅ YOLO model loaded successfully.")

    print("\nModel classes:")

    for class_id, class_name in model.names.items():
        print(f"  {class_id}: {class_name}")

    # =====================================================
    # INITIALIZE DEEPSORT
    # =====================================================

    print("\nInitializing DeepSORT tracker...")

    try:
        tracker = ObjectTracker()

    except Exception as error:
        print("\n❌ Failed to initialize DeepSORT.")
        print(error)
        return

    print("✅ DeepSORT tracker initialized.")

    # =====================================================
    # CREATE OUTPUT DIRECTORY
    # =====================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    # =====================================================
    # TELEMETRY
    # =====================================================

    telemetry = get_simulated_telemetry()

    print("\nDrone telemetry:")

    print(
        f"Latitude  : "
        f"{telemetry.latitude:.6f}"
    )

    print(
        f"Longitude : "
        f"{telemetry.longitude:.6f}"
    )

    print(
        f"Altitude  : "
        f"{telemetry.altitude:.2f} m"
    )

    print(
        f"Yaw       : "
        f"{telemetry.yaw:.2f}°"
    )

    print(
        f"Pitch     : "
        f"{telemetry.pitch:.2f}°"
    )

    print(
        f"Roll      : "
        f"{telemetry.roll:.2f}°"
    )

    # =====================================================
    # FIND IMAGES
    # =====================================================

    image_paths = sorted(
        list(IMAGE_DIR.glob("*.jpg"))
        + list(IMAGE_DIR.glob("*.jpeg"))
        + list(IMAGE_DIR.glob("*.png"))
    )

    if not image_paths:

        print("\n❌ No images found.")
        print(f"Checked: {IMAGE_DIR}")
        return

    print(
        f"\nFound {len(image_paths)} images."
    )

    # =====================================================
    # OPEN CSV
    # =====================================================

    try:

        csv_file = open(
            CSV_PATH,
            "w",
            newline="",
            encoding="utf-8"
        )

        csv_writer = csv.writer(csv_file)

        csv_writer.writerow(
            CSV_HEADER
        )

    except Exception as error:

        print("\n❌ Could not create CSV file.")
        print(error)
        return

    # =====================================================
    # STATISTICS
    # =====================================================

    total_detections = 0
    total_tracks = 0

    images_processed = 0
    images_with_detections = 0
    images_with_tracks = 0

    class_counts = {}
    track_ids = set()

    # =====================================================
    # START PIPELINE
    # =====================================================

    print("\nStarting pipeline:")
    print(
        "YOLO → DeepSORT → Bounding Box Center"
        " → Geolocation"
    )

    print("-" * 70)

    # =====================================================
    # PROCESS IMAGES
    # =====================================================

    for frame_number, image_path in enumerate(
        image_paths,
        start=1
    ):

        # -------------------------------------------------
        # READ IMAGE
        # -------------------------------------------------

        frame = cv2.imread(
            str(image_path)
        )

        if frame is None:

            print(
                f"\n❌ Could not read image:"
                f" {image_path.name}"
            )

            continue

        images_processed += 1

        image_height, image_width = (
            frame.shape[:2]
        )

        # -------------------------------------------------
        # YOLO DETECTION
        # -------------------------------------------------

        results = model(
            frame,
            conf=CONFIDENCE,
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

                bbox = [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                ]

                detections.append(
                    (
                        bbox,
                        confidence,
                        class_name,
                    )
                )

                # Class statistics

                class_counts[
                    class_name
                ] = (
                    class_counts.get(
                        class_name,
                        0
                    )
                    + 1
                )

        detection_count = len(
            detections
        )

        total_detections += (
            detection_count
        )

        if detection_count > 0:
            images_with_detections += 1

        # -------------------------------------------------
        # DEEPSORT TRACKING
        # -------------------------------------------------

        tracks = tracker.update(
            detections,
            frame,
        )

        track_count = len(
            tracks
        )

        total_tracks += (
            track_count
        )

        if track_count > 0:
            images_with_tracks += 1

        # -------------------------------------------------
        # ANNOTATED FRAME
        # -------------------------------------------------

        display_frame = frame.copy()

        # -------------------------------------------------
        # PROCESS EACH TRACK
        # -------------------------------------------------

        for track in tracks:

            track_id = track["id"]

            class_name = track["class"]

            bbox = track["bbox"]

            track_ids.add(
                track_id
            )

            # ---------------------------------------------
            # BOUNDING BOX
            # ---------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                bbox
            )

            # ---------------------------------------------
            # BOUNDING BOX CENTER
            # ---------------------------------------------

            target_x, target_y = (
                bbox_center(
                    bbox
                )
            )

            # ---------------------------------------------
            # GEOLOCATION
            # ---------------------------------------------

            try:

                latitude, longitude = (
                    calculate_target_location(
                        drone_lat=(
                            telemetry.latitude
                        ),
                        drone_lon=(
                            telemetry.longitude
                        ),
                        altitude=(
                            telemetry.altitude
                        ),
                        image_width=(
                            image_width
                        ),
                        image_height=(
                            image_height
                        ),
                        target_x=target_x,
                        target_y=target_y,
                    )
                )

            except Exception as error:

                print(
                    f"\n⚠ Geolocation error "
                    f"for Track {track_id}: "
                    f"{error}"
                )

                latitude = None
                longitude = None

            # ---------------------------------------------
            # DRAW BOUNDING BOX
            # ---------------------------------------------

            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            # ---------------------------------------------
            # DRAW CENTER POINT
            # ---------------------------------------------

            center_x = int(
                target_x
            )

            center_y = int(
                target_y
            )

            cv2.circle(
                display_frame,
                (
                    center_x,
                    center_y
                ),
                5,
                (0, 0, 255),
                -1,
            )

            # ---------------------------------------------
            # TRACK LABEL
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
                    max(
                        y1 - 10,
                        20
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # ---------------------------------------------
            # GEOLOCATION LABEL
            # ---------------------------------------------

            if (
                latitude is not None
                and longitude is not None
            ):

                location_label = (
                    f"{latitude:.6f}, "
                    f"{longitude:.6f}"
                )

                cv2.putText(
                    display_frame,
                    location_label,
                    (
                        x1,
                        min(
                            y2 + 20,
                            image_height - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                )

            # ---------------------------------------------
            # SAVE CSV RECORD
            # ---------------------------------------------

            csv_writer.writerow(
                [
                    frame_number,
                    image_path.name,
                    track_id,
                    class_name,

                    f"{bbox[0]:.2f}",
                    f"{bbox[1]:.2f}",
                    f"{bbox[2]:.2f}",
                    f"{bbox[3]:.2f}",

                    f"{target_x:.2f}",
                    f"{target_y:.2f}",

                    f"{telemetry.latitude:.8f}",
                    f"{telemetry.longitude:.8f}",
                    f"{telemetry.altitude:.2f}",

                    f"{telemetry.yaw:.2f}",
                    f"{telemetry.pitch:.2f}",
                    f"{telemetry.roll:.2f}",

                    (
                        f"{latitude:.8f}"
                        if latitude is not None
                        else ""
                    ),

                    (
                        f"{longitude:.8f}"
                        if longitude is not None
                        else ""
                    ),
                ]
            )

            # ---------------------------------------------
            # CONSOLE OUTPUT
            # ---------------------------------------------

            if (
                latitude is not None
                and longitude is not None
            ):

                print(
                    f"    Track {track_id:<4} | "
                    f"{class_name:<20} | "
                    f"Center "
                    f"({target_x:7.1f}, "
                    f"{target_y:7.1f}) | "
                    f"Location "
                    f"{latitude:.6f}, "
                    f"{longitude:.6f}"
                )

            else:

                print(
                    f"    Track {track_id:<4} | "
                    f"{class_name:<20} | "
                    f"Center "
                    f"({target_x:7.1f}, "
                    f"{target_y:7.1f}) | "
                    f"Location unavailable"
                )

        # -------------------------------------------------
        # TELEMETRY OVERLAY
        # -------------------------------------------------

        telemetry_text = [

            (
                f"LAT: "
                f"{telemetry.latitude:.6f}"
            ),

            (
                f"LON: "
                f"{telemetry.longitude:.6f}"
            ),

            (
                f"ALT: "
                f"{telemetry.altitude:.1f} m"
            ),

            (
                f"YAW: "
                f"{telemetry.yaw:.1f} deg"
            ),

            (
                f"Frame: "
                f"{frame_number}"
            ),

            (
                f"Detections: "
                f"{detection_count}"
            ),

            (
                f"Tracks: "
                f"{track_count}"
            ),
        ]

        y_position = 30

        for text in telemetry_text:

            cv2.putText(
                display_frame,
                text,
                (
                    20,
                    y_position
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            y_position += 23

        # -------------------------------------------------
        # SAVE ANNOTATED IMAGE
        # -------------------------------------------------

        output_path = (
            OUTPUT_DIR
            / image_path.name
        )

        success = cv2.imwrite(
            str(output_path),
            display_frame
        )

        if not success:

            print(
                f"\n⚠ Could not save:"
                f" {output_path}"
            )

        # -------------------------------------------------
        # PROGRESS
        # -------------------------------------------------

        print(
            f"[{frame_number:03d}/"
            f"{len(image_paths):03d}] "
            f"{image_path.name:<35} "
            f"Detections: "
            f"{detection_count:2d} | "
            f"Tracks: "
            f"{track_count:2d}"
        )

        # -------------------------------------------------
        # OPTIONAL DISPLAY
        # -------------------------------------------------

        if SHOW_IMAGES:

            cv2.imshow(
                "KIIT-MiTA YOLO + DeepSORT + GEO",
                display_frame,
            )

            key = (
                cv2.waitKey(
                    DISPLAY_DELAY
                )
                & 0xFF
            )

            if key == ord("q"):

                print(
                    "\nQ pressed. "
                    "Stopping pipeline..."
                )

                break

    # =====================================================
    # CLEANUP
    # =====================================================

    csv_file.close()

    cv2.destroyAllWindows()

    # =====================================================
    # FINAL REPORT
    # =====================================================

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print(
        f"Images processed       : "
        f"{images_processed}"
    )

    print(
        f"Images with detections : "
        f"{images_with_detections}"
    )

    print(
        f"Images with tracks      : "
        f"{images_with_tracks}"
    )

    print(
        f"Total detections        : "
        f"{total_detections}"
    )

    print(
        f"Total track outputs     : "
        f"{total_tracks}"
    )

    print(
        f"Unique track IDs        : "
        f"{len(track_ids)}"
    )

    print(
        f"Annotated images        : "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Tracking CSV            : "
        f"{CSV_PATH}"
    )

    # =====================================================
    # CLASS STATISTICS
    # =====================================================

    print("\nDetections by class:")
    print("-" * 45)

    for class_name, count in sorted(
        class_counts.items()
    ):

        print(
            f"{class_name:<25}: "
            f"{count}"
        )

    print("=" * 70)

    print("\nPerson 2 pipeline:")

    print("✅ YOLO object detection")
    print("✅ DeepSORT multi-object tracking")
    print("✅ Bounding-box center extraction")
    print("✅ Drone telemetry integration")
    print("✅ Target coordinate estimation")
    print("✅ Annotated tracking output")
    print("✅ CSV tracking/geolocation output")

    print("=" * 70)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()