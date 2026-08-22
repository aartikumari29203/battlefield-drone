from pathlib import Path

import cv2
from ultralytics import YOLO

from telemetry import get_simulated_telemetry
from tracking.tracker import ObjectTracker
from geolocation.geolocation import (
    bbox_center,
    calculate_target_location,
)


MODEL_PATH = "models/visdrone_best.pt"

IMAGE_DIR = Path(
    "datasets/VisDrone/VisDrone-DET/val/"
    "VisDrone2019-DET-val/images"
)


def main():
    # Load YOLO model
    model = YOLO(MODEL_PATH)

    # Load DeepSORT tracker
    tracker = ObjectTracker()

    # Get simulated drone telemetry
    # Real telemetry is not currently available.
    telemetry = get_simulated_telemetry()

    drone_lat = telemetry.latitude
    drone_lon = telemetry.longitude
    altitude = telemetry.altitude

    print("Drone telemetry:")
    print(f"Latitude  : {drone_lat}")
    print(f"Longitude : {drone_lon}")
    print(f"Altitude  : {altitude}")
    print(f"Yaw       : {telemetry.yaw}")
    print(f"Pitch     : {telemetry.pitch}")
    print(f"Roll      : {telemetry.roll}")

    # Select first 5 VisDrone validation images
    image_paths = sorted(
        IMAGE_DIR.glob("*.jpg")
    )[:5]

    if not image_paths:
        print("No VisDrone images found.")
        return

    for frame_number, image_path in enumerate(
        image_paths,
        start=1
    ):

        print(
            f"\n========== FRAME {frame_number} =========="
        )

        print(
            f"Image: {image_path.name}"
        )

        frame = cv2.imread(str(image_path))

        if frame is None:
            print("Could not load image.")
            continue

        # -------------------------
        # YOLO detection
        # -------------------------

        results = model(
            frame,
            conf=0.25,
            verbose=False,
        )

        detections = []

        for result in results:

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
                    x1,
                    y1,
                    x2,
                    y2,
                ]

                detections.append(
                    (
                        bbox,
                        confidence,
                        class_name,
                    )
                )

        print(
            f"YOLO detections: "
            f"{len(detections)}"
        )

        # -------------------------
        # DeepSORT tracking
        # -------------------------

        tracks = tracker.update(
            detections,
            frame,
        )

        print(
            f"Confirmed tracks: "
            f"{len(tracks)}"
        )

        # -------------------------
        # Geolocation
        # -------------------------

        for track in tracks:

            track_id = track["id"]

            tracked_bbox = track["bbox"]

            target_x, target_y = bbox_center(
                tracked_bbox
            )

            image_height, image_width = (
                frame.shape[:2]
            )

            latitude, longitude = (
                calculate_target_location(
                    drone_lat=drone_lat,
                    drone_lon=drone_lon,
                    altitude=altitude,
                    image_width=image_width,
                    image_height=image_height,
                    target_x=target_x,
                    target_y=target_y,
                )
            )

            print(
                f"Track ID: {track_id}"
            )

            print(
                f"Bounding box: "
                f"{tracked_bbox}"
            )

            print(
                f"Estimated location: "
                f"{latitude:.6f}, "
                f"{longitude:.6f}"
            )


if __name__ == "__main__":
    main()