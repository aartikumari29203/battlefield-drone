from pathlib import Path

import cv2
from ultralytics import YOLO

from tracking.tracker import ObjectTracker
from geolocation.geolocation import (
    bbox_center,
    calculate_target_location,
)


MODEL_PATH = "models/visdrone_best.pt"
VIDEO_PATH = "integration/visdrone_test.mp4"


def main():
    model = YOLO(MODEL_PATH)
    tracker = ObjectTracker()

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Could not open video.")
        return

    frame_number = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

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

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                class_name = model.names[class_id]

                detections.append(
                    (
                        [x1, y1, x2, y2],
                        confidence,
                        class_name,
                    )
                )

        # -------------------------
        # DeepSORT
        # -------------------------
        tracks = tracker.update(
            detections,
            frame,
        )

        print(
            f"\nFrame {frame_number}: "
            f"{len(detections)} detections, "
            f"{len(tracks)} confirmed tracks"
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

            # Prototype drone information
            drone_lat = 12.9716
            drone_lon = 77.5946
            altitude = 100.0

            image_height, image_width = frame.shape[:2]

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
                f"  Track {track_id}: "
                f"location = "
                f"{latitude:.6f}, "
                f"{longitude:.6f}"
            )

    cap.release()

    print("\nVideo processing completed.")


if __name__ == "__main__":
    main()