import numpy as np

from deep_sort_realtime.deepsort_tracker import DeepSort


class ObjectTracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=30,
            n_init=2,
            nms_max_overlap=1.0
        )

    def update(self, detections, frame):
        tracks = self.tracker.update_tracks(
            detections,
            frame=frame
        )

        results = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            results.append({
                "id": track.track_id,
                "bbox": track.to_ltrb()
            })

        return results


def main():
    tracker = ObjectTracker()

    # Dummy video frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Simulate the same object moving across several frames
    positions = [
        [100, 100, 200, 200],
        [110, 100, 210, 200],
        [120, 100, 220, 200],
        [130, 100, 230, 200],
        [140, 100, 240, 200],
    ]

    for frame_number, bbox in enumerate(positions, start=1):

        detections = [
            (
                bbox,
                0.90,
                "object"
            )
        ]

        tracks = tracker.update(detections, frame)

        print(f"Frame {frame_number}: {tracks}")


if __name__ == "__main__":
    main()