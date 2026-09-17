from deep_sort_realtime.deepsort_tracker import DeepSort


class ObjectTracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=30,
            n_init=1,
            nms_max_overlap=1.0,
        )

    def update(self, detections, frame):
        """
        Update DeepSORT with YOLO detections.

        detections format:
        [
            (
                [x1, y1, x2, y2],
                confidence,
                class_name,
            ),
            ...
        ]

        Returns:
        [
            {
                "id": track_id,
                "bbox": [x1, y1, x2, y2],
                "class": class_name,
            },
            ...
        ]
        """

        # Convert YOLO detections into the format
        # expected by deep-sort-realtime.
        deepsort_detections = []

        for bbox, confidence, class_name in detections:
            x1, y1, x2, y2 = bbox

            width = x2 - x1
            height = y2 - y1

            deepsort_detections.append(
                (
                    [x1, y1, width, height],
                    confidence,
                    class_name,
                )
            )

        tracks = self.tracker.update_tracks(
            deepsort_detections,
            frame=frame,
        )

        results = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            bbox = track.to_ltrb()

            # DeepSORT stores the detection class
            # as the track's detection class.
            class_name = track.get_det_class()

            if class_name is None:
                class_name = "object"

            results.append(
                {
                    "id": track.track_id,
                    "bbox": bbox,
                    "class": class_name,
                }
            )

        return results