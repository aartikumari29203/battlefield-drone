from pathlib import Path

import cv2
from ultralytics import YOLO

from unlearning.unlearning import ModelUnlearner
from tracking.tracker import ObjectTracker

from geolocation.geolocation import (
    bbox_center,
    calculate_target_location,
)

from telemetry import get_simulated_telemetry


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


# =========================================================
# KIIT-MiTA CLASSES
# =========================================================

CLASS_NAMES = {
    0: "Artillary",
    1: "Missile",
    2: "Radar",
    3: "M. Rocket Launcher",
    4: "Soldier",
    5: "Tank",
    6: "Vehicle",
}


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
# GET IMAGE PATHS
# =========================================================

def get_image_paths():
    """
    Return all KIIT-MiTA test images.

    Supports:
        .jpg
        .jpeg
        .png
    """

    image_paths = []

    for extension in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(
            IMAGE_DIR.glob(extension)
        )

    return sorted(image_paths)


# =========================================================
# NORMALIZE CLASS
# =========================================================

def normalize_class_name(class_name):
    """
    Normalize class names for comparisons.

    Example:
        'Tank' -> 'tank'
        ' tank ' -> 'tank'
    """

    return str(
        class_name
    ).strip().lower()


# =========================================================
# DRAW DETECTION
# =========================================================

def draw_detection(
    image,
    bbox,
    class_name,
    confidence,
):
    """
    Draw a YOLO detection on an image.
    """

    x1, y1, x2, y2 = map(
        int,
        bbox,
    )

    normalized_class = normalize_class_name(
        class_name
    )

    color = CLASS_COLORS.get(
        normalized_class,
        (0, 255, 0),
    )

    # -----------------------------------------------------
    # BOUNDING BOX
    # -----------------------------------------------------

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        3,
    )

    # -----------------------------------------------------
    # LABEL
    # -----------------------------------------------------

    label = (
        f"{class_name} "
        f"{confidence:.2f}"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
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

    # -----------------------------------------------------
    # LABEL BACKGROUND
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # LABEL TEXT
    # -----------------------------------------------------

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
# LOAD MODEL
# =========================================================

def load_model():
    """
    Load the KIIT-MiTA YOLO model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"KIIT-MiTA model not found: "
            f"{MODEL_PATH}"
        )

    return YOLO(
        str(MODEL_PATH)
    )


# =========================================================
# DETECTION
# =========================================================

def run_detection(
    image_path,
    forgotten_class=None,
):
    """
    Run KIIT-MiTA YOLO detection.

    If forgotten_class is supplied,
    detections belonging to that class
    are removed from the displayed result.
    """

    model = load_model()

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    # -----------------------------------------------------
    # YOLO
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # MACHINE UNLEARNING FILTER
    # -----------------------------------------------------

    if forgotten_class:

        unlearner = ModelUnlearner()

        unlearner.forget(
            forgotten_class
        )

        detections = (
            unlearner.filter_predictions(
                detections
            )
        )

    # -----------------------------------------------------
    # DRAW
    # -----------------------------------------------------

    annotated = image.copy()

    for detection in detections:

        draw_detection(
            annotated,
            detection["bbox"],
            detection["class"],
            detection["confidence"],
        )

    return {
        "image": annotated,
        "detections": detections,
    }


# =========================================================
# TRACKING
# =========================================================

def run_tracking(image_paths):
    """
    Run YOLO + DeepSORT tracking.

    Each returned track contains:

        id
        bbox
        class
    """

    model = load_model()

    tracker = ObjectTracker()

    final_image = None
    final_tracks = []
    final_frame = 0

    # -----------------------------------------------------
    # PROCESS EVERY IMAGE
    # -----------------------------------------------------

    for frame_number, image_path in enumerate(
        image_paths,
        start=1,
    ):

        frame = cv2.imread(
            str(image_path)
        )

        if frame is None:
            continue

        # -------------------------------------------------
        # YOLO DETECTION
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
                        [
                            x1,
                            y1,
                            x2,
                            y2,
                        ],
                        confidence,
                        class_name,
                    )
                )

        # -------------------------------------------------
        # DEEPSORT
        # -------------------------------------------------

        tracks = tracker.update(
            detections,
            frame,
        )

        annotated = frame.copy()

        # -------------------------------------------------
        # DRAW TRACKS
        # -------------------------------------------------

        for track in tracks:

            track_id = track["id"]

            bbox = track["bbox"]

            class_name = track.get(
                "class",
                "object",
            )

            x1, y1, x2, y2 = map(
                int,
                bbox,
            )

            normalized_class = (
                normalize_class_name(
                    class_name
                )
            )

            object_color = (
                CLASS_COLORS.get(
                    normalized_class,
                    (0, 255, 0),
                )
            )

            # -------------------------------------------------
            # OBJECT BOX
            # -------------------------------------------------

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                object_color,
                3,
            )

            # -------------------------------------------------
            # TRACK LABEL
            # -------------------------------------------------

            track_label = (
                f"ID: {track_id} | "
                f"{class_name}"
            )

            font = (
                cv2.FONT_HERSHEY_SIMPLEX
            )

            font_scale = 0.70
            label_thickness = 2

            (
                text_width,
                text_height,
            ), baseline = (
                cv2.getTextSize(
                    track_label,
                    font,
                    font_scale,
                    label_thickness,
                )
            )

            label_x = x1
            label_y = y1 - 10

            if (
                label_y - text_height
                < 0
            ):
                label_y = (
                    y1
                    + text_height
                    + 10
                )

            # -------------------------------------------------
            # LABEL BACKGROUND
            # -------------------------------------------------

            cv2.rectangle(
                annotated,
                (
                    label_x,
                    label_y
                    - text_height
                    - 8,
                ),
                (
                    label_x
                    + text_width
                    + 10,
                    label_y
                    + baseline,
                ),
                (30, 30, 30),
                -1,
            )

            # -------------------------------------------------
            # LABEL TEXT
            # -------------------------------------------------

            cv2.putText(
                annotated,
                track_label,
                (
                    label_x + 5,
                    label_y,
                ),
                font,
                font_scale,
                (255, 255, 255),
                label_thickness,
                cv2.LINE_AA,
            )

        final_image = annotated
        final_tracks = tracks
        final_frame = frame_number

    return {
        "image": final_image,
        "tracks": final_tracks,
        "frame": final_frame,
    }


# =========================================================
# GEOLOCATION
# =========================================================

def run_geolocation(
    image_path,
    forgotten_class=None,
):
    """
    Run KIIT-MiTA detection followed by
    simplified target geolocation.
    """

    telemetry = (
        get_simulated_telemetry()
    )

    model = load_model()

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    # -----------------------------------------------------
    # YOLO DETECTION
    # -----------------------------------------------------

    results = model(
        image,
        conf=0.25,
        verbose=False,
    )

    image_height, image_width = (
        image.shape[:2]
    )

    detections = []

    # -----------------------------------------------------
    # COLLECT DETECTIONS
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # MACHINE UNLEARNING FILTER
    # -----------------------------------------------------

    if forgotten_class:

        unlearner = ModelUnlearner()

        unlearner.forget(
            forgotten_class
        )

        detections = (
            unlearner.filter_predictions(
                detections
            )
        )

    # -----------------------------------------------------
    # GEOLOCATION
    # -----------------------------------------------------

    locations = []

    for detection in detections:

        bbox = detection[
            "bbox"
        ]

        class_name = detection[
            "class"
        ]

        confidence = detection[
            "confidence"
        ]

        center_x, center_y = (
            bbox_center(
                bbox
            )
        )

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
                image_width=image_width,
                image_height=image_height,
                target_x=center_x,
                target_y=center_y,
            )
        )

        locations.append(
            {
                "class": class_name,

                "confidence": round(
                    confidence,
                    3,
                ),

                "latitude": round(
                    latitude,
                    6,
                ),

                "longitude": round(
                    longitude,
                    6,
                ),
            }
        )

    # -----------------------------------------------------
    # DRAW GEOLOCATION IMAGE
    # -----------------------------------------------------

    annotated = image.copy()

    for detection in detections:

        draw_detection(
            annotated,
            detection["bbox"],
            detection["class"],
            detection["confidence"],
        )

    # -----------------------------------------------------
    # RETURN RESULT
    # -----------------------------------------------------

    return {
        "telemetry": telemetry,
        "locations": locations,
        "image": annotated,
    }