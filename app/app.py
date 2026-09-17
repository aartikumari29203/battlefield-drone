import contextlib
import io
import sys
from pathlib import Path

import streamlit as st
from ultralytics import YOLO


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

# app.py is assumed to be inside:
# Battlefield-Drone/app/app.py
#
# Therefore:
# parent     = app
# parent.parent = Battlefield-Drone

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "sisa_unlearning_fixed"
)

MODELS_DIR = BASE_OUTPUT / "models"
SHARDS_DIR = BASE_OUTPUT / "shards"

UNLEARNED_MODELS_DIR = (
    BASE_OUTPUT / "unlearned_models"
)

DATASET_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "KIIT-MiTA"
)

VALIDATION_YAML = (
    DATASET_DIR
    / "KIIT-MiTA.yml"
)

NUM_SHARDS = 3

# Final demonstration training configuration
EPOCHS = 2

# YOLO class names
CLASS_NAMES = [
    "Artillary",
    "Missile",
    "Radar",
    "M. Rocket Launcher",
    "Soldier",
    "Tank",
    "Vehicle",
]


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Battlefield Drone",
    page_icon="🚁",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "selected_class": None,
    "class_id": None,
    "affected_shards": [],
    "instances": 0,
    "before_metrics": None,
    "after_metrics": None,
    "deletion_done": False,
    "retraining_done": False,
    "deletion_mode": None,
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# MODEL PATH
# ============================================================

def get_model_path(shard_number):

    return (
        MODELS_DIR
        / f"shard_{shard_number}"
        / "weights"
        / "best.pt"
    )


# ============================================================
# LOAD SISA MODELS
# ============================================================

@st.cache_resource
def load_sisa_models():

    models = {}

    for shard_number in range(
        1,
        NUM_SHARDS + 1
    ):

        model_path = get_model_path(
            shard_number
        )

        if not model_path.exists():
            continue

        try:

            models[shard_number] = YOLO(
                str(model_path)
            )

        except Exception:
            continue

    return models


# ============================================================
# FIND AFFECTED SHARDS
# ============================================================

def find_affected_shards(class_id):

    affected = []

    for shard_number in range(
        1,
        NUM_SHARDS + 1
    ):

        labels_dir = (
            SHARDS_DIR
            / f"shard_{shard_number}"
            / "labels"
        )

        if not labels_dir.exists():
            continue

        found = False

        for label_file in labels_dir.glob(
            "*.txt"
        ):

            try:

                with open(
                    label_file,
                    "r",
                    encoding="utf-8"
                ) as file:

                    for line in file:

                        parts = (
                            line.strip().split()
                        )

                        if not parts:
                            continue

                        try:

                            detected_class = int(
                                float(parts[0])
                            )

                        except Exception:
                            continue

                        if (
                            detected_class
                            == class_id
                        ):

                            found = True
                            break

                if found:
                    break

            except Exception:
                continue

        if found:
            affected.append(
                shard_number
            )

    return affected


# ============================================================
# COUNT CLASS INSTANCES
# ============================================================

def count_class_instances(class_id):

    total = 0

    for label_file in SHARDS_DIR.glob(
        "shard_*/labels/*.txt"
    ):

        try:

            with open(
                label_file,
                "r",
                encoding="utf-8"
            ) as file:

                for line in file:

                    parts = (
                        line.strip().split()
                    )

                    if not parts:
                        continue

                    try:

                        detected_class = int(
                            float(parts[0])
                        )

                    except Exception:
                        continue

                    if (
                        detected_class
                        == class_id
                    ):

                        total += 1

        except Exception:
            continue

    return total


# ============================================================
# DELETE SENSITIVE DATA
# ============================================================

def delete_sensitive_data(
    shard_number,
    class_id
):

    shard_dir = (
        SHARDS_DIR
        / f"shard_{shard_number}"
    )

    images_dir = (
        shard_dir / "images"
    )

    labels_dir = (
        shard_dir / "labels"
    )

    if not labels_dir.exists():
        return 0, 0

    deleted_images = 0
    deleted_annotations = 0

    image_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    ]

    label_files = list(
        labels_dir.glob("*.txt")
    )

    for label_file in label_files:

        try:

            with open(
                label_file,
                "r",
                encoding="utf-8"
            ) as file:

                lines = file.readlines()

            sensitive_found = False

            for line in lines:

                parts = (
                    line.strip().split()
                )

                if not parts:
                    continue

                try:

                    detected_class = int(
                        float(parts[0])
                    )

                except Exception:
                    continue

                if (
                    detected_class
                    == class_id
                ):

                    sensitive_found = True
                    break

            if not sensitive_found:
                continue

            # ------------------------------------------------
            # Delete image containing sensitive class.
            # ------------------------------------------------

            image_deleted = False

            if images_dir.exists():

                for extension in image_extensions:

                    image_file = (
                        images_dir
                        / f"{label_file.stem}{extension}"
                    )

                    if image_file.exists():

                        try:

                            image_file.unlink()

                            deleted_images += 1
                            image_deleted = True

                        except Exception:
                            pass

                        break

            # ------------------------------------------------
            # Delete annotation.
            # ------------------------------------------------

            try:

                label_file.unlink()

                deleted_annotations += 1

            except Exception:
                pass

        except Exception:
            continue

    return (
        deleted_images,
        deleted_annotations
    )


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(model):

    if not VALIDATION_YAML.exists():
        return None

    try:

        buffer = io.StringIO()

        with contextlib.redirect_stdout(
            buffer
        ):

            metrics = model.val(
                data=str(
                    VALIDATION_YAML
                ),
                verbose=False,
                plots=False,
            )

        return {
            "precision":
                float(metrics.box.mp),

            "recall":
                float(metrics.box.mr),

            "map50":
                float(metrics.box.map50),

            "map5095":
                float(metrics.box.map),
        }

    except Exception:
        return None


# ============================================================
# EVALUATE ALL MODELS
# ============================================================

def evaluate_all_models(models):

    results = {}

    for shard_number, model in (
        models.items()
    ):

        results[shard_number] = (
            evaluate_model(model)
        )

    return results


# ============================================================
# AGGREGATE METRICS
# ============================================================

def aggregate_metrics(results):

    valid_results = [
        value
        for value in results.values()
        if value is not None
    ]

    if not valid_results:
        return None

    return {

        "precision":
            sum(
                x["precision"]
                for x in valid_results
            )
            / len(valid_results),

        "recall":
            sum(
                x["recall"]
                for x in valid_results
            )
            / len(valid_results),

        "map50":
            sum(
                x["map50"]
                for x in valid_results
            )
            / len(valid_results),

        "map5095":
            sum(
                x["map5095"]
                for x in valid_results
            )
            / len(valid_results),
    }


# ============================================================
# RETRAIN AFFECTED SHARD
# ============================================================

def retrain_affected_shard(
    shard_number
):

    shard_dir = (
        SHARDS_DIR
        / f"shard_{shard_number}"
    )

    yaml_file = (
        shard_dir / "data.yaml"
    )

    original_model = (
        get_model_path(
            shard_number
        )
    )

    if not yaml_file.exists():
        return None

    if not original_model.exists():
        return None

    try:

        model = YOLO(
            str(original_model)
        )

        output_dir = (
            UNLEARNED_MODELS_DIR
            / f"shard_{shard_number}"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        model.train(

            data=str(yaml_file),

            epochs=EPOCHS,

            project=str(
                output_dir
            ),

            name="unlearned",

            exist_ok=True,

            verbose=False,
        )

        trained_model = (
            output_dir
            / "unlearned"
            / "weights"
            / "best.pt"
        )

        if trained_model.exists():

            return YOLO(
                str(trained_model)
            )

    except Exception:
        return None

    return None


# ============================================================
# RESET WORKFLOW
# ============================================================

def reset_workflow():

    st.session_state.selected_class = None
    st.session_state.class_id = None
    st.session_state.affected_shards = []
    st.session_state.instances = 0
    st.session_state.before_metrics = None
    st.session_state.after_metrics = None
    st.session_state.deletion_done = False
    st.session_state.retraining_done = False
    st.session_state.deletion_mode = None


# ============================================================
# HEADER
# ============================================================

st.title(
    "🚁 Battlefield Drone"
)

st.caption(
    "YOLO Object Detection + "
    "SISA Machine Unlearning"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Navigation"
)

module = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "YOLO Detection",
        "SISA Machine Unlearning",
    ],
)


# ============================================================
# DASHBOARD
# ============================================================

if module == "Dashboard":

    st.header(
        "Battlefield Drone Dashboard"
    )

    st.write(
        "The system combines YOLO-based "
        "battlefield object detection with "
        "SISA-based selective machine unlearning."
    )

    st.divider()

    models = load_sisa_models()

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.metric(
            "Dataset",
            "KIIT-MiTA"
        )

    with col2:

        st.metric(
            "Object Classes",
            "7"
        )

    with col3:

        st.metric(
            "SISA Shards",
            f"{len(models)}/{NUM_SHARDS}"
        )

    with col4:

        st.metric(
            "Training Epochs",
            EPOCHS
        )

    st.divider()

    st.subheader(
        "System Workflow"
    )

    st.code(
        """
KIIT-MiTA Dataset
        ↓
SISA Sharding
        ↓
3 Independent Shards
        ↓
3 YOLO Models
        ↓
Sensitive Class Selection
        ↓
Affected-Shard Identification
        ↓
Sensitive Data Deletion
        ↓
Selective Retraining
        ↓
Before / After Evaluation
        ↓
Performance Comparison
        """,
        language="text"
    )

    st.info(
        "The original KIIT-MiTA dataset is "
        "preserved. Deletion is performed on "
        "the SISA working shard data."
    )


# ============================================================
# YOLO DETECTION
# ============================================================

elif module == "YOLO Detection":

    st.header(
        "🎯 YOLO Object Detection"
    )

    models = load_sisa_models()

    if not models:

        st.error(
            "No SISA models were found."
        )

        st.write(
            "Expected:"
        )

        st.code(
            str(MODELS_DIR)
        )

        st.stop()

    # --------------------------------------------------------
    # Select shard
    # --------------------------------------------------------

    shard_number = st.selectbox(
        "Select SISA shard",
        sorted(models.keys())
    )

    model = models[
        shard_number
    ]

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    image_directories = [

        DATASET_DIR
        / "test"
        / "images",

        DATASET_DIR
        / "valid"
        / "images",

        DATASET_DIR
        / "train"
        / "images",
    ]

    image_paths = []

    for directory in (
        image_directories
    ):

        if directory.exists():

            image_paths.extend(
                sorted(
                    [
                        path
                        for path
                        in directory.iterdir()
                        if path.is_file()
                        and path.suffix.lower()
                        in [
                            ".jpg",
                            ".jpeg",
                            ".png",
                            ".bmp",
                            ".webp",
                        ]
                    ]
                )
            )

    if not image_paths:

        st.error(
            "No KIIT-MiTA images found."
        )

        st.stop()

    image_names = [
        path.name
        for path in image_paths
    ]

    selected_name = st.selectbox(
        "Select image",
        image_names
    )

    selected_path = (
        image_paths[
            image_names.index(
                selected_name
            )
        ]
    )

    if st.button(
        "Run YOLO Detection",
        type="primary"
    ):

        with st.spinner(
            f"Running Shard {shard_number} YOLO..."
        ):

            try:

                buffer = io.StringIO()

                with contextlib.redirect_stdout(
                    buffer
                ):

                    results = model.predict(
                        source=str(
                            selected_path
                        ),
                        verbose=False,
                    )

                result = results[0]

                annotated = (
                    result.plot()
                )

                st.success(
                    f"Detection completed using "
                    f"Shard {shard_number}."
                )

                st.image(
                    annotated,
                    channels="BGR",
                    use_container_width=True,
                )

                detections = []

                if result.boxes is not None:

                    for box in result.boxes:

                        class_id = int(
                            box.cls[0]
                        )

                        confidence = float(
                            box.conf[0]
                        )

                        if (
                            0
                            <= class_id
                            < len(CLASS_NAMES)
                        ):

                            detections.append(
                                {
                                    "Class":
                                        CLASS_NAMES[
                                            class_id
                                        ],

                                    "Confidence":
                                        round(
                                            confidence,
                                            3
                                        ),
                                }
                            )

                if detections:

                    st.subheader(
                        "Detected Objects"
                    )

                    st.dataframe(
                        detections,
                        use_container_width=True,
                        hide_index=True,
                    )

                else:

                    st.info(
                        "No objects detected."
                    )

            except Exception as e:

                st.error(
                    f"Detection failed: {e}"
                )


# ============================================================
# SISA MACHINE UNLEARNING
# ============================================================

elif module == "SISA Machine Unlearning":

    st.header(
        "🧠 SISA Machine Unlearning"
    )

    st.write(
        "Select sensitive information, identify "
        "affected shards, delete the corresponding "
        "working data, selectively retrain the "
        "affected models, and compare performance."
    )

    # --------------------------------------------------------
    # MODEL STATUS
    # --------------------------------------------------------

    st.subheader(
        "SISA Model Status"
    )

    models = load_sisa_models()

    status_columns = st.columns(
        NUM_SHARDS
    )

    for shard_number in range(
        1,
        NUM_SHARDS + 1
    ):

        with status_columns[
            shard_number - 1
        ]:

            if shard_number in models:

                st.success(
                    f"Shard {shard_number}\n\n"
                    "MODEL LOADED"
                )

            else:

                st.error(
                    f"Shard {shard_number}\n\n"
                    "MODEL MISSING"
                )

    if len(models) != NUM_SHARDS:

        st.warning(
            f"Expected {NUM_SHARDS} shard models, "
            f"but only {len(models)} were found."
        )

        st.stop()

    st.divider()

    # ========================================================
    # STEP 1
    # ========================================================

    st.subheader(
        "1. Select Sensitive Class"
    )

    selected_class = st.selectbox(
        "Class to forget",
        CLASS_NAMES,
    )

    class_id = (
        CLASS_NAMES.index(
            selected_class
        )
    )

    if st.button(
        "Identify Affected Shards",
        type="primary"
    ):

        affected = (
            find_affected_shards(
                class_id
            )
        )

        instances = (
            count_class_instances(
                class_id
            )
        )

        st.session_state.selected_class = (
            selected_class
        )

        st.session_state.class_id = (
            class_id
        )

        st.session_state.affected_shards = (
            affected
        )

        st.session_state.instances = (
            instances
        )

        st.session_state.before_metrics = None
        st.session_state.after_metrics = None
        st.session_state.deletion_done = False
        st.session_state.retraining_done = False

    # --------------------------------------------------------
    # Display selection
    # --------------------------------------------------------

    if st.session_state.selected_class:

        st.success(
            f"Sensitive class: "
            f"**{st.session_state.selected_class}**"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Class ID",
                st.session_state.class_id
            )

        with col2:

            st.metric(
                "Sensitive Instances",
                st.session_state.instances
            )

        if (
            not st.session_state.affected_shards
        ):

            st.warning(
                "No shard contains the selected "
                "sensitive class."
            )

            st.stop()

    # ========================================================
    # STEP 2 — SHARD ANALYSIS
    # ========================================================

    if st.session_state.affected_shards:

        st.divider()

        st.subheader(
            "SISA Shard Analysis"
        )

        shard_table = []

        for shard_number in range(
            1,
            NUM_SHARDS + 1
        ):

            if (
                shard_number
                in st.session_state.affected_shards
            ):

                shard_table.append(
                    {
                        "Shard":
                            f"Shard {shard_number}",

                        "Status":
                            "AFFECTED",

                        "Action":
                            "Delete + Retrain",
                    }
                )

            else:

                shard_table.append(
                    {
                        "Shard":
                            f"Shard {shard_number}",

                        "Status":
                            "UNAFFECTED",

                        "Action":
                            "Preserve",
                    }
                )

        st.dataframe(
            shard_table,
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # STEP 3 — BEFORE METRICS
    # ========================================================

    if st.session_state.affected_shards:

        st.divider()

        st.subheader(
            "Before Unlearning"
        )

        st.write(
            "Evaluate the original SISA models "
            "before deleting the selected data."
        )

        if st.button(
            "Calculate Before Metrics"
        ):

            with st.spinner(
                "Evaluating original models..."
            ):

                results = (
                    evaluate_all_models(
                        models
                    )
                )

                metrics = (
                    aggregate_metrics(
                        results
                    )
                )

            st.session_state.before_metrics = (
                metrics
            )

        if st.session_state.before_metrics:

            metrics = (
                st.session_state.before_metrics
            )

            col1, col2, col3, col4 = (
                st.columns(4)
            )

            col1.metric(
                "Precision",
                f"{metrics['precision']:.3f}"
            )

            col2.metric(
                "Recall",
                f"{metrics['recall']:.3f}"
            )

            col3.metric(
                "mAP50",
                f"{metrics['map50']:.3f}"
            )

            col4.metric(
                "mAP50-95",
                f"{metrics['map5095']:.3f}"
            )

    # ========================================================
    # STEP 4 — DELETION
    # ========================================================

    if st.session_state.affected_shards:

        st.divider()

        st.subheader(
            "2. Sensitive Data Deletion"
        )

        deletion_mode = st.radio(
            "Choose deletion method",
            [
                "Automatic Sensitive-Data Deletion",
                "Voluntary Data Deletion",
            ],
        )

        # ----------------------------------------------------
        # AUTOMATIC DELETION
        # ----------------------------------------------------

        if deletion_mode == (
            "Automatic Sensitive-Data Deletion"
        ):

            st.info(
                f"The system will automatically delete "
                f"images and annotations containing "
                f"**{st.session_state.selected_class}** "
                f"from the affected SISA shards."
            )

            if st.button(
                "Run Automatic Deletion"
            ):

                total_images = 0
                total_annotations = 0

                progress = st.progress(
                    0
                )

                affected = (
                    st.session_state.affected_shards
                )

                for index, shard_number in enumerate(
                    affected
                ):

                    deleted_images, deleted_annotations = (
                        delete_sensitive_data(
                            shard_number,
                            st.session_state.class_id
                        )
                    )

                    total_images += (
                        deleted_images
                    )

                    total_annotations += (
                        deleted_annotations
                    )

                    progress.progress(
                        (index + 1)
                        / len(affected)
                    )

                st.session_state.deletion_done = True
                st.session_state.deletion_mode = (
                    "Automatic"
                )

                st.success(
                    "Automatic sensitive-data "
                    "deletion completed."
                )

                col1, col2 = (
                    st.columns(2)
                )

                col1.metric(
                    "Images Deleted",
                    total_images
                )

                col2.metric(
                    "Annotations Deleted",
                    total_annotations
                )

                st.info(
                    "Original KIIT-MiTA dataset "
                    "was preserved."
                )

        # ----------------------------------------------------
        # VOLUNTARY DELETION
        # ----------------------------------------------------

        else:

            st.warning(
                f"You selected "
                f"**{st.session_state.selected_class}** "
                f"for voluntary deletion."
            )

            confirmation = st.checkbox(
                "I confirm that I want to delete "
                "this sensitive data from the "
                "SISA working shards."
            )

            if st.button(
                "Run Voluntary Deletion"
            ):

                if not confirmation:

                    st.warning(
                        "Please confirm the deletion."
                    )

                else:

                    total_images = 0
                    total_annotations = 0

                    for shard_number in (
                        st.session_state.affected_shards
                    ):

                        deleted_images, deleted_annotations = (
                            delete_sensitive_data(
                                shard_number,
                                st.session_state.class_id
                            )
                        )

                        total_images += (
                            deleted_images
                        )

                        total_annotations += (
                            deleted_annotations
                        )

                    st.session_state.deletion_done = True
                    st.session_state.deletion_mode = (
                        "Voluntary"
                    )

                    st.success(
                        "Voluntary data deletion completed."
                    )

                    col1, col2 = (
                        st.columns(2)
                    )

                    col1.metric(
                        "Images Deleted",
                        total_images
                    )

                    col2.metric(
                        "Annotations Deleted",
                        total_annotations
                    )

                    st.info(
                        "Original KIIT-MiTA dataset "
                        "was preserved."
                    )

    # ========================================================
    # STEP 5 — SELECTIVE RETRAINING
    # ========================================================

    if (
        st.session_state.deletion_done
        and st.session_state.affected_shards
    ):

        st.divider()

        st.subheader(
            "3. Selective SISA Retraining"
        )

        st.write(
            "Only affected shards are retrained. "
            "Unaffected shards remain preserved."
        )

        st.info(
            f"Training configuration: "
            f"**{EPOCHS} epochs**"
        )

        if st.button(
            "Start Selective Retraining",
            type="primary"
        ):

            after_models = {}

            affected = (
                st.session_state.affected_shards
            )

            # ------------------------------------------------
            # Preserve unaffected models
            # ------------------------------------------------

            for shard_number, model in (
                models.items()
            ):

                if (
                    shard_number
                    not in affected
                ):

                    after_models[
                        shard_number
                    ] = model

            # ------------------------------------------------
            # Retrain affected models
            # ------------------------------------------------

            progress = st.progress(
                0
            )

            successful = 0

            for index, shard_number in enumerate(
                affected
            ):

                with st.spinner(
                    f"Retraining Shard "
                    f"{shard_number}..."
                ):

                    retrained_model = (
                        retrain_affected_shard(
                            shard_number
                        )
                    )

                if retrained_model is not None:

                    after_models[
                        shard_number
                    ] = retrained_model

                    successful += 1

                else:

                    # Fallback to original model
                    after_models[
                        shard_number
                    ] = models[
                        shard_number
                    ]

                progress.progress(
                    (index + 1)
                    / len(affected)
                )

            st.session_state.retraining_done = True

            st.success(
                f"Selective retraining completed: "
                f"{successful}/{len(affected)} "
                f"affected shards."
            )

            # ------------------------------------------------
            # AFTER METRICS
            # ------------------------------------------------

            with st.spinner(
                "Calculating AFTER metrics..."
            ):

                after_results = (
                    evaluate_all_models(
                        after_models
                    )
                )

                after_metrics = (
                    aggregate_metrics(
                        after_results
                    )
                )

            st.session_state.after_metrics = (
                after_metrics
            )

    # ========================================================
    # STEP 6 — BEFORE VS AFTER
    # ========================================================

    if (
        st.session_state.before_metrics
        and st.session_state.after_metrics
    ):

        st.divider()

        st.subheader(
            "4. Before vs After Performance"
        )

        before = (
            st.session_state.before_metrics
        )

        after = (
            st.session_state.after_metrics
        )

        rows = []

        for key, label in [

            (
                "precision",
                "Precision"
            ),

            (
                "recall",
                "Recall"
            ),

            (
                "map50",
                "mAP50"
            ),

            (
                "map5095",
                "mAP50-95"
            ),

        ]:

            old = before[key]
            new = after[key]
            change = new - old

            rows.append(
                {
                    "Metric":
                        label,

                    "Before":
                        round(
                            old,
                            3
                        ),

                    "After":
                        round(
                            new,
                            3
                        ),

                    "Change":
                        round(
                            change,
                            3
                        ),
                }
            )

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        st.subheader(
            "Performance Summary"
        )

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        col1.metric(
            "Precision",
            f"{after['precision']:.3f}",
            f"{after['precision'] - before['precision']:+.3f}"
        )

        col2.metric(
            "Recall",
            f"{after['recall']:.3f}",
            f"{after['recall'] - before['recall']:+.3f}"
        )

        col3.metric(
            "mAP50",
            f"{after['map50']:.3f}",
            f"{after['map50'] - before['map50']:+.3f}"
        )

        col4.metric(
            "mAP50-95",
            f"{after['map5095']:.3f}",
            f"{after['map5095'] - before['map5095']:+.3f}"
        )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    if st.session_state.retraining_done:

        st.divider()

        st.success(
            "✅ SISA unlearning workflow completed."
        )

        st.write(
            f"**Sensitive class:** "
            f"{st.session_state.selected_class}"
        )

        st.write(
            f"**Affected shards:** "
            f"{st.session_state.affected_shards}"
        )

        st.write(
            f"**Deletion mode:** "
            f"{st.session_state.deletion_mode}"
        )

        st.write(
            "✓ Sensitive working data deleted"
        )

        st.write(
            "✓ Affected shards selectively retrained"
        )

        st.write(
            "✓ Unaffected shards preserved"
        )

        st.write(
            "✓ Before/after metrics calculated"
        )

        st.info(
            "Original KIIT-MiTA data remains preserved."
        )

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    st.divider()

    if st.button(
        "Reset Unlearning Workflow"
    ):

        reset_workflow()

        st.rerun()