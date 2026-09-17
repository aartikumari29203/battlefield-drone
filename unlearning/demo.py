from pathlib import Path
import contextlib
import io

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

BASE_OUTPUT = ROOT / "outputs" / "sisa_unlearning_fixed"

MODELS_DIR = BASE_OUTPUT / "models"
SHARDS_DIR = BASE_OUTPUT / "shards"

DATASET_DIR = ROOT / "datasets" / "KIIT-MiTA"

NUM_SHARDS = 3

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
# FAST SISA DEMONSTRATION SETTING
# ============================================================
# Only affected shards are retrained.
# 2 epochs keeps the demonstration fast.

EPOCHS = 2

# Faster training settings
BATCH_SIZE = 16
IMAGE_SIZE = 640
WORKERS = 2

# Validation dataset used for before/after comparison.
VALIDATION_YAML = DATASET_DIR / "KIIT-MiTA.yml"


# ============================================================
# TERMINAL FORMATTING
# ============================================================

WIDTH = 72


def header(title):

    print()
    print("╔" + "═" * WIDTH + "╗")
    print("║" + title.center(WIDTH) + "║")
    print("╚" + "═" * WIDTH + "╝")


def section(title):

    print()
    print("┌" + "─" * WIDTH + "┐")
    print("│" + title.center(WIDTH) + "│")
    print("└" + "─" * WIDTH + "┘")


def separator():

    print("  " + "─" * (WIDTH - 4))


# ============================================================
# MODEL LOADING
# ============================================================

def load_models():

    models = {}

    section("LOADING SISA SHARD MODELS")

    for shard in range(1, NUM_SHARDS + 1):

        path = (
            MODELS_DIR
            / f"shard_{shard}"
            / "weights"
            / "best.pt"
        )

        if not path.exists():

            print(
                f"  Shard {shard:<5} : NOT FOUND"
            )

            continue

        try:

            models[shard] = YOLO(str(path))

            print(
                f"  Shard {shard:<5} : LOADED"
            )

        except Exception as e:

            print(
                f"  Shard {shard:<5} : FAILED"
            )

            print(
                f"  Reason: {e}"
            )

    separator()

    print(
        f"  Models available : "
        f"{len(models)}/{NUM_SHARDS}"
    )

    return models


# ============================================================
# CLASS SELECTION
# ============================================================

def select_class():

    section("SELECT SENSITIVE CLASS")

    print()

    for i, name in enumerate(
        CLASS_NAMES,
        start=1
    ):

        print(
            f"  {i}. {name}"
        )

    print()
    print("  0. Cancel")
    print()

    while True:

        choice = input(
            "  Enter class number: "
        ).strip()

        try:

            value = int(choice)

        except ValueError:

            print(
                "  Please enter a number."
            )

            continue

        if value == 0:

            return None

        if 1 <= value <= len(CLASS_NAMES):

            return value - 1

        print(
            "  Invalid selection."
        )


# ============================================================
# FIND AFFECTED SHARDS
# ============================================================

def find_affected_shards(class_id):

    affected = []

    for shard in range(
        1,
        NUM_SHARDS + 1
    ):

        labels_dir = (
            SHARDS_DIR
            / f"shard_{shard}"
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
                ) as f:

                    for line in f:

                        parts = (
                            line.strip().split()
                        )

                        if not parts:
                            continue

                        if int(
                            float(parts[0])
                        ) == class_id:

                            found = True
                            break

                if found:
                    break

            except Exception:

                continue

        if found:

            affected.append(shard)

    return affected


# ============================================================
# COUNT SENSITIVE INSTANCES
# ============================================================

def count_instances(class_id):

    total = 0

    for label_file in SHARDS_DIR.glob(
        "shard_*/labels/*.txt"
    ):

        try:

            with open(
                label_file,
                "r",
                encoding="utf-8"
            ) as f:

                for line in f:

                    parts = (
                        line.strip().split()
                    )

                    if not parts:
                        continue

                    if int(
                        float(parts[0])
                    ) == class_id:

                        total += 1

        except Exception:

            pass

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

    images_dir = shard_dir / "images"
    labels_dir = shard_dir / "labels"

    if not labels_dir.exists():

        return 0, 0

    deleted_images = 0
    deleted_labels = 0

    image_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    ]

    for label_file in list(
        labels_dir.glob("*.txt")
    ):

        try:

            with open(
                label_file,
                "r",
                encoding="utf-8"
            ) as f:

                lines = f.readlines()

            sensitive_found = False
            remaining = []

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

                    remaining.append(line)
                    continue

                if detected_class == class_id:

                    sensitive_found = True

                else:

                    remaining.append(line)

            if not sensitive_found:

                continue

            stem = label_file.stem

            # ------------------------------------------------
            # Remove image containing sensitive information.
            # ------------------------------------------------

            if images_dir.exists():

                for ext in image_extensions:

                    image_file = (
                        images_dir
                        / (stem + ext)
                    )

                    if image_file.exists():

                        try:

                            image_file.unlink()

                            deleted_images += 1

                        except Exception as e:

                            print(
                                f"  Could not delete "
                                f"{image_file.name}: {e}"
                            )

                        break

            # ------------------------------------------------
            # Remove corresponding annotation.
            # ------------------------------------------------

            try:

                label_file.unlink()

                deleted_labels += 1

            except Exception:

                pass

        except Exception:

            continue

    return deleted_images, deleted_labels


# ============================================================
# AUTOMATIC SENSITIVE DATA DELETION
# ============================================================

def automatic_sensitive_deletion(
    class_id,
    affected_shards
):

    class_name = CLASS_NAMES[class_id]

    section(
        "2. AUTOMATIC SENSITIVE DATA DELETION"
    )

    print()

    print(
        f"  Sensitive class : {class_name}"
    )

    print(
        f"  Class ID        : {class_id}"
    )

    print()
    print(
        "  Automatic trigger:"
    )

    print(
        "  Sensitive class detected in SISA data."
    )

    print()

    total_images = 0
    total_labels = 0

    for shard in affected_shards:

        deleted_images, deleted_labels = (
            delete_sensitive_data(
                shard,
                class_id
            )
        )

        total_images += deleted_images
        total_labels += deleted_labels

        print(
            f"  Shard {shard}: "
            f"{deleted_images} images deleted, "
            f"{deleted_labels} annotations deleted"
        )

    separator()

    print(
        f"  Total images deleted : "
        f"{total_images}"
    )

    print(
        f"  Total annotations    : "
        f"{total_labels}"
    )

    print()
    print(
        "  ✓ Original KIIT-MiTA dataset preserved"
    )

    print(
        "  ✓ SISA working data modified"
    )


# ============================================================
# VOLUNTARY DELETION
# ============================================================

def voluntary_deletion(
    class_id,
    affected_shards
):

    class_name = CLASS_NAMES[class_id]

    section(
        "3. VOLUNTARY DATA DELETION"
    )

    print()

    print(
        f"  Requested deletion : "
        f"{class_name}"
    )

    print(
        f"  Affected shards    : "
        f"{affected_shards}"
    )

    print()

    confirmation = input(
        "  Confirm voluntary deletion? [y/N]: "
    ).strip().lower()

    if confirmation != "y":

        print()
        print(
            "  Voluntary deletion cancelled."
        )

        return False

    total_images = 0
    total_labels = 0

    for shard in affected_shards:

        deleted_images, deleted_labels = (
            delete_sensitive_data(
                shard,
                class_id
            )
        )

        total_images += deleted_images
        total_labels += deleted_labels

    print()

    print(
        f"  Images deleted     : "
        f"{total_images}"
    )

    print(
        f"  Annotations deleted: "
        f"{total_labels}"
    )

    print()
    print(
        "  ✓ Voluntary deletion completed."
    )

    return True


# ============================================================
# VALIDATION
# ============================================================

def evaluate_model(model):

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
                plots=False
            )

        precision = float(
            metrics.box.mp
        )

        recall = float(
            metrics.box.mr
        )

        map50 = float(
            metrics.box.map50
        )

        map5095 = float(
            metrics.box.map
        )

        return {
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map5095": map5095,
        }

    except Exception as e:

        print(
            f"Validation failed: {e}"
        )

        return None


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(models):

    results = {}

    for shard, model in models.items():

        print(
            f"  Evaluating Shard {shard}..."
        )

        results[shard] = evaluate_model(
            model
        )

    return results


# ============================================================
# AGGREGATE METRICS
# ============================================================

def aggregate_metrics(results):

    valid = [
        value
        for value in results.values()
        if value is not None
    ]

    if not valid:

        return None

    return {

        "precision":
            sum(
                x["precision"]
                for x in valid
            ) / len(valid),

        "recall":
            sum(
                x["recall"]
                for x in valid
            ) / len(valid),

        "map50":
            sum(
                x["map50"]
                for x in valid
            ) / len(valid),

        "map5095":
            sum(
                x["map5095"]
                for x in valid
            ) / len(valid),
    }


# ============================================================
# RETRAIN AFFECTED SHARD
# ============================================================

def retrain_affected_shard(
    shard_number,
    class_id
):

    shard_dir = (
        SHARDS_DIR
        / f"shard_{shard_number}"
    )

    yaml_file = (
        shard_dir
        / "data.yaml"
    )

    original_model = (
        MODELS_DIR
        / f"shard_{shard_number}"
        / "weights"
        / "best.pt"
    )

    if not yaml_file.exists():

        print(
            f"  Shard {shard_number}: "
            f"data.yaml not found."
        )

        return None

    if not original_model.exists():

        print(
            f"  Shard {shard_number}: "
            f"model not found."
        )

        return None

    try:

        model = YOLO(
            str(original_model)
        )

        output_dir = (
            BASE_OUTPUT
            / "unlearned_models"
            / f"shard_{shard_number}"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print()
        print(
            f"  Retraining Shard {shard_number}"
        )

        print(
            f"  Epochs : {EPOCHS}"
        )

        print(
            f"  Batch  : {BATCH_SIZE}"
        )

        print(
            f"  Image  : {IMAGE_SIZE}"
        )

        print()

        model.train(

            data=str(
                yaml_file
            ),

            epochs=EPOCHS,

            project=str(
                output_dir
            ),

            name="unlearned",

            exist_ok=True,

            verbose=True,

            # ----------------------------------------------
            # Faster demonstration settings
            # ----------------------------------------------

            batch=BATCH_SIZE,

            imgsz=IMAGE_SIZE,

            workers=WORKERS,

            cache=False,
        )

        trained_model = (
            output_dir
            / "unlearned"
            / "weights"
            / "best.pt"
        )

        if trained_model.exists():

            print()
            print(
                f"  ✓ Shard {shard_number} "
                f"retraining completed."
            )

            return YOLO(
                str(trained_model)
            )

    except Exception as e:

        print()
        print(
            f"  Shard {shard_number} "
            f"retraining failed:"
        )

        print(
            f"  {e}"
        )

    return None


# ============================================================
# CREATE AFTER MODELS
# ============================================================

def create_after_models(
    models,
    affected_shards,
    class_id
):

    after_models = {}

    # --------------------------------------------------------
    # Preserve unaffected shards.
    # --------------------------------------------------------

    for shard, model in models.items():

        if shard not in affected_shards:

            after_models[shard] = model

            print(
                f"  Shard {shard}: "
                f"PRESERVED"
            )

    # --------------------------------------------------------
    # Retrain only affected shards.
    # --------------------------------------------------------

    for shard in affected_shards:

        retrained = (
            retrain_affected_shard(
                shard,
                class_id
            )
        )

        if retrained is not None:

            after_models[shard] = (
                retrained
            )

        else:

            print(
                f"  Shard {shard}: "
                f"original model retained."
            )

            after_models[shard] = (
                models[shard]
            )

    return after_models


# ============================================================
# BEFORE VS AFTER COMPARISON
# ============================================================

def display_comparison(
    before,
    after
):

    section(
        "1. BEFORE VS AFTER METRICS"
    )

    print()

    if before is None or after is None:

        print(
            "  Metrics unavailable."
        )

        return

    print(
        f"  {'Metric':<18}"
        f"{'Before':>12}"
        f"{'After':>12}"
        f"{'Change':>12}"
    )

    separator()

    for metric, label in [

        ("precision", "Precision"),

        ("recall", "Recall"),

        ("map50", "mAP50"),

        ("map5095", "mAP50-95"),

    ]:

        old = before[metric]
        new = after[metric]

        change = new - old

        print(
            f"  {label:<18}"
            f"{old:>12.3f}"
            f"{new:>12.3f}"
            f"{change:>+12.3f}"
        )

    print()

    print(
        "  Before = original SISA shard models"
    )

    print(
        "  After  = models after affected-shard retraining"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    header(
        "BATTLEFIELD DRONE"
    )

    print(
        " " * 18
        + "SISA UNLEARNING DEMONSTRATION"
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    models = load_models()

    if len(models) != NUM_SHARDS:

        print()

        print(
            f"ERROR: Expected "
            f"{NUM_SHARDS} shard models."
        )

        print(
            f"Found: "
            f"{len(models)}/{NUM_SHARDS}"
        )

        return

    # --------------------------------------------------------
    # Choose deletion mode
    # --------------------------------------------------------

    section(
        "DELETION MODE"
    )

    print()

    print(
        "  1. Automatic sensitive-data deletion"
    )

    print(
        "  2. Voluntary deletion"
    )

    print(
        "  0. Exit"
    )

    print()

    mode = input(
        "  Select option: "
    ).strip()

    if mode == "0":

        return

    if mode not in ("1", "2"):

        print()
        print(
            "  Invalid option."
        )

        return

    # --------------------------------------------------------
    # Select sensitive class
    # --------------------------------------------------------

    class_id = select_class()

    if class_id is None:

        print()

        print(
            "  No deletion requested."
        )

        return

    class_name = (
        CLASS_NAMES[class_id]
    )

    # --------------------------------------------------------
    # Find affected shards
    # --------------------------------------------------------

    affected_shards = (
        find_affected_shards(
            class_id
        )
    )

    instances = (
        count_instances(
            class_id
        )
    )

    print()

    print(
        f"  Sensitive class : "
        f"{class_name}"
    )

    print(
        f"  Class ID        : "
        f"{class_id}"
    )

    print(
        f"  Instances       : "
        f"{instances}"
    )

    print(
        f"  Affected shards : "
        f"{affected_shards}"
    )

    if not affected_shards:

        print()

        print(
            "  No SISA shard contains "
            "this class."
        )

        return

    # --------------------------------------------------------
    # BEFORE METRICS
    # --------------------------------------------------------

    section(
        "CALCULATING BEFORE METRICS"
    )

    print()

    before_results = (
        calculate_metrics(
            models
        )
    )

    before_metrics = (
        aggregate_metrics(
            before_results
        )
    )

    # --------------------------------------------------------
    # DELETE DATA
    # --------------------------------------------------------

    if mode == "1":

        automatic_sensitive_deletion(
            class_id,
            affected_shards
        )

    else:

        if not voluntary_deletion(
            class_id,
            affected_shards
        ):

            return

    # --------------------------------------------------------
    # RETRAIN ONLY AFFECTED SHARDS
    # --------------------------------------------------------

    section(
        "SISA SELECTIVE RETRAINING"
    )

    print()

    print(
        f"  Total shards      : "
        f"{NUM_SHARDS}"
    )

    print(
        f"  Affected shards   : "
        f"{affected_shards}"
    )

    print(
        f"  Epochs            : "
        f"{EPOCHS}"
    )

    print()

    print(
        "  ✓ Unaffected shards will NOT be retrained."
    )

    print(
        "  ✓ Only affected shards will be retrained."
    )

    print()

    after_models = (
        create_after_models(
            models,
            affected_shards,
            class_id
        )
    )

    # --------------------------------------------------------
    # AFTER METRICS
    # --------------------------------------------------------

    section(
        "CALCULATING AFTER METRICS"
    )

    print()

    after_results = (
        calculate_metrics(
            after_models
        )
    )

    after_metrics = (
        aggregate_metrics(
            after_results
        )
    )

    # --------------------------------------------------------
    # BEFORE VS AFTER
    # --------------------------------------------------------

    display_comparison(
        before_metrics,
        after_metrics
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    header(
        "SISA DEMONSTRATION COMPLETE"
    )

    print()

    print(
        f"  Sensitive class : "
        f"{class_name}"
    )

    print(
        f"  Affected shards : "
        f"{affected_shards}"
    )

    print(
        f"  Models loaded   : "
        f"{len(models)}/{NUM_SHARDS}"
    )

    print(
        f"  Retraining      : "
        f"{EPOCHS} epochs"
    )

    print()

    print(
        "  ✓ Sensitive data deletion completed"
    )

    print(
        "  ✓ Only affected shards retrained"
    )

    print(
        "  ✓ Unaffected shards preserved"
    )

    print(
        "  ✓ Before/after metrics calculated"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()