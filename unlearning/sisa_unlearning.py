from pathlib import Path
import shutil
import random
import yaml
import traceback

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "datasets" / "KIIT-MiTA"
DATASET_YAML = DATASET_DIR / "KIIT-MiTA.yml"

BASE_MODEL = ROOT / "models" / "kiit_mita_best.pt"

OUTPUT_DIR = ROOT / "outputs" / "sisa_unlearning_fixed"

SHARDS_DIR = OUTPUT_DIR / "shards"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"

NUM_SHARDS = 3
EPOCHS = 10
IMAGE_SIZE = 640
BATCH_SIZE = 8
SEED = 42


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
# HELPERS
# ============================================================

def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def normalize_name(path):
    return path.stem.lower()


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset():

    banner("VALIDATING KIIT-MiTA DATASET")

    train_images = DATASET_DIR / "train" / "images"
    train_labels = DATASET_DIR / "train" / "labels"

    valid_images = DATASET_DIR / "valid" / "images"
    valid_labels = DATASET_DIR / "valid" / "labels"

    test_images = DATASET_DIR / "test" / "images"
    test_labels = DATASET_DIR / "test" / "labels"

    required_dirs = [
        train_images,
        train_labels,
        valid_images,
        valid_labels,
        test_images,
        test_labels,
    ]

    for directory in required_dirs:
        if not directory.exists():
            raise FileNotFoundError(
                f"Required directory not found:\n{directory}"
            )

    train_image_files = [
        p for p in train_images.iterdir()
        if p.is_file() and p.suffix.lower()
        in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ]

    train_label_files = [
        p for p in train_labels.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    ]

    print(f"Train images : {len(train_image_files)}")
    print(f"Train labels : {len(train_label_files)}")

    label_names = {
        normalize_name(p)
        for p in train_label_files
    }

    paired = []
    missing = []

    for image in train_image_files:

        if normalize_name(image) in label_names:
            label = train_labels / f"{image.stem}.txt"
            paired.append((image, label))
        else:
            missing.append(image)

    print(f"Paired images: {len(paired)}")
    print(f"Unlabeled    : {len(missing)}")

    if not paired:
        raise RuntimeError(
            "No image/label pairs were found in train/images and "
            "train/labels."
        )

    print("\n✓ Dataset structure is correct.")

    return paired


# ============================================================
# CREATE SISA SHARDS
# ============================================================

def create_shards(paired_data):

    banner("CREATING SISA SHARDS")

    random.seed(SEED)

    shuffled = paired_data.copy()
    random.shuffle(shuffled)

    # Remove previous SISA output safely.
    if SHARDS_DIR.exists():
        print("Removing previous shard directory...")

        try:
            shutil.rmtree(SHARDS_DIR)
        except PermissionError:

            print(
                "\n⚠ Previous shard directory is locked by "
                "another program."
            )

            print(
                "Close File Explorer/VS Code windows pointing to "
                "the SISA output and run the script again."
            )

            raise

    SHARDS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    total = len(shuffled)

    shard_sizes = []

    base_size = total // NUM_SHARDS
    remainder = total % NUM_SHARDS

    start = 0

    for shard_index in range(NUM_SHARDS):

        size = base_size

        if shard_index < remainder:
            size += 1

        end = start + size

        shard_data = shuffled[start:end]

        shard_sizes.append(len(shard_data))

        shard_dir = (
            SHARDS_DIR /
            f"shard_{shard_index + 1}"
        )

        images_dir = shard_dir / "images"
        labels_dir = shard_dir / "labels"

        images_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        labels_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for image, label in shard_data:

            destination_image = (
                images_dir / image.name
            )

            destination_label = (
                labels_dir / label.name
            )

            shutil.copy2(
                image,
                destination_image
            )

            shutil.copy2(
                label,
                destination_label
            )

        # Create shard YAML.
        shard_yaml = shard_dir / "data.yaml"

        yaml_data = {
            "path": str(shard_dir).replace("\\", "/"),
            "train": "images",
            "val": str(
                (DATASET_DIR / "test" / "images")
            ).replace("\\", "/"),
            "nc": len(CLASS_NAMES),
            "names": CLASS_NAMES,
        }

        with open(
            shard_yaml,
            "w",
            encoding="utf-8"
        ) as f:

            yaml.safe_dump(
                yaml_data,
                f,
                sort_keys=False
            )

        print(
            f"Shard {shard_index + 1}: "
            f"{len(shard_data)} images + "
            f"{len(shard_data)} labels"
        )

        start = end

    # Manifest
    manifest = OUTPUT_DIR / "sisa_manifest.txt"

    with open(
        manifest,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("SISA MANIFEST\n")
        f.write("=" * 60 + "\n\n")

        for i, size in enumerate(shard_sizes, 1):

            f.write(
                f"Shard {i}: {size} paired samples\n"
            )

    print("\n✓ SISA shards created.")
    print(f"Manifest: {manifest}")

    return shard_sizes


# ============================================================
# TRAIN ONE MODEL PER SHARD
# ============================================================

def train_shards():

    banner("SISA SHARD TRAINING")

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    trained = []

    for shard_number in range(1, NUM_SHARDS + 1):

        banner(
            f"TRAINING SISA SHARD {shard_number}"
        )

        shard_dir = (
            SHARDS_DIR /
            f"shard_{shard_number}"
        )

        data_yaml = shard_dir / "data.yaml"

        output_dir = (
            MODELS_DIR /
            f"shard_{shard_number}"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print(f"Dataset : {data_yaml}")
        print(f"Output  : {output_dir}")
        print(f"Epochs  : {EPOCHS}")

        try:

            model = YOLO(
                str(BASE_MODEL)
            )

            model.train(
                data=str(data_yaml),
                epochs=EPOCHS,
                imgsz=IMAGE_SIZE,
                batch=BATCH_SIZE,
                workers=2,
                device="cpu",
                project=str(MODELS_DIR),
                name=f"shard_{shard_number}",
                exist_ok=True,
                pretrained=True,
                verbose=True,
                seed=SEED,
            )

            best_model = (
                MODELS_DIR /
                f"shard_{shard_number}" /
                "weights" /
                "best.pt"
            )

            if best_model.exists():

                print(
                    f"\n✓ Shard {shard_number} training complete."
                )

                print(
                    f"Model: {best_model}"
                )

                trained.append(
                    best_model
                )

            else:

                print(
                    f"\n⚠ Shard {shard_number} finished "
                    "but best.pt was not found."
                )

        except Exception as e:

            print(
                f"\n❌ Shard {shard_number} training failed."
            )

            print(
                f"Reason: {e}"
            )

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("SISA TRAINING SUMMARY")
    print("=" * 70)

    print(
        f"Models trained: "
        f"{len(trained)}/{NUM_SHARDS}"
    )

    return trained


# ============================================================
# FIND SHARD CONTAINING FORGOTTEN CLASS
# ============================================================

def shard_contains_class(shard_number, class_id):

    labels_dir = (
        SHARDS_DIR /
        f"shard_{shard_number}" /
        "labels"
    )

    if not labels_dir.exists():
        return False

    for label_file in labels_dir.glob("*.txt"):

        try:

            with open(
                label_file,
                "r",
                encoding="utf-8"
            ) as f:

                for line in f:

                    parts = line.strip().split()

                    if not parts:
                        continue

                    if int(float(parts[0])) == class_id:
                        return True

        except Exception:
            continue

    return False


# ============================================================
# SISA UNLEARNING
# ============================================================

def perform_unlearning(
    forgotten_class
):

    banner("SISA MACHINE UNLEARNING")

    forgotten_class = forgotten_class.strip()

    if forgotten_class not in CLASS_NAMES:

        print(
            f"❌ Unknown class: {forgotten_class}"
        )

        print("\nAvailable classes:")

        for i, name in enumerate(CLASS_NAMES):
            print(f"  {i}: {name}")

        return

    class_id = CLASS_NAMES.index(
        forgotten_class
    )

    print(
        f"Forgotten class: "
        f"{forgotten_class}"
    )

    print(
        f"Class ID: {class_id}"
    )

    affected = []

    for shard_number in range(
        1,
        NUM_SHARDS + 1
    ):

        if shard_contains_class(
            shard_number,
            class_id
        ):

            affected.append(
                shard_number
            )

    print(
        f"\nAffected shards: "
        f"{affected}"
    )

    if not affected:

        print(
            "\nNo shard contains the requested "
            "class."
        )

        return

    print(
        "\nSISA principle:"
    )

    print(
        "Only affected shards need to be "
        "retrained."
    )

    # --------------------------------------------------------
    # Retrain affected shards after removing forgotten class
    # --------------------------------------------------------

    for shard_number in affected:

        banner(
            f"RETRAINING AFFECTED SHARD {shard_number}"
        )

        shard_dir = (
            SHARDS_DIR /
            f"shard_{shard_number}"
        )

        labels_dir = (
            shard_dir / "labels"
        )

        images_dir = (
            shard_dir / "images"
        )

        unlearn_images = (
            shard_dir /
            "unlearning_images"
        )

        unlearn_labels = (
            shard_dir /
            "unlearning_labels"
        )

        unlearn_images.mkdir(
            exist_ok=True
        )

        unlearn_labels.mkdir(
            exist_ok=True
        )

        kept = 0
        removed = 0

        for label_file in labels_dir.glob("*.txt"):

            remove_sample = False

            try:

                with open(
                    label_file,
                    "r",
                    encoding="utf-8"
                ) as f:

                    for line in f:

                        parts = line.strip().split()

                        if parts:

                            if int(float(parts[0])) == class_id:
                                remove_sample = True
                                break

            except Exception:
                remove_sample = False

            image_candidates = [
                images_dir / f"{label_file.stem}.jpg",
                images_dir / f"{label_file.stem}.jpeg",
                images_dir / f"{label_file.stem}.png",
                images_dir / f"{label_file.stem}.bmp",
                images_dir / f"{label_file.stem}.webp",
            ]

            image_file = next(
                (
                    p for p in image_candidates
                    if p.exists()
                ),
                None
            )

            if remove_sample:

                removed += 1

                if image_file:
                    image_file.unlink(
                        missing_ok=True
                    )

                label_file.unlink(
                    missing_ok=True
                )

            else:

                kept += 1

        print(
            f"Removed samples: {removed}"
        )

        print(
            f"Remaining samples: {kept}"
        )

        # Recreate YAML.
        shard_yaml = shard_dir / "data.yaml"

        yaml_data = {
            "path": str(shard_dir).replace("\\", "/"),
            "train": "images",
            "val": str(
                DATASET_DIR /
                "test" /
                "images"
            ).replace("\\", "/"),
            "nc": len(CLASS_NAMES),
            "names": CLASS_NAMES,
        }

        with open(
            shard_yaml,
            "w",
            encoding="utf-8"
        ) as f:

            yaml.safe_dump(
                yaml_data,
                f,
                sort_keys=False
            )

        # Retrain.
        try:

            model = YOLO(
                str(BASE_MODEL)
            )

            model.train(
                data=str(shard_yaml),
                epochs=EPOCHS,
                imgsz=IMAGE_SIZE,
                batch=BATCH_SIZE,
                workers=2,
                device="cpu",
                project=str(MODELS_DIR),
                name=f"shard_{shard_number}_unlearned",
                exist_ok=True,
                pretrained=True,
                verbose=True,
                seed=SEED,
            )

            print(
                f"\n✓ Shard {shard_number} "
                "successfully retrained."
            )

        except Exception as e:

            print(
                f"\n❌ Retraining failed for "
                f"shard {shard_number}: {e}"
            )

    print("\n" + "=" * 70)
    print("UNLEARNING COMPLETE")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    banner("SISA MACHINE UNLEARNING")

    print(f"Dataset:")
    print(DATASET_DIR)

    print(f"\nDataset YAML:")
    print(DATASET_YAML)

    print(f"\nBase model:")
    print(BASE_MODEL)

    print(f"\nOutput:")
    print(OUTPUT_DIR)

    # Check model.
    if not BASE_MODEL.exists():

        print(
            "\n❌ Base YOLO model not found:"
        )

        print(BASE_MODEL)

        return

    print(
        "\n✓ Base YOLO model found."
    )

    # Validate.
    try:

        paired_data = validate_dataset()

    except Exception as e:

        print(
            f"\n❌ Dataset validation failed:"
        )

        print(e)

        return

    # Create shards.
    try:

        create_shards(
            paired_data
        )

    except Exception as e:

        print(
            "\n❌ SISA shard creation failed."
        )

        print(
            f"Reason: {e}"
        )

        return

    # Train.
    trained = train_shards()

    # Optional unlearning.
    print("\n")

    forgotten = input(
        "Enter class to forget "
        "(or press ENTER to skip): "
    ).strip()

    if forgotten:

        perform_unlearning(
            forgotten
        )

    else:

        print(
            "No unlearning request selected."
        )

    banner("SISA PIPELINE COMPLETE")

    print(
        f"Shard data:\n{SHARDS_DIR}"
    )

    print(
        f"\nShard models:\n{MODELS_DIR}"
    )

    print(
        f"\nResults:\n{RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()