from pathlib import Path
from PIL import Image


# Original VisDrone category IDs:
# 1 pedestrian
# 2 person
# 3 car
# 4 van
# 5 truck
# 6 tricycle
# 7 awning-tricycle
# 8 bus
# 9 motor
# 10 others
#
# YOLO uses class IDs starting from 0.
# We keep all 10 VisDrone categories.

CLASS_MAP = {
    1: 0,  # pedestrian
    2: 1,  # person
    3: 2,  # car
    4: 3,  # van
    5: 4,  # truck
    6: 5,  # tricycle
    7: 6,  # awning-tricycle
    8: 7,  # bus
    9: 8,  # motor
    10: 9,  # others
}


def convert_dataset():
    base = Path("datasets/VisDrone/VisDrone-DET")

    image_dir = base / "images"
    annotation_dir = base / "annotations"
    label_dir = base / "labels"

    label_dir.mkdir(exist_ok=True)

    annotation_files = list(annotation_dir.glob("*.txt"))

    print(f"Found {len(annotation_files)} annotation files.")

    converted = 0
    skipped = 0

    for annotation_file in annotation_files:
        image_file = image_dir / f"{annotation_file.stem}.jpg"

        if not image_file.exists():
            print(f"Missing image: {image_file.name}")
            skipped += 1
            continue

        with Image.open(image_file) as img:
            image_width, image_height = img.size

        yolo_lines = []

        with open(annotation_file, "r") as f:
            for line in f:
                values = line.strip().split(",")

                if len(values) < 8:
                    continue

                x, y, width, height = map(float, values[:4])
                category = int(values[5])
                ignored = int(values[4])

                # Ignore invalid/unknown categories
                if category not in CLASS_MAP:
                    continue

                # Ignore VisDrone ignored regions
                if ignored == 0:
                    continue

                x_center = (x + width / 2) / image_width
                y_center = (y + height / 2) / image_height
                norm_width = width / image_width
                norm_height = height / image_height

                class_id = CLASS_MAP[category]

                yolo_lines.append(
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{norm_width:.6f} "
                    f"{norm_height:.6f}"
                )

        output_file = label_dir / annotation_file.name

        with open(output_file, "w") as f:
            f.write("\n".join(yolo_lines))

        converted += 1

    print(f"Converted: {converted}")
    print(f"Skipped: {skipped}")
    print(f"YOLO labels saved to: {label_dir}")


if __name__ == "__main__":
    convert_dataset()