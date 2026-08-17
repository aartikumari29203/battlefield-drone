from pathlib import Path

import cv2


IMAGE_DIR = Path(
    "datasets/VisDrone/VisDrone-DET/val/"
    "VisDrone2019-DET-val/images"
)

OUTPUT_PATH = Path("integration/visdrone_test.mp4")


def main():
    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))[:5]

    if not image_paths:
        print("No images found.")
        return

    first_frame = cv2.imread(str(image_paths[0]))

    if first_frame is None:
        print("Could not read first image.")
        return

    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        fourcc,
        2.0,
        (width, height),
    )

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))

        if frame is None:
            continue

        writer.write(frame)

    writer.release()

    print(f"Created: {OUTPUT_PATH}")
    print(f"Frames: {len(image_paths)}")
    print(f"Resolution: {width}x{height}")


if __name__ == "__main__":
    main()