from ultralytics import YOLO

from unlearning.unlearning import ModelUnlearner


MODEL_PATH = "models/visdrone_best.pt"
ORIGINAL_MODEL_PATH = "models/visdrone_original.pt"


def count_class_detections(model, image_path, class_name):
    results = model(
        image_path,
        conf=0.25,
        verbose=False,
    )

    count = 0

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])
            detected_class = model.names[class_id]

            if detected_class == class_name:
                count += 1

    return count


def main():

    image_path = (
        "datasets/VisDrone/"
        "VisDrone2019-DET-val/images/"
        "0000001_02999_d_0000005.jpg"
    )

    class_to_forget = "van"

    print("\nLoading original model...")
    original_model = YOLO(ORIGINAL_MODEL_PATH)

    print("Loading current model...")
    current_model = YOLO(MODEL_PATH)

    print("\nTesting BEFORE unlearning...")

    before_count = count_class_detections(
        current_model,
        image_path,
        class_to_forget,
    )

    print(
        f"'{class_to_forget}' detections before unlearning: "
        f"{before_count}"
    )

    print("\nPerforming unlearning...")

    unlearner = ModelUnlearner(
        model=current_model
    )

    result = unlearner.forget(
        class_to_forget
    )

    print(result["message"])
    print(
        "Selected for forgetting:",
        result["forgotten"],
    )

    print("\nNOTE:")
    print(
        "The current ModelUnlearner is a prototype."
    )
    print(
        "It records the class to forget but does not "
        "modify the neural-network weights yet."
    )

    print("\nTesting AFTER prototype unlearning...")

    after_count = count_class_detections(
        current_model,
        image_path,
        class_to_forget,
    )

    print(
        f"'{class_to_forget}' detections after unlearning: "
        f"{after_count}"
    )

    print("\nOriginal model verification...")

    original_count = count_class_detections(
        original_model,
        image_path,
        class_to_forget,
    )

    print(
        f"'{class_to_forget}' detections in original model: "
        f"{original_count}"
    )

    print("\n==============================")
    print("UNLEARNING TEST RESULT")
    print("==============================")

    if before_count == after_count:
        print(
            "Prototype unlearning completed, "
            "but model predictions are unchanged."
        )
        print(
            "This is expected because the current "
            "ModelUnlearner only records the forgotten class."
        )
    else:
        print(
            "Model predictions changed after unlearning."
        )


if __name__ == "__main__":
    main()