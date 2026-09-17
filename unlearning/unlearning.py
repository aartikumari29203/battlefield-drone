class ModelUnlearner:
    """
    Class-based machine unlearning controller.

    This module maintains the classes selected for forgetting
    and filters detections belonging to those classes.

    Note:
        This is an inference-time unlearning prototype.
        It does not modify the neural-network weights.
    """

    def __init__(self, model=None):
        self.model = model
        self.forgotten_data = []

    def forget(self, data_to_forget):
        """
        Add a class/data category to the forgotten set.
        """

        if data_to_forget is None:
            raise ValueError(
                "Data to forget cannot be empty."
            )

        data_to_forget = str(
            data_to_forget
        ).strip().lower()

        if not data_to_forget:
            raise ValueError(
                "Data to forget cannot be empty."
            )

        if data_to_forget not in self.forgotten_data:
            self.forgotten_data.append(
                data_to_forget
            )

        return {
            "status": "success",
            "forgotten": data_to_forget,
            "message": (
                f"'{data_to_forget}' has been "
                "selected for unlearning."
            ),
        }

    def restore(self, data_to_restore):
        """
        Remove a class from the forgotten set.
        """

        if data_to_restore is None:
            raise ValueError(
                "Data to restore cannot be empty."
            )

        data_to_restore = str(
            data_to_restore
        ).strip().lower()

        if not data_to_restore:
            raise ValueError(
                "Data to restore cannot be empty."
            )

        if data_to_restore in self.forgotten_data:
            self.forgotten_data.remove(
                data_to_restore
            )

        return {
            "status": "success",
            "restored": data_to_restore,
            "message": (
                f"'{data_to_restore}' is no longer "
                "selected for unlearning."
            ),
        }

    def filter_predictions(self, detections):
        """
        Remove detections belonging to forgotten classes.

        Expected detection format:

        {
            "bbox": [...],
            "confidence": 0.95,
            "class": "Tank"
        }
        """

        if not detections:
            return []

        if not self.forgotten_data:
            return detections

        filtered = []

        for detection in detections:

            class_name = str(
                detection.get("class", "")
            ).strip().lower()

            if class_name in self.forgotten_data:
                continue

            filtered.append(detection)

        return filtered

    def get_forgotten_data(self):
        """
        Return all currently forgotten classes.
        """

        return list(
            self.forgotten_data
        )

    def is_forgotten(self, class_name):
        """
        Check whether a class is currently forgotten.
        """

        if class_name is None:
            return False

        class_name = str(
            class_name
        ).strip().lower()

        return (
            class_name in self.forgotten_data
        )

    def clear(self):
        """
        Clear all forgotten classes.
        """

        self.forgotten_data.clear()

        return {
            "status": "success",
            "message": (
                "All forgotten classes have been cleared."
            ),
        }


if __name__ == "__main__":

    unlearner = ModelUnlearner()

    # Select Tank for forgetting
    result = unlearner.forget("Tank")

    print(result)

    detections = [
        {
            "bbox": [100, 100, 200, 200],
            "confidence": 0.91,
            "class": "Tank",
        },
        {
            "bbox": [300, 300, 400, 400],
            "confidence": 0.88,
            "class": "Vehicle",
        },
        {
            "bbox": [500, 200, 600, 350],
            "confidence": 0.82,
            "class": "Soldier",
        },
    ]

    print("\nBefore unlearning:")
    print(detections)

    filtered = unlearner.filter_predictions(
        detections
    )

    print("\nAfter unlearning:")
    print(filtered)

    print("\nForgotten classes:")
    print(
        unlearner.get_forgotten_data()
    )