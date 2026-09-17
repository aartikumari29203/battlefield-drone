class ModelUnlearner:
    def __init__(self, model=None):
        self.model = model
        self.forgotten_data = []

    def forget(self, data_to_forget):
        """
        Prototype machine-unlearning operation.

        Records the data/class selected for forgetting.
        """

        if not data_to_forget:
            raise ValueError("Data to forget cannot be empty.")

        self.forgotten_data.append(data_to_forget)

        return {
            "status": "success",
            "forgotten": data_to_forget,
            "message": "Unlearning operation completed."
        }

    def get_forgotten_data(self):
        """Return the data/classes selected for forgetting."""
        return self.forgotten_data


if __name__ == "__main__":

    unlearner = ModelUnlearner()

    result = unlearner.forget("example_data")

    print(result)
    print("Forgotten data:", unlearner.get_forgotten_data())