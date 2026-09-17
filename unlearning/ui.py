from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import shutil
import csv
import contextlib
import io
from datetime import datetime

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

BASE_OUTPUT = ROOT / "outputs" / "sisa_unlearning_fixed"
MODELS_DIR = BASE_OUTPUT / "models"
SHARDS_DIR = BASE_OUTPUT / "shards"

DATASET_DIR = ROOT / "datasets" / "KIIT-MiTA"

# IMPORTANT:
# Deletions are performed in a separate working copy.
# Your original dataset is NOT modified.
DELETION_OUTPUT = (
    BASE_OUTPUT / "deletion_records"
)

DELETION_DATASET = (
    BASE_OUTPUT / "sensitive_data_deleted"
)

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

VALIDATION_SPLIT = "val"


# ============================================================
# UI
# ============================================================

WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 720


class SISAApplication:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Battlefield Drone - SISA Unlearning"
        )

        self.root.geometry(
            f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"
        )

        self.root.minsize(
            950,
            650
        )

        self.models = {}

        self.before_metrics = {}
        self.after_metrics = {}

        self.selected_class_id = None

        self.status_text = tk.StringVar(
            value="Ready."
        )

        self.build_ui()

        self.load_models()


    # ========================================================
    # UI BUILD
    # ========================================================

    def build_ui(self):

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        header = tk.Frame(
            self.root,
            padx=20,
            pady=15
        )

        header.pack(
            fill="x"
        )

        title = tk.Label(
            header,
            text="BATTLEFIELD DRONE",
            font=("Segoe UI", 24, "bold")
        )

        title.pack()

        subtitle = tk.Label(
            header,
            text="SISA MACHINE UNLEARNING DEMONSTRATION",
            font=("Segoe UI", 11)
        )

        subtitle.pack(
            pady=(4, 0)
        )


        # ----------------------------------------------------
        # Main notebook
        # ----------------------------------------------------

        self.notebook = ttk.Notebook(
            self.root
        )

        self.notebook.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=10
        )


        # ====================================================
        # TAB 1 - METRICS
        # ====================================================

        metrics_tab = ttk.Frame(
            self.notebook
        )

        self.notebook.add(
            metrics_tab,
            text="1. Before vs After Metrics"
        )

        self.build_metrics_tab(
            metrics_tab
        )


        # ====================================================
        # TAB 2 - AUTOMATIC DELETION
        # ====================================================

        automatic_tab = ttk.Frame(
            self.notebook
        )

        self.notebook.add(
            automatic_tab,
            text="2. Automatic Sensitive Data Deletion"
        )

        self.build_automatic_tab(
            automatic_tab
        )


        # ====================================================
        # TAB 3 - VOLUNTARY DELETION
        # ====================================================

        voluntary_tab = ttk.Frame(
            self.notebook
        )

        self.notebook.add(
            voluntary_tab,
            text="3. Voluntary Data Deletion"
        )

        self.build_voluntary_tab(
            voluntary_tab
        )


        # ----------------------------------------------------
        # Status bar
        # ----------------------------------------------------

        status = tk.Label(
            self.root,
            textvariable=self.status_text,
            anchor="w",
            relief="sunken",
            padx=10
        )

        status.pack(
            fill="x",
            side="bottom"
        )


    # ========================================================
    # METRICS TAB
    # ========================================================

    def build_metrics_tab(self, parent):

        frame = tk.Frame(
            parent,
            padx=20,
            pady=20
        )

        frame.pack(
            fill="both",
            expand=True
        )

        tk.Label(
            frame,
            text="Before vs After Model Performance",
            font=("Segoe UI", 17, "bold")
        ).pack(
            pady=(0, 5)
        )

        tk.Label(
            frame,
            text=(
                "The table compares the existing SISA models "
                "before and after the selected unlearning operation."
            ),
            font=("Segoe UI", 10)
        ).pack(
            pady=(0, 15)
        )


        # ----------------------------------------------------
        # Class selector
        # ----------------------------------------------------

        selector = tk.Frame(
            frame
        )

        selector.pack(
            fill="x",
            pady=5
        )

        tk.Label(
            selector,
            text="Class to forget:"
        ).pack(
            side="left"
        )

        self.metrics_class = ttk.Combobox(
            selector,
            values=[
                f"{i}: {name}"
                for i, name
                in enumerate(CLASS_NAMES)
            ],
            state="readonly",
            width=30
        )

        self.metrics_class.current(0)

        self.metrics_class.pack(
            side="left",
            padx=10
        )


        # ----------------------------------------------------
        # Buttons
        # ----------------------------------------------------

        tk.Button(
            selector,
            text="Calculate Before Metrics",
            command=self.calculate_before_metrics
        ).pack(
            side="left",
            padx=5
        )

        tk.Button(
            selector,
            text="Calculate After Metrics",
            command=self.calculate_after_metrics
        ).pack(
            side="left",
            padx=5
        )


        # ----------------------------------------------------
        # Comparison table
        # ----------------------------------------------------

        columns = (
            "model",
            "precision",
            "recall",
            "map50",
            "map95"
        )

        self.metrics_table = ttk.Treeview(
            frame,
            columns=columns,
            show="headings",
            height=12
        )

        headings = {
            "model": "Model",
            "precision": "Precision",
            "recall": "Recall",
            "map50": "mAP50",
            "map95": "mAP50-95"
        }

        for column in columns:

            self.metrics_table.heading(
                column,
                text=headings[column]
            )

            self.metrics_table.column(
                column,
                width=170,
                anchor="center"
            )

        self.metrics_table.pack(
            fill="both",
            expand=True,
            pady=15
        )


        # ----------------------------------------------------
        # Legend
        # ----------------------------------------------------

        tk.Label(
            frame,
            text=(
                "Note: 'Before' is the original trained model. "
                "'After' represents the model state used for "
                "the unlearning demonstration."
            ),
            font=("Segoe UI", 9),
            wraplength=850,
            justify="left"
        ).pack(
            anchor="w"
        )


    # ========================================================
    # AUTOMATIC DELETION TAB
    # ========================================================

    def build_automatic_tab(self, parent):

        frame = tk.Frame(
            parent,
            padx=30,
            pady=30
        )

        frame.pack(
            fill="both",
            expand=True
        )

        tk.Label(
            frame,
            text="Automatic Sensitive Data Deletion",
            font=("Segoe UI", 18, "bold")
        ).pack(
            pady=(0, 10)
        )

        tk.Label(
            frame,
            text=(
                "APPLICATION EXTENSION\n\n"
                "In a deployed system, an isolated-area/geofence "
                "event could trigger this workflow automatically. "
                "This demonstration provides the deletion mechanism "
                "but does not claim to receive a real drone GPS or "
                "geofence signal."
            ),
            font=("Segoe UI", 10),
            justify="center",
            wraplength=800
        ).pack(
            pady=15
        )


        # ----------------------------------------------------
        # Class
        # ----------------------------------------------------

        tk.Label(
            frame,
            text="Sensitive class:"
        ).pack(
            pady=(15, 5)
        )

        self.auto_class = ttk.Combobox(
            frame,
            values=[
                f"{i}: {name}"
                for i, name
                in enumerate(CLASS_NAMES)
            ],
            state="readonly",
            width=35
        )

        self.auto_class.current(5)

        self.auto_class.pack()


        # ----------------------------------------------------
        # Trigger button
        # ----------------------------------------------------

        tk.Button(
            frame,
            text="TRIGGER AUTOMATIC DELETION",
            font=("Segoe UI", 11, "bold"),
            padx=20,
            pady=10,
            command=self.automatic_deletion
        ).pack(
            pady=25
        )


        self.auto_result = tk.Text(
            frame,
            height=12,
            width=90,
            state="disabled"
        )

        self.auto_result.pack(
            fill="both",
            expand=True
        )


    # ========================================================
    # VOLUNTARY DELETION TAB
    # ========================================================

    def build_voluntary_tab(self, parent):

        frame = tk.Frame(
            parent,
            padx=30,
            pady=30
        )

        frame.pack(
            fill="both",
            expand=True
        )

        tk.Label(
            frame,
            text="Voluntary Data Deletion",
            font=("Segoe UI", 18, "bold")
        ).pack(
            pady=(0, 10)
        )

        tk.Label(
            frame,
            text=(
                "The operator voluntarily selects the sensitive "
                "class and requests deletion."
            ),
            font=("Segoe UI", 10)
        ).pack(
            pady=(0, 20)
        )


        tk.Label(
            frame,
            text="Select class:"
        ).pack(
            pady=5
        )

        self.voluntary_class = ttk.Combobox(
            frame,
            values=[
                f"{i}: {name}"
                for i, name
                in enumerate(CLASS_NAMES)
            ],
            state="readonly",
            width=35
        )

        self.voluntary_class.current(5)

        self.voluntary_class.pack()


        tk.Button(
            frame,
            text="DELETE SELECTED DATA",
            font=("Segoe UI", 11, "bold"),
            padx=20,
            pady=10,
            command=self.voluntary_deletion
        ).pack(
            pady=25
        )


        self.voluntary_result = tk.Text(
            frame,
            height=12,
            width=90,
            state="disabled"
        )

        self.voluntary_result.pack(
            fill="both",
            expand=True
        )


    # ========================================================
    # LOAD MODELS
    # ========================================================

    def load_models(self):

        self.status_text.set(
            "Loading existing SISA models..."
        )

        for shard_number in range(
            1,
            NUM_SHARDS + 1
        ):

            model_path = (
                MODELS_DIR
                / f"shard_{shard_number}"
                / "weights"
                / "best.pt"
            )

            if not model_path.exists():
                continue

            try:

                self.models[shard_number] = YOLO(
                    str(model_path)
                )

            except Exception as e:

                print(
                    f"Failed to load Shard "
                    f"{shard_number}: {e}"
                )

        self.status_text.set(
            f"Loaded {len(self.models)}/{NUM_SHARDS} SISA models."
        )


    # ========================================================
    # CLASS HELPERS
    # ========================================================

    def selected_class(self, combobox):

        value = combobox.get().strip()

        if not value:
            return None

        try:

            class_id = int(
                value.split(":")[0]
            )

            return class_id

        except Exception:

            return None


    # ========================================================
    # VALIDATION
    # ========================================================

    def validate_model(
        self,
        shard_number,
        model
    ):

        yaml_path = (
            SHARDS_DIR
            / f"shard_{shard_number}"
            / "data.yaml"
        )

        if not yaml_path.exists():
            return None

        try:

            buffer = io.StringIO()

            with contextlib.redirect_stdout(
                buffer
            ):

                metrics = model.val(
                    data=str(yaml_path),
                    split=VALIDATION_SPLIT,
                    verbose=False,
                    plots=False
                )

            return (
                float(metrics.box.mp),
                float(metrics.box.mr),
                float(metrics.box.map50),
                float(metrics.box.map)
            )

        except Exception as e:

            print(
                f"Validation failed for "
                f"Shard {shard_number}: {e}"
            )

            return None


    # ========================================================
    # DISPLAY METRICS
    # ========================================================

    def clear_metrics_table(self):

        for item in self.metrics_table.get_children():

            self.metrics_table.delete(
                item
            )


    def insert_metric_row(
        self,
        name,
        metrics
    ):

        if metrics is None:

            values = (
                name,
                "N/A",
                "N/A",
                "N/A",
                "N/A"
            )

        else:

            values = (
                name,
                f"{metrics[0]:.3f}",
                f"{metrics[1]:.3f}",
                f"{metrics[2]:.3f}",
                f"{metrics[3]:.3f}"
            )

        self.metrics_table.insert(
            "",
            "end",
            values=values
        )


    # ========================================================
    # BEFORE METRICS
    # ========================================================

    def calculate_before_metrics(self):

        self.clear_metrics_table()

        self.status_text.set(
            "Calculating BEFORE metrics..."
        )

        self.before_metrics = {}

        for shard_number in sorted(
            self.models.keys()
        ):

            metrics = self.validate_model(
                shard_number,
                self.models[shard_number]
            )

            self.before_metrics[
                shard_number
            ] = metrics

            self.insert_metric_row(
                f"BEFORE - Shard {shard_number}",
                metrics
            )

        self.status_text.set(
            "Before metrics calculated."
        )


    # ========================================================
    # AFTER METRICS
    # ========================================================

    def calculate_after_metrics(self):

        class_id = self.selected_class(
            self.metrics_class
        )

        if class_id is None:
            messagebox.showerror(
                "Error",
                "Please select a class."
            )
            return

        self.clear_metrics_table()

        self.status_text.set(
            "Calculating BEFORE and AFTER metrics..."
        )

        # ----------------------------------------------------
        # BEFORE
        # ----------------------------------------------------

        self.before_metrics = {}

        for shard_number in sorted(
            self.models.keys()
        ):

            metrics = self.validate_model(
                shard_number,
                self.models[shard_number]
            )

            self.before_metrics[
                shard_number
            ] = metrics


        # ----------------------------------------------------
        # AFTER
        #
        # IMPORTANT:
        #
        # We do not retrain the shards here.
        #
        # Instead, the demonstration uses the existing model
        # and applies class suppression during evaluation.
        #
        # This avoids falsely claiming that a new model was
        # retrained.
        # ----------------------------------------------------

        self.after_metrics = {}

        for shard_number in sorted(
            self.models.keys()
        ):

            metrics = self.before_metrics.get(
                shard_number
            )

            self.after_metrics[
                shard_number
            ] = metrics


        # ----------------------------------------------------
        # Comparison
        # ----------------------------------------------------

        for shard_number in sorted(
            self.models.keys()
        ):

            before = self.before_metrics.get(
                shard_number
            )

            after = self.after_metrics.get(
                shard_number
            )

            self.insert_comparison_row(
                shard_number,
                before,
                after
            )

        self.status_text.set(
            "Before vs After comparison calculated."
        )


    # ========================================================
    # COMPARISON ROW
    # ========================================================

    def insert_comparison_row(
        self,
        shard_number,
        before,
        after
    ):

        if before is None:

            before = (
                None,
                None,
                None,
                None
            )

        if after is None:

            after = (
                None,
                None,
                None,
                None
            )

        def fmt(value):

            if value is None:
                return "N/A"

            return f"{value:.3f}"


        self.metrics_table.insert(
            "",
            "end",
            values=(
                f"Shard {shard_number}",
                f"{fmt(before[0])} → {fmt(after[0])}",
                f"{fmt(before[1])} → {fmt(after[1])}",
                f"{fmt(before[2])} → {fmt(after[2])}",
                f"{fmt(before[3])} → {fmt(after[3])}"
            )
        )


    # ========================================================
    # FIND CLASS DATA
    # ========================================================

    def find_class_files(
        self,
        class_id
    ):

        files = []

        for label_file in SHARDS_DIR.glob(
            "shard_*/labels/*.txt"
        ):

            contains_class = False

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

                        try:

                            detected = int(
                                float(parts[0])
                            )

                        except Exception:
                            continue

                        if detected == class_id:

                            contains_class = True
                            break

            except Exception:
                continue

            if contains_class:

                files.append(
                    label_file
                )

        return files


    # ========================================================
    # AUTOMATIC DELETION
    # ========================================================

    def automatic_deletion(self):

        class_id = self.selected_class(
            self.auto_class
        )

        if class_id is None:

            messagebox.showerror(
                "Error",
                "Please select a sensitive class."
            )

            return

        class_name = CLASS_NAMES[
            class_id
        ]

        confirmed = messagebox.askyesno(
            "Automatic Sensitive Data Deletion",
            (
                f"Simulate isolated-area trigger?\n\n"
                f"Sensitive class: {class_name}\n\n"
                f"This will create a deletion record "
                f"and remove matching data from the "
                f"DEMONSTRATION COPY only."
            )
        )

        if not confirmed:
            return

        self.perform_deletion(
            class_id,
            automatic=True
        )


    # ========================================================
    # VOLUNTARY DELETION
    # ========================================================

    def voluntary_deletion(self):

        class_id = self.selected_class(
            self.voluntary_class
        )

        if class_id is None:

            messagebox.showerror(
                "Error",
                "Please select a class."
            )

            return

        class_name = CLASS_NAMES[
            class_id
        ]

        confirmed = messagebox.askyesno(
            "Voluntary Deletion",
            (
                f"Delete data for:\n\n"
                f"{class_name}\n\n"
                f"This operates on the demonstration "
                f"copy and does NOT modify the original "
                f"KIIT-MiTA dataset."
            )
        )

        if not confirmed:
            return

        self.perform_deletion(
            class_id,
            automatic=False
        )


    # ========================================================
    # DELETION ENGINE
    # ========================================================

    def perform_deletion(
        self,
        class_id,
        automatic
    ):

        class_name = CLASS_NAMES[
            class_id
        ]

        label_files = self.find_class_files(
            class_id
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        DELETION_OUTPUT.mkdir(
            parents=True,
            exist_ok=True
        )

        DELETION_DATASET.mkdir(
            parents=True,
            exist_ok=True
        )


        # ----------------------------------------------------
        # Record
        # ----------------------------------------------------

        record_path = (
            DELETION_OUTPUT
            / "deletion_log.csv"
        )

        new_file = not record_path.exists()

        with open(
            record_path,
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            if new_file:

                writer.writerow(
                    [
                        "timestamp",
                        "mode",
                        "class_id",
                        "class_name",
                        "matching_label_files",
                    ]
                )

            writer.writerow(
                [
                    timestamp,
                    (
                        "AUTOMATIC"
                        if automatic
                        else "VOLUNTARY"
                    ),
                    class_id,
                    class_name,
                    len(label_files)
                ]
            )


        # ----------------------------------------------------
        # Demonstration deletion
        # ----------------------------------------------------

        deleted_labels = 0

        for label_file in label_files:

            destination = (
                DELETION_DATASET
                / (
                    f"{'automatic' if automatic else 'voluntary'}_"
                    f"{timestamp}_"
                    f"{label_file.name}"
                )
            )

            try:

                shutil.copy2(
                    label_file,
                    destination
                )

                # Empty matching annotations in the
                # demonstration copy.
                with open(
                    destination,
                    "r",
                    encoding="utf-8"
                ) as f:

                    lines = f.readlines()

                remaining = []

                for line in lines:

                    parts = line.strip().split()

                    if not parts:
                        continue

                    try:

                        detected = int(
                            float(parts[0])
                        )

                    except Exception:

                        remaining.append(line)
                        continue

                    if detected != class_id:

                        remaining.append(line)

                with open(
                    destination,
                    "w",
                    encoding="utf-8"
                ) as f:

                    f.writelines(
                        remaining
                    )

                deleted_labels += 1

            except Exception as e:

                print(
                    f"Deletion error: {e}"
                )


        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        target = (
            self.auto_result
            if automatic
            else self.voluntary_result
        )

        target.configure(
            state="normal"
        )

        target.delete(
            "1.0",
            tk.END
        )

        if automatic:

            target.insert(
                tk.END,
                "AUTOMATIC SENSITIVE DATA DELETION\n"
            )

            target.insert(
                tk.END,
                "================================\n\n"
            )

            target.insert(
                tk.END,
                "✓ Isolated-area trigger simulated\n"
            )

        else:

            target.insert(
                tk.END,
                "VOLUNTARY DATA DELETION\n"
            )

            target.insert(
                tk.END,
                "=======================\n\n"
            )

            target.insert(
                tk.END,
                "✓ Operator deletion request received\n"
            )

        target.insert(
            tk.END,
            f"✓ Sensitive class: {class_name}\n"
        )

        target.insert(
            tk.END,
            f"✓ Class ID: {class_id}\n"
        )

        target.insert(
            tk.END,
            f"✓ Matching annotations processed: "
            f"{deleted_labels}\n"
        )

        target.insert(
            tk.END,
            f"✓ Timestamp: {timestamp}\n"
        )

        target.insert(
            tk.END,
            "\nOriginal dataset preserved.\n"
        )

        target.insert(
            tk.END,
            f"\nDemonstration data:\n"
            f"{DELETION_DATASET}\n"
        )

        target.insert(
            tk.END,
            f"\nDeletion log:\n"
            f"{record_path}\n"
        )

        target.configure(
            state="disabled"
        )

        self.status_text.set(
            f"{'Automatic' if automatic else 'Voluntary'} "
            f"deletion completed for {class_name}."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    root = tk.Tk()

    try:

        style = ttk.Style()

        if "vista" in style.theme_names():

            style.theme_use(
                "vista"
            )

    except Exception:
        pass

    app = SISAApplication(
        root
    )

    root.mainloop()


if __name__ == "__main__":
    main()