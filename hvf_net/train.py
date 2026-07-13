"""
Train HVF-Net supervisor model.

Run:
    python data/generate_hvf_data.py
    python hvf_net/train.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hvf_net.model import build_hvf_net  # noqa: E402
from hvf_net.pipeline import load_class_info, preprocess_classifier_image  # noqa: E402

HVF_DATA_DIR = PROJECT_ROOT / "hvf_data"
TRAINING_CSV = PROJECT_ROOT / "hvf_training_data.csv"
MODEL_OUTPUT = PROJECT_ROOT / "models" / "hvf_net_model.keras"
PLOT_OUTPUT = PROJECT_ROOT / "hvf_training_results.png"
METRICS_OUTPUT = PROJECT_ROOT / "hvf_evaluation_report.json"
HISTORY_JSON_OUTPUT = PROJECT_ROOT / "hvf_training_history.json"
HISTORY_CSV_OUTPUT = PROJECT_ROOT / "hvf_training_history.csv"

BATCH_SIZE = 16
EPOCHS = 40
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42


def load_training_arrays(csv_path: Path):
    df = pd.read_csv(csv_path)
    classes, _, _ = load_class_info()
    prob_cols = [f"cls_prob_{name}" for name in classes]

    images = []
    classifier_probs = []
    specialist_features = []
    route_ids = []
    y_trust = []
    y_action = []

    for _, row in df.iterrows():
        image_path = HVF_DATA_DIR / str(row["image_path"]).replace("\\", "/")
        image = preprocess_classifier_image(str(image_path))[0]
        images.append(image)
        classifier_probs.append(row[prob_cols].astype(np.float32).values)
        specialist_features.append(
            np.array(
                [
                    row["specialist_score"],
                    row["specialist_confidence"],
                    row["specialist_positive"],
                ],
                dtype=np.float32,
            )
        )
        route_ids.append(np.array([int(row["route_id"])], dtype=np.int32))
        y_trust.append(float(row["y_trust"]))
        y_action.append(int(row["y_action"]))

    return {
        "images": np.array(images, dtype=np.float32),
        "classifier_probs": np.array(classifier_probs, dtype=np.float32),
        "specialist_features": np.array(specialist_features, dtype=np.float32),
        "route_ids": np.array(route_ids, dtype=np.int32),
        "y_trust": np.array(y_trust, dtype=np.float32),
        "y_action": np.array(y_action, dtype=np.int32),
    }


def plot_history(history, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["trust_loss"], label="train")
    axes[0].plot(history.history["val_trust_loss"], label="val")
    axes[0].set_title("Trust Loss")
    axes[0].legend()

    axes[1].plot(history.history["trust_accuracy"], label="train acc")
    axes[1].plot(history.history["val_trust_accuracy"], label="val acc")
    axes[1].plot(history.history["trust_auc"], label="train auc")
    axes[1].plot(history.history["val_trust_auc"], label="val auc")
    axes[1].set_title("Trust Metrics")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_training_history(history, json_path: Path, csv_path: Path) -> None:
    """Save per-epoch metrics (all epochs) to JSON and CSV."""
    history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
    num_epochs = len(next(iter(history_dict.values()), []))

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "epochs": num_epochs,
                "metrics_per_epoch": history_dict,
            },
            file,
            indent=2,
        )

    rows = []
    for epoch in range(num_epochs):
        row = {"epoch": epoch + 1}
        for key, values in history_dict.items():
            row[key] = values[epoch]
        rows.append(row)

    pd.DataFrame(rows).to_csv(csv_path, index=False)


def main() -> None:
    os.chdir(PROJECT_ROOT)
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    if not TRAINING_CSV.exists():
        raise FileNotFoundError(
            f"{TRAINING_CSV} not found. Run: python data/generate_hvf_data.py"
        )

    data = load_training_arrays(TRAINING_CSV)
    indices = np.arange(len(data["y_trust"]))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        stratify=data["y_action"],
    )

    train_inputs = {
        "image": data["images"][train_idx],
        "classifier_probs": data["classifier_probs"][train_idx],
        "specialist_features": data["specialist_features"][train_idx],
        "route_id": data["route_ids"][train_idx],
    }
    val_inputs = {
        "image": data["images"][val_idx],
        "classifier_probs": data["classifier_probs"][val_idx],
        "specialist_features": data["specialist_features"][val_idx],
        "route_id": data["route_ids"][val_idx],
    }
    train_outputs = {
        "trust": data["y_trust"][train_idx],
        "action": data["y_action"][train_idx],
    }
    val_outputs = {
        "trust": data["y_trust"][val_idx],
        "action": data["y_action"][val_idx],
    }

    classes, _, _ = load_class_info()
    model = build_hvf_net(num_classes=len(classes))

    callbacks = [
        EarlyStopping(monitor="val_trust_auc", mode="max", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_trust_loss", mode="min", factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint(
            filepath=str(MODEL_OUTPUT),
            monitor="val_trust_auc",
            mode="max",
            save_best_only=True,
        ),
    ]

    print("Starting HVF-Net training...")
    history = model.fit(
        train_inputs,
        train_outputs,
        validation_data=(val_inputs, val_outputs),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    plot_history(history, PLOT_OUTPUT)
    save_training_history(history, HISTORY_JSON_OUTPUT, HISTORY_CSV_OUTPUT)

    metrics = {
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "final_trust_accuracy": float(history.history["trust_accuracy"][-1]),
        "final_val_trust_accuracy": float(history.history["val_trust_accuracy"][-1]),
        "final_trust_auc": float(history.history["trust_auc"][-1]),
        "final_val_trust_auc": float(history.history["val_trust_auc"][-1]),
        "final_action_accuracy": float(history.history["action_accuracy"][-1]),
        "final_val_action_accuracy": float(history.history["val_action_accuracy"][-1]),
        "model_path": str(MODEL_OUTPUT),
    }
    with open(METRICS_OUTPUT, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nTraining complete.")
    print(f"Model saved to: {MODEL_OUTPUT}")
    print(f"Plot saved to: {PLOT_OUTPUT}")
    print(f"Metrics saved to: {METRICS_OUTPUT}")
    print(f"Epoch history (JSON): {HISTORY_JSON_OUTPUT}")
    print(f"Epoch history (CSV): {HISTORY_CSV_OUTPUT}")


if __name__ == "__main__":
    main()
