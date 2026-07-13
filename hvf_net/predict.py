"""HVF-Net inference helpers."""

from __future__ import annotations

import os

import numpy as np

from hvf_net.layers import CrossAttentionLayer
from hvf_net.pipeline import ROUTE_TO_ID, action_name, preprocess_classifier_image
from model_loader import load_compatible_model

HVF_MODEL_CANDIDATES = [
    "models/hvf_net_model.keras",
    "models/hvf_net_model.h5",
]

_hvf_model = None


def load_hvf_model():
    global _hvf_model
    if _hvf_model is not None:
        return _hvf_model

    for path in HVF_MODEL_CANDIDATES:
        if os.path.exists(path):
            _hvf_model = load_compatible_model(
                path,
                compile=False,
                custom_objects={"CrossAttentionLayer": CrossAttentionLayer},
            )
            return _hvf_model
    return None


def verify_prediction(
    image_path: str,
    classifier_probs: np.ndarray,
    specialist_score: float,
    specialist_confidence: float,
    specialist_positive: float,
    predicted_route: str,
):
    model = load_hvf_model()
    if model is None:
        return {
            "available": False,
            "trust_score": None,
            "action": None,
            "action_name": None,
            "message": "HVF-Net model not loaded.",
        }

    route_key = predicted_route if predicted_route in ROUTE_TO_ID else "none"
    route_id = np.array([[ROUTE_TO_ID[route_key]]], dtype=np.int32)

    image = preprocess_classifier_image(image_path)
    classifier_probs = np.asarray(classifier_probs, dtype=np.float32).reshape(1, -1)
    specialist_features = np.array(
        [[specialist_score, specialist_confidence, specialist_positive]],
        dtype=np.float32,
    )

    trust_pred, action_pred = model.predict(
        {
            "image": image,
            "classifier_probs": classifier_probs,
            "specialist_features": specialist_features,
            "route_id": route_id,
        },
        verbose=0,
    )

    trust_score = float(trust_pred[0][0])
    action_id = int(np.argmax(action_pred[0]))
    action = action_name(action_id)

    if action == "accept":
        message = "HVF-Net verified this result with high trust."
    elif action == "reject":
        message = "HVF-Net marked this result as uncertain. Please consult a doctor."
    else:
        message = "HVF-Net detected a possible routing mismatch."

    return {
        "available": True,
        "trust_score": round(trust_score, 4),
        "action": action,
        "action_name": action,
        "message": message,
    }
