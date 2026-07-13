"""Shared hybrid pipeline helpers for HVF-Net training and inference."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

from model_loader import load_compatible_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
CLASS_INFO_PATH = MODELS_DIR / "class_info.json"

SUPPORTED_CLASSES = {"brain_mri", "chest_xray", "retina_oct"}
UNSUPPORTED_CLASSES = {"dental_xray", "skin_image", "natural_image"}

ROUTE_TO_ID = {
    "brain_mri": 0,
    "chest_xray": 1,
    "retina_oct": 2,
    "none": 3,
}

DISEASE_POSITIVE = {
    ("brain_mri", "tumor"): True,
    ("brain_mri", "no_tumor"): False,
    ("chest_xray", "pneumonia"): True,
    ("chest_xray", "normal"): False,
    ("retina_oct", "cnv"): True,
    ("retina_oct", "normal"): False,
}

DISPLAY_NAMES = {
    "brain_mri": "Brain MRI",
    "chest_xray": "Chest X-ray",
    "retina_oct": "Retina OCT",
    "dental_xray": "Dental X-ray",
    "skin_image": "Skin Image",
    "natural_image": "Natural Image",
}


def load_class_info():
    if CLASS_INFO_PATH.exists():
        with open(CLASS_INFO_PATH, "r", encoding="utf-8") as file:
            info = json.load(file)
        classes = info["classes"]
        display_names = DISPLAY_NAMES.copy()
        display_names.update(info.get("display_names", {}))
        supported = set(info.get("supported_classes", list(SUPPORTED_CLASSES)))
        return classes, display_names, supported
    classes = [
        "brain_mri",
        "chest_xray",
        "dental_xray",
        "natural_image",
        "retina_oct",
        "skin_image",
    ]
    return classes, DISPLAY_NAMES.copy(), SUPPORTED_CLASSES.copy()


def load_first_model(candidates, name):
    for path in candidates:
        if os.path.exists(path):
            print(f"Loading {name}: {path}")
            return load_compatible_model(path, compile=False)
    raise FileNotFoundError(f"{name} model not found. Tried: {candidates}")


def load_all_models():
    classifier = load_first_model([
        str(MODELS_DIR / "classifier_model.h5"),
        str(MODELS_DIR / "classifier_model.keras"),
    ], "Classifier")
    brain = load_first_model([
        str(MODELS_DIR / "best_brain_tumor_model.keras"),
        str(MODELS_DIR / "best_brain_tumor_MobileNetV2.keras"),
        str(MODELS_DIR / "brain_tumor_model.keras"),
    ], "Brain")
    pneumonia = load_first_model([
        str(MODELS_DIR / "best_model.keras"),
        str(MODELS_DIR / "best_pneumonia_model.keras"),
        str(MODELS_DIR / "final_pneumonia_model.keras"),
    ], "Pneumonia")
    eye = load_first_model([
        str(MODELS_DIR / "best_eye_model.keras"),
        str(MODELS_DIR / "best_eye_disease_EfficientNetB0.keras"),
        str(MODELS_DIR / "final_eye_disease_EfficientNetB0.keras"),
    ], "Eye")
    return classifier, brain, pneumonia, eye


def preprocess_classifier_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def preprocess_brain_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)).astype(np.float32)
    return np.expand_dims(preprocess_input(img), axis=0)


def preprocess_pneumonia_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2RGB).astype(np.float32)
    return np.expand_dims(preprocess_input(clahe_img), axis=0)


def preprocess_retina_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224)).astype(np.float32)
    return np.expand_dims(preprocess_input(img), axis=0)


def run_classifier(classifier_model, classes, image_path: str):
    image = preprocess_classifier_image(image_path)
    probs = classifier_model.predict(image, verbose=0)[0]
    class_index = int(np.argmax(probs))
    predicted_route = classes[class_index]
    confidence = float(probs[class_index]) * 100.0
    return predicted_route, confidence, probs


def disease_detected(route: str, raw_score: float) -> bool:
    if route == "retina_oct":
        return raw_score <= 0.5
    return raw_score > 0.5


def run_specialist(route: str, image_path: str, brain_model, pneumonia_model, eye_model):
    if route == "brain_mri":
        score = float(brain_model.predict(preprocess_brain_image(image_path), verbose=0)[0][0])
    elif route == "chest_xray":
        score = float(pneumonia_model.predict(preprocess_pneumonia_image(image_path), verbose=0)[0][0])
    elif route == "retina_oct":
        score = float(eye_model.predict(preprocess_retina_image(image_path), verbose=0)[0][0])
    else:
        return 0.0, 0.0, 0.0

    positive = 1.0 if disease_detected(route, score) else 0.0
    if route == "retina_oct":
        confidence = (1.0 - score) if positive else score
    else:
        confidence = score if positive else (1.0 - score)
    return score, confidence, positive


def compute_labels(
    gt_route: str,
    gt_diagnosis: str,
    predicted_route: str,
    classifier_confidence: float,
    specialist_score: float,
    specialist_confidence: float,
    specialist_ran: bool,
):
    cls_conf = classifier_confidence / 100.0
    spec_conf = specialist_confidence

    if gt_route in UNSUPPORTED_CLASSES:
        if predicted_route in SUPPORTED_CLASSES:
            return 0, 1
        if predicted_route == gt_route and cls_conf >= 0.5:
            return 1, 0
        if predicted_route == gt_route:
            return 0, 1
        return 0, 1

    correct_route = predicted_route == gt_route
    if not correct_route:
        return (0, 2) if cls_conf >= 0.5 else (0, 1)

    if not specialist_ran:
        return 0, 1

    gt_positive = DISEASE_POSITIVE[(gt_route, gt_diagnosis)]
    pred_positive = disease_detected(gt_route, specialist_score)

    if gt_positive == pred_positive and spec_conf >= 0.5:
        if spec_conf >= 0.8 and cls_conf >= 0.7:
            return 1, 0
        return 1, 0

    if spec_conf < 0.5:
        return 0, 1
    return 0, 2


def action_name(action_id: int) -> str:
    return {0: "accept", 1: "reject", 2: "re-route"}[int(action_id)]
