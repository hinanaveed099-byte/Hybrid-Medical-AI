from flask import Flask, request, jsonify
from tensorflow.keras.applications.efficientnet import preprocess_input

import json
import numpy as np
import cv2
import os

from model_loader import load_compatible_model

app = Flask(__name__)

SUPPORTED_SCAN_TYPES = ("Brain MRI", "Chest X-ray", "Retina OCT")

UNSUPPORTED_MESSAGE_TEMPLATE = (
    "This appears to be a {image_type}. "
    "Our system does not support this image type. "
    "You can upload Brain MRI, Chest X-ray, or Retina OCT images "
    "for disease detection in this system."
)

# ======================
# LOAD MODELS
# ======================


def load_first_model(candidates, name):
    for path in candidates:
        if os.path.exists(path):
            print(f"Loading {name}: {path}")
            return load_compatible_model(path, compile=False)
    raise FileNotFoundError(
        f"{name} model not found. Tried: {candidates}. "
        "Place the model files in the models/ folder."
    )


def load_classifier_model():
    return load_first_model([
        "models/classifier_model.h5",
        "models/classifier_model.keras",
    ], "Classifier")


classifier_model = load_classifier_model()

brain_model = load_first_model([
    "models/best_brain_tumor_model.keras",
    "models/best_brain_tumor_MobileNetV2.keras",
    "models/brain_tumor_model.keras",
    "models/best_brain_tumor_EfficientNetB0.keras",
    "models/final_brain_tumor_EfficientNetB0.keras",
], "Brain")

pneumonia_model = load_first_model([
    "models/best_model.keras",
    "models/best_pneumonia_model.keras",
    "models/best_pneumonia_MobileNetV2.keras",
    "models/final_pneumonia_model.keras",
], "Pneumonia")

eye_model = load_first_model([
    "models/best_eye_model.keras",
    "models/best_eye_disease_EfficientNetB0.keras",
    "models/final_eye_disease_EfficientNetB0.keras",
], "Eye")

# ======================
# CLASSIFIER CLASSES (6-class)
# ======================

CLASS_INFO_PATH = "models/class_info.json"

DISPLAY_NAMES = {
    "brain_mri": "Brain MRI",
    "chest_xray": "Chest X-ray",
    "retina_oct": "Retina OCT",
    "dental_xray": "Dental X-ray",
    "skin_image": "Skin Image",
    "natural_image": "Natural Image",
}

SUPPORTED_CLASSES = {"brain_mri", "chest_xray", "retina_oct"}

if os.path.exists(CLASS_INFO_PATH):
    with open(CLASS_INFO_PATH, "r", encoding="utf-8") as f:
        class_info = json.load(f)
    classifier_classes = class_info["classes"]
    DISPLAY_NAMES.update(class_info.get("display_names", {}))
    SUPPORTED_CLASSES = set(class_info.get("supported_classes", list(SUPPORTED_CLASSES)))
else:
    classifier_classes = [
        "brain_mri",
        "chest_xray",
        "dental_xray",
        "natural_image",
        "retina_oct",
        "skin_image",
    ]

# ======================
# PREPROCESS FUNCTIONS
# ======================


def preprocess_classifier(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Unable to read the uploaded image. Please upload a valid image file.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)


def preprocess_brain(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)


def preprocess_retina(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)


def clahe_preprocessing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    clahe_img = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
    clahe_img = clahe_img.astype(np.float32)
    return preprocess_input(clahe_img)


def preprocess_pneumonia(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = clahe_preprocessing(img)
    return np.expand_dims(img, axis=0)


def get_severity(confidence):
    if confidence < 50:
        return "Mild"
    if confidence < 80:
        return "Moderate"
    return "Severe"


def classify_image_type(image_path):
    classifier_img = preprocess_classifier(image_path)
    prediction = classifier_model.predict(classifier_img, verbose=0)

    class_index = int(np.argmax(prediction))
    image_type = classifier_classes[class_index]
    confidence = round(float(prediction[0][class_index]) * 100, 2)
    display_name = DISPLAY_NAMES.get(image_type, image_type.replace("_", " ").title())

    return image_type, display_name, confidence


def unsupported_response(display_name, classifier_confidence):
    return jsonify({
        "success": True,
        "supported": False,
        "detected_image_type": display_name,
        "classifier_confidence": f"{classifier_confidence}%",
        "supported_scan_types": list(SUPPORTED_SCAN_TYPES),
        "message": UNSUPPORTED_MESSAGE_TEMPLATE.format(image_type=display_name),
    }), 200


def run_brain_model(image_path, classifier_confidence):
    img = preprocess_brain(image_path)
    score = float(brain_model.predict(img, verbose=0)[0][0])

    if score > 0.5:
        diagnosis = "Brain Tumor"
        confidence = round(score * 100, 2)
        severity = get_severity(confidence)
        answer = "YES"
        message = "Brain MRI detected. Brain tumor analysis completed."
    else:
        diagnosis = "No Brain Tumor"
        confidence = round((1 - score) * 100, 2)
        severity = "Normal"
        answer = "NO"
        message = "Brain MRI detected. No brain tumor signs were found."

    return jsonify({
        "success": True,
        "supported": True,
        "uploaded_scan_type": "Brain MRI",
        "classifier_confidence": f"{classifier_confidence}%",
        "selected_model": "Brain Tumor AI Model",
        "result": answer,
        "diagnosis": diagnosis,
        "confidence": f"{confidence}%",
        "severity": severity,
        "message": message,
    })


def run_pneumonia_model(image_path, classifier_confidence):
    img = preprocess_pneumonia(image_path)
    score = float(pneumonia_model.predict(img, verbose=0)[0][0])

    if score > 0.5:
        diagnosis = "Pneumonia"
        confidence = round(score * 100, 2)
        severity = get_severity(confidence)
        answer = "YES"
        message = "Chest X-ray detected. Pneumonia analysis completed."
    else:
        diagnosis = "Normal Chest X-Ray"
        confidence = round((1 - score) * 100, 2)
        severity = "Normal"
        answer = "NO"
        message = "Chest X-ray detected. No pneumonia signs were found."

    return jsonify({
        "success": True,
        "supported": True,
        "uploaded_scan_type": "Chest X-ray",
        "classifier_confidence": f"{classifier_confidence}%",
        "selected_model": "Pneumonia AI Model",
        "result": answer,
        "diagnosis": diagnosis,
        "confidence": f"{confidence}%",
        "severity": severity,
        "message": message,
    })


def run_eye_model(image_path, classifier_confidence):
    img = preprocess_retina(image_path)
    score = float(eye_model.predict(img, verbose=0)[0][0])

    if score > 0.5:
        diagnosis = "Normal Retina"
        confidence = round(score * 100, 2)
        severity = "Normal"
        answer = "NO"
        message = "Retina OCT detected. No eye disease signs were found."
    else:
        diagnosis = "CNV Disease"
        confidence = round((1 - score) * 100, 2)
        severity = get_severity(confidence)
        answer = "YES"
        message = "Retina OCT detected. Eye disease analysis completed."

    return jsonify({
        "success": True,
        "supported": True,
        "uploaded_scan_type": "Retina OCT",
        "classifier_confidence": f"{classifier_confidence}%",
        "selected_model": "Eye Disease AI Model",
        "result": answer,
        "diagnosis": diagnosis,
        "confidence": f"{confidence}%",
        "severity": severity,
        "message": message,
    })


# ======================
# ROUTES
# ======================


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "Hybrid Medical AI API is running.",
        "endpoint": "/hybrid-predict",
        "method": "POST",
        "supported_scan_types": list(SUPPORTED_SCAN_TYPES),
        "message": (
            "Upload an image to /hybrid-predict. "
            "Supported types: Brain MRI, Chest X-ray, and Retina OCT."
        ),
    })


@app.route("/hybrid-predict", methods=["POST"])
def hybrid_predict():
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image uploaded. Please send an image file in the 'image' field.",
        }), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({
            "success": False,
            "error": "Empty filename. Please upload a valid image file.",
        }), 400

    image_path = "temp.jpg"
    file.save(image_path)

    try:
        image_type, display_name, classifier_confidence = classify_image_type(image_path)

        if image_type not in SUPPORTED_CLASSES:
            return unsupported_response(display_name, classifier_confidence)

        if image_type == "brain_mri":
            return run_brain_model(image_path, classifier_confidence)

        if image_type == "chest_xray":
            return run_pneumonia_model(image_path, classifier_confidence)

        if image_type == "retina_oct":
            return run_eye_model(image_path, classifier_confidence)

        return jsonify({
            "success": False,
            "error": "Unable to process this image type.",
            "detected_image_type": display_name,
        }), 400

    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
