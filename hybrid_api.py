from flask import Flask, request, jsonify
from tensorflow.keras.applications.efficientnet import preprocess_input

import json
import numpy as np
import cv2
import os

from model_loader import load_compatible_model
from hvf_net.predict import load_hvf_model, verify_prediction
from hvf_net.pipeline import disease_detected

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

print("Loading HVF-Net supervisor...")
hvf_model = load_hvf_model()
if hvf_model is not None:
    print("HVF-Net loaded: models/hvf_net_model.keras")
else:
    print("WARNING: HVF-Net model not found. API will run without verification.")

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

SPECIALIST_MODEL_NAMES = {
    "brain_mri": "Brain Tumor AI Model",
    "chest_xray": "Pneumonia AI Model",
    "retina_oct": "Eye Disease AI Model",
}

SCAN_TYPE_NAMES = {
    "brain_mri": "Brain MRI",
    "chest_xray": "Chest X-ray",
    "retina_oct": "Retina OCT",
}

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

    return image_type, display_name, confidence, prediction[0]


def specialist_features(image_type, score):
    positive = 1.0 if disease_detected(image_type, score) else 0.0
    if image_type == "retina_oct":
        spec_conf = (1.0 - score) if positive else score
    else:
        spec_conf = score if positive else (1.0 - score)
    return positive, spec_conf


def build_diagnosis(image_type, score):
    if image_type == "brain_mri":
        if score > 0.5:
            confidence = round(score * 100, 2)
            return {
                "result": "YES",
                "diagnosis": "Brain Tumor",
                "confidence": confidence,
                "severity": get_severity(confidence),
                "message": "Brain MRI detected. Brain tumor analysis completed.",
            }
        confidence = round((1 - score) * 100, 2)
        return {
            "result": "NO",
            "diagnosis": "No Brain Tumor",
            "confidence": confidence,
            "severity": "Normal",
            "message": "Brain MRI detected. No brain tumor signs were found.",
        }

    if image_type == "chest_xray":
        if score > 0.5:
            confidence = round(score * 100, 2)
            return {
                "result": "YES",
                "diagnosis": "Pneumonia",
                "confidence": confidence,
                "severity": get_severity(confidence),
                "message": "Chest X-ray detected. Pneumonia analysis completed.",
            }
        confidence = round((1 - score) * 100, 2)
        return {
            "result": "NO",
            "diagnosis": "Normal Chest X-Ray",
            "confidence": confidence,
            "severity": "Normal",
            "message": "Chest X-ray detected. No pneumonia signs were found.",
        }

    # retina_oct (inverted score: high = normal)
    if score > 0.5:
        confidence = round(score * 100, 2)
        return {
            "result": "NO",
            "diagnosis": "Normal Retina",
            "confidence": confidence,
            "severity": "Normal",
            "message": "Retina OCT detected. No eye disease signs were found.",
        }
    confidence = round((1 - score) * 100, 2)
    return {
        "result": "YES",
        "diagnosis": "CNV Disease",
        "confidence": confidence,
        "severity": get_severity(confidence),
        "message": "Retina OCT detected. Eye disease analysis completed.",
    }


def run_specialist_raw(image_type, image_path):
    if image_type == "brain_mri":
        score = float(brain_model.predict(preprocess_brain(image_path), verbose=0)[0][0])
    elif image_type == "chest_xray":
        score = float(pneumonia_model.predict(preprocess_pneumonia(image_path), verbose=0)[0][0])
    elif image_type == "retina_oct":
        score = float(eye_model.predict(preprocess_retina(image_path), verbose=0)[0][0])
    else:
        raise ValueError(f"Unsupported specialist route: {image_type}")
    return score


def build_specialist_response(image_type, score, classifier_confidence):
    diagnosis = build_diagnosis(image_type, score)
    return {
        "success": True,
        "supported": True,
        "uploaded_scan_type": SCAN_TYPE_NAMES[image_type],
        "classifier_confidence": f"{classifier_confidence}%",
        "selected_model": SPECIALIST_MODEL_NAMES[image_type],
        "result": diagnosis["result"],
        "diagnosis": diagnosis["diagnosis"],
        "confidence": f"{diagnosis['confidence']}%",
        "severity": diagnosis["severity"],
        "message": diagnosis["message"],
        "route": image_type,
        "raw_score": score,
    }


def pick_reroute_candidate(classifier_probs, current_route):
    """Next-best supported class from classifier probabilities (excluding current)."""
    ranked = sorted(
        (
            (classifier_classes[i], float(classifier_probs[i]))
            for i in range(len(classifier_classes))
            if classifier_classes[i] in SUPPORTED_CLASSES
            and classifier_classes[i] != current_route
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[0][0] if ranked else None


def apply_hvf_verification(
    response_dict,
    image_path,
    image_type,
    classifier_probs,
    score,
    allow_reroute=True,
):
    """Run HVF-Net and apply accept / reject / re-route to the API response."""
    positive, spec_conf = specialist_features(image_type, score)

    hvf = verify_prediction(
        image_path=image_path,
        classifier_probs=classifier_probs,
        specialist_score=score,
        specialist_confidence=spec_conf,
        specialist_positive=positive,
        predicted_route=image_type,
    )
    response_dict["hvf_net"] = hvf

    if not hvf.get("available"):
        response_dict["hvf_available"] = False
        response_dict["verified"] = None
        return response_dict

    response_dict["hvf_available"] = True
    response_dict["trust_score"] = hvf["trust_score"]
    response_dict["hvf_action"] = hvf["action"]

    action = hvf["action"]

    if action == "accept":
        response_dict["verified"] = True
        response_dict["message"] = (
            f"{response_dict['message']} HVF-Net verified this result "
            f"(trust={hvf['trust_score']:.2f})."
        )
        return response_dict

    if action == "reject":
        response_dict["verified"] = False
        response_dict["message"] = hvf["message"]
        response_dict["warning"] = (
            "HVF-Net rejected this prediction due to low trust. "
            "Please consult a doctor and do not treat this as a final diagnosis."
        )
        return response_dict

    # action == "re-route"
    response_dict["verified"] = False
    response_dict["rerouted"] = False

    if not allow_reroute:
        response_dict["message"] = (
            "HVF-Net suggested re-routing, but no further alternate route was tried. "
            + hvf["message"]
        )
        return response_dict

    alternate = pick_reroute_candidate(classifier_probs, image_type)
    if alternate is None:
        response_dict["message"] = (
            "HVF-Net detected a possible routing mismatch, but no alternate "
            "supported route was available. " + hvf["message"]
        )
        return response_dict

    print(f"HVF-Net re-route: {image_type} → {alternate}")
    alt_score = run_specialist_raw(alternate, image_path)
    alt_response = build_specialist_response(
        alternate,
        alt_score,
        round(float(classifier_probs[classifier_classes.index(alternate)]) * 100, 2),
    )
    alt_response["original_route"] = image_type
    alt_response["original_diagnosis"] = response_dict.get("diagnosis")
    alt_response["rerouted"] = True
    alt_response["reroute_from"] = SCAN_TYPE_NAMES.get(image_type, image_type)
    alt_response["reroute_to"] = SCAN_TYPE_NAMES.get(alternate, alternate)
    alt_response["classifier_confidence"] = response_dict["classifier_confidence"]

    # Re-verify once after re-route (no nested re-route loops)
    return apply_hvf_verification(
        alt_response,
        image_path,
        alternate,
        classifier_probs,
        alt_score,
        allow_reroute=False,
    )


def unsupported_response(display_name, classifier_confidence, image_path=None, classifier_probs=None):
    payload = {
        "success": True,
        "supported": False,
        "detected_image_type": display_name,
        "classifier_confidence": f"{classifier_confidence}%",
        "supported_scan_types": list(SUPPORTED_SCAN_TYPES),
        "message": UNSUPPORTED_MESSAGE_TEMPLATE.format(image_type=display_name),
    }

    # Optional HVF check for unsupported images (route = none)
    if image_path is not None and classifier_probs is not None and hvf_model is not None:
        hvf = verify_prediction(
            image_path=image_path,
            classifier_probs=classifier_probs,
            specialist_score=0.0,
            specialist_confidence=0.0,
            specialist_positive=0.0,
            predicted_route="none",
        )
        payload["hvf_net"] = hvf
        payload["hvf_available"] = bool(hvf.get("available"))
        if hvf.get("available"):
            payload["trust_score"] = hvf["trust_score"]
            payload["hvf_action"] = hvf["action"]

    return jsonify(payload), 200


def run_with_hvf(image_type, image_path, classifier_confidence, classifier_probs):
    score = run_specialist_raw(image_type, image_path)
    response = build_specialist_response(image_type, score, classifier_confidence)
    response = apply_hvf_verification(
        response,
        image_path,
        image_type,
        classifier_probs,
        score,
        allow_reroute=True,
    )
    # Internal fields — keep API clean for clients that only need clinical fields
    response.pop("raw_score", None)
    response.pop("route", None)
    return jsonify(response)


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
        "models": {
            "classifier": True,
            "brain": True,
            "pneumonia": True,
            "eye": True,
            "hvf_net": hvf_model is not None,
        },
        "hvf_net": hvf_model is not None,
        "message": (
            "Upload an image to /hybrid-predict. "
            "Supported types: Brain MRI, Chest X-ray, and Retina OCT. "
            "HVF-Net verifies each diagnosis (accept / reject / re-route)."
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
        image_type, display_name, classifier_confidence, classifier_probs = classify_image_type(image_path)

        if image_type not in SUPPORTED_CLASSES:
            return unsupported_response(
                display_name,
                classifier_confidence,
                image_path=image_path,
                classifier_probs=classifier_probs,
            )

        if image_type in SUPPORTED_CLASSES:
            return run_with_hvf(
                image_type,
                image_path,
                classifier_confidence,
                classifier_probs,
            )

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
