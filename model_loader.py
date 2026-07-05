"""Load models saved with newer Keras on older local installs."""

import json
import os
import shutil
import tempfile
import zipfile

import h5py
from tensorflow.keras.models import load_model


def _strip_unsupported_keys(obj):
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        for value in obj.values():
            _strip_unsupported_keys(value)
    elif isinstance(obj, list):
        for item in obj:
            _strip_unsupported_keys(item)


def _patch_json_attrs(h5_file):
    for attr_name in list(h5_file.attrs.keys()):
        val = h5_file.attrs[attr_name]
        try:
            if isinstance(val, bytes):
                val_str = val.decode("utf-8")
            elif isinstance(val, str):
                val_str = val
            else:
                continue

            if not val_str.startswith(("{", "[")):
                continue

            config = json.loads(val_str)
            _strip_unsupported_keys(config)
            new_val = json.dumps(config)
            del h5_file.attrs[attr_name]
            h5_file.attrs[attr_name] = new_val
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue


def _load_patched_h5(model_path, compile=False):
    fd, tmp_path = tempfile.mkstemp(suffix=".h5")
    os.close(fd)
    shutil.copy2(model_path, tmp_path)

    try:
        with h5py.File(tmp_path, "r+") as h5_file:
            _patch_json_attrs(h5_file)
        return load_model(tmp_path, compile=compile)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _load_patched_keras(model_path, compile=False):
    with tempfile.TemporaryDirectory() as tmp_dir:
        patched_path = os.path.join(tmp_dir, "patched_model.keras")

        with zipfile.ZipFile(model_path, "r") as zin:
            with zipfile.ZipFile(patched_path, "w", compression=zipfile.ZIP_STORED) as zout:
                for item in zin.infolist():
                    data = zin.read(item.filename)
                    if item.filename.endswith(".json"):
                        config = json.loads(data.decode("utf-8"))
                        _strip_unsupported_keys(config)
                        data = json.dumps(config).encode("utf-8")
                    zout.writestr(item, data)

        return load_model(patched_path, compile=compile)


def load_compatible_model(model_path, compile=False):
    """Load .h5 or .keras model, patching Colab/local Keras version mismatch."""
    try:
        return load_model(model_path, compile=compile)
    except (TypeError, ValueError) as exc:
        if "quantization_config" not in str(exc):
            raise

    if model_path.endswith(".h5"):
        return _load_patched_h5(model_path, compile=compile)
    if model_path.endswith(".keras"):
        return _load_patched_keras(model_path, compile=compile)

    raise RuntimeError(f"Unsupported model format: {model_path}")


# Backward compatible alias
load_keras_model = load_compatible_model
