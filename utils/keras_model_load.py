"""
Load .keras models saved with newer Keras (e.g. ``quantization_config`` on ``Dense``)
into environments whose ``Dense`` layer does not accept that argument.

Also strips ``shared_object_id`` from serialized ``DTypePolicy`` blobs when needed.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import zipfile
from typing import Any, Callable


def _sanitize_config_tree(obj: Any) -> Any:
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        # Older loaders choke on object-id handles embedded in newer Keras JSON
        obj.pop("shared_object_id", None)
        if obj.get("class_name") in ("DTypePolicy",):
            pass  # already popped shared_object_id
        for k in list(obj.keys()):
            obj[k] = _sanitize_config_tree(obj[k])
        return obj
    if isinstance(obj, list):
        return [_sanitize_config_tree(x) for x in obj]
    return obj


def _keras_zip_with_sanitized_config(src: pathlib.Path) -> pathlib.Path:
    """Write a new .keras zip next to temp; identical weights, cleaned config.json."""
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="keras_cfg_"))
    dst = tmpdir / "sanitized.keras"
    with zipfile.ZipFile(src, "r") as zin:
        with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                data = zin.read(info.filename)
                if info.filename == "config.json":
                    cfg = json.loads(data.decode("utf-8"))
                    cfg = _sanitize_config_tree(cfg)
                    data = json.dumps(cfg, separators=(",", ":")).encode("utf-8")
                # Preserve member name; minimal ZipInfo avoids Windows path quirks
                zout.writestr(info.filename, data)
    return dst


def _apply_dense_quantization_shim() -> None:
    """Pop ``quantization_config`` before ``Dense.__init__`` runs (version skew)."""
    import tensorflow as tf

    def _patch(Dense):
        if getattr(Dense, "_streamlit_quant_shim", False):
            return
        _orig_init: Callable = Dense.__init__

        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            kwargs.pop("quantization_config", None)
            return _orig_init(self, *args, **kwargs)

        Dense.__init__ = __init__  # type: ignore[assignment]
        Dense._streamlit_quant_shim = True

    _patch(tf.keras.layers.Dense)
    try:
        import keras as pure_keras  # type: ignore

        if pure_keras.layers.Dense is not tf.keras.layers.Dense:
            _patch(pure_keras.layers.Dense)
    except Exception:
        pass


def load_keras_model_file(path_str: str) -> Any:
    """
    ``tf.keras.models.load_model`` with ``compile=False`` and compatibility shims.

    Raises if all strategies fail.
    """
    import tensorflow as tf

    path = pathlib.Path(path_str)
    _apply_dense_quantization_shim()
    load_model = tf.keras.models.load_model

    def _load_one(p: pathlib.Path):
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                model = load_model(str(p), compile=False, safe_mode=False)
            except TypeError:
                model = load_model(str(p), compile=False)
        for w in caught:
            print(f"[CNN load warning] {w.category.__name__}: {w.message}", flush=True)
        return model

    first_exc: Exception | None = None
    try:
        return _load_one(path)
    except Exception as exc:
        first_exc = exc
        print(f"[CNN load] first attempt failed: {exc}", flush=True)

    if path.suffix.lower() == ".keras":
        try:
            sanitized = _keras_zip_with_sanitized_config(path)
            print(
                "[CNN] Retrying with sanitized config.json (removed stale Keras fields).",
                flush=True,
            )
            return _load_one(sanitized)
        except Exception as exc2:
            if first_exc is not None:
                raise exc2 from first_exc
            raise

    if first_exc is not None:
        raise first_exc
    raise RuntimeError("load_keras_model_file: unreachable")

