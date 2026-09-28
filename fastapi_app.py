from typing import List, Dict, Any, Optional
import os
import inspect
import joblib
import pandas as pd
import numpy as np
import configparser
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Body, Query
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from starlette.responses import JSONResponse
from pathlib import Path

# Determine model path: precedence -> ENV MODEL_PATH > config.ini MODEL_PATH > default

# configure logger for messages (minimal)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _sanitize_for_json(obj: Any):
    """Recursively convert numpy/pandas types to plain Python types for JSON serialization."""
    import numpy as _np
    import pandas as _pd

    # Primitives
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int, float)):
        return obj

    # Numpy scalars
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)

    # Numpy generic
    try:
        if isinstance(obj, _np.generic):
            return obj.item()
    except Exception:
        pass

    # Numpy arrays
    if isinstance(obj, _np.ndarray):
        return [_sanitize_for_json(x) for x in obj.tolist()]

    # Pandas types
    try:
        if isinstance(obj, _pd.Timestamp) or isinstance(obj, _pd.Timedelta):
            return str(obj)
        if obj is _pd.NA:
            return None
        # pandas Series or Index
        if isinstance(obj, (_pd.Series, _pd.Index)):
            return [_sanitize_for_json(x) for x in obj.tolist()]
        # pandas DataFrame -> to records
        if isinstance(obj, _pd.DataFrame):
            try:
                return [_sanitize_for_json(r) for r in obj.to_dict(orient="records")]
            except Exception:
                return str(obj)
    except Exception:
        pass

    # Mapping types
    from collections.abc import Mapping, Sequence

    if isinstance(obj, Mapping):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}

    # Sequences
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [_sanitize_for_json(x) for x in obj]

    # Fallback: try to convert via tolist/to_dict/item, else stringify
    try:
        if hasattr(obj, "to_dict"):
            return _sanitize_for_json(obj.to_dict())
    except Exception:
        pass
    try:
        if hasattr(obj, "tolist"):
            return _sanitize_for_json(obj.tolist())
    except Exception:
        pass
    try:
        if hasattr(obj, "item"):
            return obj.item()
    except Exception:
        pass

    return str(obj)

# Read config.ini (if present) to allow MODEL_PATH and MODEL_FILENAME customization
_env_model = os.environ.get("MODEL_PATH")
MODEL_PATH = None
MODEL_FILENAME = "random_forest_pipeline.pkl"
try:
    cfg_path = Path(__file__).parent / "config.ini"
    if cfg_path.exists():
        _cfg = configparser.ConfigParser()
        _cfg.read(cfg_path)
        _section = _cfg["ml"] if "ml" in _cfg else _cfg["DEFAULT"] if "DEFAULT" in _cfg else None
        if _section is not None:
            # optional custom filename inside a directory
            MODEL_FILENAME = _section.get("MODEL_FILENAME", MODEL_FILENAME)
            _cfg_model = _section.get("MODEL_PATH", None)
        else:
            _cfg_model = None
    else:
        _cfg_model = None
except Exception:
    _cfg_model = None

# Determine final MODEL_PATH with precedence and support for directory+filename
MODEL_SOURCE = "default"
if _env_model:
    MODEL_SOURCE = "env"
    env_p = Path(_env_model)
    if env_p.exists() and env_p.is_dir():
        candidate = env_p / MODEL_FILENAME
        if candidate.exists():
            MODEL_PATH = str(candidate)
        else:
            # keep as directory so load_pipeline can raise a clear error later
            MODEL_PATH = str(env_p)
    else:
        MODEL_PATH = str(env_p)
elif _cfg_model:
    MODEL_SOURCE = "config"
    cfg_p = Path(_cfg_model)
    if cfg_p.exists() and cfg_p.is_dir():
        candidate = cfg_p / MODEL_FILENAME
        if candidate.exists():
            MODEL_PATH = str(candidate)
        else:
            MODEL_PATH = str(cfg_p)
    else:
        MODEL_PATH = str(cfg_p)
else:
    # fallback to package-local filename
    MODEL_PATH = str(Path(__file__).parent / MODEL_FILENAME)

logger.info("MODEL_PATH resolved from %s: %s", MODEL_SOURCE, MODEL_PATH)

RELOAD_KEY = os.environ.get("RELOAD_KEY", "changeme")  # simple reload protection; use proper auth in prod

app = FastAPI(title="RF Prediction Service")


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]


def load_pipeline(path: str):
    p = Path(path)
    # If user provided a directory, try a sensible default filename inside it
    if p.exists() and p.is_dir():
        candidate = p / "random_forest_pipeline.pkl"
        if candidate.exists() and candidate.is_file():
            p = candidate
        else:
            raise FileNotFoundError(
                f"MODEL_PATH points to a directory ({path}) but no 'random_forest_pipeline.pkl' was found inside. "
                "Set MODEL_PATH to the full pipeline file path or place the pipeline file inside the directory."
            )

    if not p.exists():
        raise FileNotFoundError(f"Pipeline file not found: {path}")

    if not p.is_file():
        raise FileNotFoundError(f"Pipeline path is not a file: {path}")

    try:
        return joblib.load(p)
    except PermissionError as e:
        # Surface a clearer error message for permission issues
        raise PermissionError(f"Permission denied while loading pipeline '{p}': {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load pipeline '{p}': {e}")


def _extract_model_info(pipeline, model_path: str) -> Dict[str, Any]:
    """Extract lightweight metadata from a trained pipeline for the API.

    Returns dict with keys: classes, has_proba, has_decision_function,
    feature_names (list, may be truncated), feature_importances (top list)
    """
    info: Dict[str, Any] = {}
    try:
        # classes
        cls = None
        try:
            cls = pipeline.named_steps.get("classifier").classes_
        except Exception:
            # pipeline may expose classes_ directly
            cls = getattr(pipeline, "classes_", None)
        info["classes"] = list(cls) if hasattr(cls, "tolist") or isinstance(cls, (list, tuple)) else cls

        info["has_proba"] = hasattr(pipeline, "predict_proba")
        info["has_decision_function"] = hasattr(pipeline, "decision_function")

        # Attempt to derive encoded feature names from preprocessor
        feature_names = None
        try:
            pre = pipeline.named_steps.get("preprocessor") if hasattr(pipeline, "named_steps") else None
            if pre is not None:
                # preferred: get_feature_names_out if available
                try:
                    sig = inspect.signature(pre.get_feature_names_out)
                    if len(sig.parameters) == 0:
                        feature_names = list(pre.get_feature_names_out())
                    else:
                        # no input feature names available; attempt transformers_
                        raise Exception("needs input names")
                except Exception:
                    # build names from transformers_
                    feature_names = []
                    try:
                        for name, trans, cols in pre.transformers_:
                            if trans == "drop":
                                continue
                            if trans == "passthrough":
                                if isinstance(cols, (list, tuple)):
                                    feature_names.extend(list(cols))
                                else:
                                    feature_names.append(cols)
                                continue

                            t = trans
                            if hasattr(trans, "named_steps"):
                                try:
                                    t = list(trans.named_steps.values())[-1]
                                except Exception:
                                    t = trans

                            try:
                                names = t.get_feature_names_out(cols if isinstance(cols, (list, tuple)) else [cols])
                                feature_names.extend(list(names))
                            except Exception:
                                if isinstance(cols, (list, tuple)):
                                    feature_names.extend(list(cols))
                                else:
                                    feature_names.append(cols)
                    except Exception:
                        feature_names = None
        except Exception:
            feature_names = None

        if feature_names is None:
            info["feature_names"] = None
        else:
            info["feature_names"] = feature_names[:200]  # truncate for API

        # Feature importances (tree-based)
        try:
            clf = pipeline.named_steps.get("classifier") if hasattr(pipeline, "named_steps") else None
            if clf is not None and hasattr(clf, "feature_importances_") and feature_names is not None:
                importances = clf.feature_importances_
                # align lengths
                if len(feature_names) != len(importances):
                    if len(feature_names) > len(importances):
                        feature_names = feature_names[: len(importances)]
                    else:
                        feature_names = feature_names + [f"f_{i}" for i in range(len(feature_names), len(importances))]

                df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
                df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)
                info["feature_importances_top"] = df_imp.head(15).to_dict(orient="records")
            else:
                # try to load CSV artifact next to model if present
                try:
                    csv_path = Path(model_path).with_suffix(".feature_importances.csv")
                    if csv_path.exists():
                        df_imp = pd.read_csv(csv_path)
                        info["feature_importances_top"] = df_imp.head(15).to_dict(orient="records")
                    else:
                        info["feature_importances_top"] = None
                except Exception:
                    info["feature_importances_top"] = None
        except Exception:
            info["feature_importances_top"] = None
    except Exception:
        return {}

    # Sanitize numeric and numpy types to plain Python types so JSONResponse can serialize
    def _sanitize(obj: Any):
        if obj is None:
            return None
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (list, tuple)):
            return [ _sanitize(x) for x in obj ]
        if isinstance(obj, dict):
            return { str(k): _sanitize(v) for k, v in obj.items() }
        # pandas/Numpy scalars
        if isinstance(obj, (pd.Timestamp,)):
            return str(obj)
        try:
            # numpy generic
            if isinstance(obj, np.generic):
                return obj.item()
        except Exception:
            pass
        return obj

    return _sanitize(info)


@app.on_event("startup")
def startup_load_model():
    try:
        app.state.pipeline = load_pipeline(MODEL_PATH)
        # extract metadata
        app.state.model_info = _extract_model_info(app.state.pipeline, MODEL_PATH)
        # Always sanitize model_info to plain Python types to avoid JSON serialization errors
        sanitized = False
        try:
            app.state.model_info = _sanitize_for_json(app.state.model_info)
            sanitized = True
        except Exception:
            try:
                app.state.model_info = jsonable_encoder(app.state.model_info)
                sanitized = True
            except Exception:
                app.state.model_info = str(app.state.model_info)

        # Short startup log: which source provided MODEL_PATH and whether model_info was sanitized
        try:
            logger.info("Startup: MODEL_PATH source='%s', path='%s', model_info_sanitized=%s", MODEL_SOURCE, MODEL_PATH, sanitized)
        except Exception:
            logger.info("Startup: model loaded; model_info_sanitized=%s", sanitized)
    except Exception as e:
        # Fail fast so orchestrator notices
        raise RuntimeError(f"Failed to load model at startup: {e}")


@app.get("/health")
def health():
    pipeline = getattr(app.state, "pipeline", None)
    ok = pipeline is not None
    info = getattr(app.state, "model_info", {}) or {}
    # Ensure model_info is JSON-serializable using our sanitizer
    try:
        serializable_info = _sanitize_for_json(info)
    except Exception:
        serializable_info = info
    # Always run through jsonable_encoder on the final payload to ensure serialization
    payload = {"ok": ok, "model_info": serializable_info}
    return JSONResponse(jsonable_encoder(payload))


@app.get("/model/info")
def model_info():
    info = getattr(app.state, "model_info", None)
    if info is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        payload = _sanitize_for_json(info)
    except Exception:
        payload = info
    return JSONResponse(jsonable_encoder(payload))


@app.post("/predict")
def predict_json(payload: PredictRequest = Body(...)):
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not payload.records:
        raise HTTPException(status_code=400, detail="No records provided")

    try:
        df = pd.DataFrame.from_records(payload.records)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid record format: {e}")

    if "TARGET" in df.columns:
        df = df.drop(columns=["TARGET"])

    if df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="No feature columns provided")

    try:
        preds = pipeline.predict(df)
        response = {"predictions": preds.tolist()}
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(df)
            if proba.ndim == 2 and proba.shape[1] == 2:
                response["prob_positive"] = proba[:, 1].tolist()
            else:
                # return per-class probabilities
                classes = getattr(app.state, "model_info", {}).get("classes")
                if classes is None:
                    try:
                        classes = pipeline.named_steps["classifier"].classes_
                    except Exception:
                        classes = None

                if classes is not None:
                    response["probabilities"] = {str(c): proba[:, i].tolist() for i, c in enumerate(classes)}
                else:
                    # fallback: return raw probability columns
                    response["probabilities_raw"] = proba.tolist()
        return JSONResponse(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        suffix = (file.filename or "").lower().split(".")[-1]
        if suffix in {"csv", "txt"}:
            df = pd.read_csv(file.file)
        elif suffix in {"xls", "xlsx"}:
            df = pd.read_excel(file.file)
        else:
            # attempt csv fallback
            df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    if "TARGET" in df.columns:
        df = df.drop(columns=["TARGET"])

    if df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="No feature columns in file")

    try:
        preds = pipeline.predict(df)
        result = pd.DataFrame({"PREDICTION": preds})
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(df)
            if proba.ndim == 2 and proba.shape[1] == 2:
                result["PROB_POS"] = proba[:, 1]
            else:
                classes = getattr(app.state, "model_info", {}).get("classes")
                if classes is None:
                    try:
                        classes = pipeline.named_steps["classifier"].classes_
                    except Exception:
                        classes = None

                if classes is not None:
                    for i, c in enumerate(classes):
                        result[f"PROB_{c}"] = proba[:, i]
                else:
                    # fallback: expand raw probabilities with numeric column names
                    for i in range(proba.shape[1]):
                        result[f"PROB_COL_{i}"] = proba[:, i]
        return JSONResponse(result.to_dict(orient="records"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


@app.post("/reload")
def reload_model(key: Optional[str] = Query(None)):
    if key != RELOAD_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    try:
        app.state.pipeline = load_pipeline(MODEL_PATH)
        # refresh metadata
        app.state.model_info = _extract_model_info(app.state.pipeline, MODEL_PATH)
        return {"reloaded": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")


# Example: run with:
# uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --workers 1
# For production use, run multiple workers behind a load balancer and use a model registry. 