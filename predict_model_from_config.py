"""predict_model_from_config.py

Read MODEL_PATH, MODEL_FILENAME and ML_INPUT_DATAFILE from config.ini,
load the trained pipeline, run predictions for all records in the input
file, and write results to MODEL_FINAL_OUTPUT.JSON.

Usage:
    python predict_model_from_config.py [--config path/to/config.ini]

The script logs progress and handles common errors (missing files, load
errors, prediction failures) with clear messages and non-zero exit codes.
"""
from __future__ import annotations

import argparse
import configparser
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> configparser.SectionProxy:
    cfg = configparser.ConfigParser()
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    cfg.read(p)
    section = cfg["DEFAULT"] if "DEFAULT" in cfg else cfg.sections()[0]
    return cfg


def _sanitize_for_json(obj: Any):
    """Convert numpy/pandas scalars/arrays/frames to plain Python types."""
    try:
        import numpy as _np
        import pandas as _pd
    except Exception:
        _np = None
        _pd = None

    # primitives
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj

    # numpy/pandas
    if _np is not None:
        if isinstance(obj, _np.integer):
            return int(obj)
        if isinstance(obj, _np.floating):
            return float(obj)
        if isinstance(obj, _np.ndarray):
            return [_sanitize_for_json(x) for x in obj.tolist()]
    if _pd is not None:
        if isinstance(obj, _pd.Timestamp) or isinstance(obj, _pd.Timedelta):
            return str(obj)
        if isinstance(obj, (_pd.Series, _pd.Index)):
            return [_sanitize_for_json(x) for x in obj.tolist()]
        if isinstance(obj, _pd.DataFrame):
            return [_sanitize_for_json(r) for r in obj.to_dict(orient="records")]

    # mapping / sequence
    try:
        from collections.abc import Mapping, Sequence

        if isinstance(obj, Mapping):
            return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            return [_sanitize_for_json(x) for x in obj]
    except Exception:
        pass

    # fallback
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


def resolve_model_path(cfg: configparser.ConfigParser) -> Path:
    # Read MODEL_PATH and MODEL_FILENAME; allow MODEL_PATH to be directory or file
    model_path_raw = None
    model_filename = "random_forest_pipeline.pkl"
    try:
        model_path_raw = cfg.get("DEFAULT", "MODEL_PATH", fallback=None)
    except Exception:
        model_path_raw = None
    try:
        model_filename = cfg.get("DEFAULT", "MODEL_FILENAME", fallback=model_filename)
    except Exception:
        model_filename = model_filename

    if not model_path_raw:
        # fallback to current working directory + filename
        candidate = Path.cwd() / model_filename
        logger.info("No MODEL_PATH in config; using %s", candidate)
        return candidate

    p = Path(model_path_raw)
    if p.exists():
        if p.is_dir():
            candidate = p / model_filename
            return candidate
        else:
            return p
    else:
        # If path doesn't exist, assume it's a file path or a directory to be created later
        if p.suffix:
            return p
        else:
            return p / model_filename


def load_pipeline(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Pipeline file not found: {p}")
    try:
        pipeline = joblib.load(p)
        logger.info("Loaded pipeline from %s", p)
        return pipeline
    except Exception as e:
        logger.exception("Failed to load pipeline: %s", e)
        raise


def read_input(input_path: str | Path) -> pd.DataFrame:
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    suffix = p.suffix.lower()
    try:
        if suffix in {".csv"}:
            df = pd.read_csv(p)
        elif suffix in {".xls", ".xlsx"}:
            df = pd.read_excel(p)
        elif suffix in {".json"}:
            df = pd.read_json(p)
        else:
            # try csv first
            try:
                df = pd.read_csv(p)
            except Exception:
                df = pd.read_excel(p)
        logger.info("Loaded input data shape: %s", df.shape)
        return df
    except Exception as e:
        logger.exception("Failed to read input file %s: %s", p, e)
        raise


def predict_all(pipeline, df: pd.DataFrame) -> pd.DataFrame:
    # drop TARGET if present
    if "TARGET" in df.columns:
        logger.info("Dropping TARGET column present in input before prediction")
        df = df.drop(columns=["TARGET"])
    # ensure pipeline can accept df
    try:
        preds = pipeline.predict(df)
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise

    out = df.copy()
    out["PREDICTION"] = preds
    # add probabilities if available
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(df)
            if proba.ndim == 2 and proba.shape[1] == 2:
                out["PREDICTION_PROBA_POS"] = proba[:, 1]
            else:
                # create columns per class
                classes = None
                try:
                    classes = pipeline.named_steps["classifier"].classes_
                except Exception:
                    classes = None
                if classes is not None and proba.ndim == 2 and proba.shape[1] == len(classes):
                    for i, c in enumerate(classes):
                        out[f"PROBA_CLASS_{c}"] = proba[:, i]
        except Exception:
            logger.exception("predict_proba failed; continuing without probabilities")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Load model from config and run predictions for ML_INPUT_DATAFILE")
    parser.add_argument("--config", default="config.ini", help="Path to config.ini")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except Exception as e:
        logger.error("Failed to load config: %s", e)
        sys.exit(2)

    # Resolve paths
    try:
        ml_input = cfg.get("DEFAULT", "ML_INPUT_DATAFILE")
    except Exception:
        ml_input = None

    if not ml_input:
        logger.error("ML_INPUT_DATAFILE not set in config.ini")
        sys.exit(2)

    try:
        model_path = resolve_model_path(cfg)
    except Exception as e:
        logger.exception("Failed to resolve model path: %s", e)
        sys.exit(2)

    try:
        pipeline = load_pipeline(model_path)
    except Exception as e:
        logger.error("Failed to load pipeline: %s", e)
        sys.exit(3)

    try:
        df_in = read_input(ml_input)
    except Exception as e:
        logger.error("Failed to read ML input file: %s", e)
        sys.exit(4)

    try:
        out_df = predict_all(pipeline, df_in)
    except Exception as e:
        logger.error("Prediction step failed: %s", e)
        sys.exit(5)

    # Determine output location: same folder as ML_INPUT_DATAFILE
    out_path = Path(ml_input).resolve().parent / "MODEL_FINAL_OUTPUT.JSON"
    try:
        records = out_df.to_dict(orient="records")
        sanitized = _sanitize_for_json(records)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(sanitized, fh, indent=2, ensure_ascii=False)
        logger.info("Wrote predictions to %s", out_path)
    except Exception as e:
        logger.exception("Failed to write output JSON: %s", e)
        sys.exit(6)


if __name__ == "__main__":
    main()
