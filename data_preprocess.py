"""
data_preprocess.py

Modular data ingestion and cleansing script for DiseaseX_EMRAlerts.

Key features:
- Reads files from a DATAFILE_LOCATION_PATH configured in config.ini
- Per-file processing functions: process_patient, process_physician, process_transactions
- Robust encoding fallback when reading CSVs (utf-8 -> utf-8-sig -> latin-1)
- Basic cleansing: trim strings, uppercase column names, drop/handle missing PKs, dedupe, datatype conversions
- Writes cleaned CSV outputs to a processed/ folder under DATAFILE_LOCATION_PATH

Usage:
    python -m DiseaseX_EMRAlerts.data_preprocess

This file is intended to be beginner-friendly and well-commented.
"""

from __future__ import annotations

import configparser
import logging
import os
from typing import Optional

import pandas as pd
import json

# Configure logging for clarity during processing
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> configparser.ConfigParser:
    """Load configuration from config.ini located next to this script by default.

    Returns a ConfigParser object with at least the DEFAULT section expected by this program.
    """
    config = configparser.ConfigParser()
    if config_path is None:
        # file located in same package directory
        base_dir = os.path.dirname(__file__)
        config_path = os.path.join(base_dir, "config.ini")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config.read(config_path)
    return config


def _read_csv_with_fallback(filepath: str, **pd_read_csv_kwargs) -> pd.DataFrame:
    """Read CSV with encoding fallbacks and common error handling.

    Tries common encodings until one succeeds.
    """
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    last_exc: Optional[Exception] = None
    for enc in encodings:
        try:
            logger.debug("Trying to read %s with encoding=%s", filepath, enc)
            df = pd.read_csv(filepath, encoding=enc, **pd_read_csv_kwargs)
            logger.info("Read %s (%d rows) using encoding=%s", filepath, len(df), enc)
            return df
        except Exception as exc:  # broad because pandas can raise different errors
            last_exc = exc
            logger.debug("Failed reading %s with encoding=%s: %s", filepath, enc, exc)

    # If all encodings failed, re-raise the last exception with more context
    raise UnicodeError(f"Unable to read CSV '{filepath}' with tried encodings. Last error: {last_exc}")


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names: strip and uppercase for consistent downstream usage."""
    df = df.copy()
    df.columns = [str(col).strip().upper() for col in df.columns]
    return df


def _trim_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace from string/object dtype columns in-place (returns new df)."""
    df = df.copy()
    # Include both pandas string dtype and object-backed string columns to be
    # compatible with pandas 2.x/3.x and avoid future deprecation warnings.
    obj_cols = df.select_dtypes(include=["string", "object"]).columns.tolist()
    if obj_cols:
        # Cast to pandas string dtype then strip in a vectorized way. Use .loc to
        # assign multiple columns at once which is faster for wide dataframes.
        df.loc[:, obj_cols] = df.loc[:, obj_cols].astype("string").apply(lambda s: s.str.strip())
    return df


def _ensure_output_folder(base_path: str, output_folder: str) -> str:
    out_path = os.path.join(base_path, output_folder)
    os.makedirs(out_path, exist_ok=True)
    return out_path


def process_patient(base_path: str, filename: str, output_folder: str, primary_key: str = "PATIENT_ID") -> pd.DataFrame:
    """Ingest and clean patient dimension file.

    Steps:
    - Read CSV with encoding fallback
    - Standardize column names
    - Drop rows with missing primary key
    - Deduplicate by primary key (keep first)
    - Trim strings
    - Basic datatype conversions if possible
    - Write cleaned CSV to output_folder and return DataFrame
    """
    file_path = os.path.join(base_path, filename)
    df = _read_csv_with_fallback(file_path)
    df = _standardize_columns(df)

    # Ensure primary key exists
    pk = primary_key.upper()
    if pk not in df.columns:
        raise KeyError(f"Primary key column '{pk}' not found in patient file")

    initial_rows = len(df)
    df = df.dropna(subset=[pk])
    dropped = initial_rows - len(df)
    if dropped:
        logger.warning("Dropped %d patient rows with missing %s", dropped, pk)

    # Trim strings and normalize
    df = _trim_string_columns(df)

    # Deduplicate
    dup_count = df.duplicated(subset=[pk]).sum()
    if dup_count:
        logger.warning("Found %d duplicate patient rows by %s; keeping first occurrence", dup_count, pk)
        df = df.drop_duplicates(subset=[pk], keep="first")

    # Example: try to parse common date columns if present
    for col in ["DOB", "DATE_OF_BIRTH", "BIRTHDATE"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Fill obvious categorical missing values
    for col in df.select_dtypes(include=["string", "object"]).columns:
        # do not overwrite ids
        if col == pk:
            continue
        df[col] = df[col].fillna("Unknown")

    # Persist
    out_dir = _ensure_output_folder(base_path, output_folder)
    out_path = os.path.join(out_dir, "dim_patient_clean.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("Wrote cleaned patient data to %s", out_path)
    return df


def process_physician(base_path: str, filename: str, output_folder: str, primary_key: str = "PHYSICIAN_ID") -> pd.DataFrame:
    """Ingest and clean physician dimension file.

    Similar steps to process_patient with domain-appropriate defaults.
    """
    file_path = os.path.join(base_path, filename)
    df = _read_csv_with_fallback(file_path)
    df = _standardize_columns(df)

    pk = primary_key.upper()
    if pk not in df.columns:
        raise KeyError(f"Primary key column '{pk}' not found in physician file")

    initial_rows = len(df)
    df = df.dropna(subset=[pk])
    dropped = initial_rows - len(df)
    if dropped:
        logger.warning("Dropped %d physician rows with missing %s", dropped, pk)

    df = _trim_string_columns(df)

    # Sub-function: handle missing values except physician id
    def _handle_missing_values_physician(df_local: pd.DataFrame, physician_col: str = "PHYSICIAN_ID") -> pd.DataFrame:
        """Fill missing values across the dataframe except for the physician id column.

        - Numeric columns: fill missing values with 0
        - Categorical/string columns: fill missing values with 'Unknown'
        """
        df_local = df_local.copy()
        cols = df_local.columns.tolist()
        if physician_col in cols:
            cols = [c for c in cols if c != physician_col]

        numeric_cols = df_local[cols].select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = [c for c in cols if c not in numeric_cols]

        if numeric_cols:
            df_local[numeric_cols] = df_local[numeric_cols].fillna(0)

        if categorical_cols:
            df_local[categorical_cols] = df_local[categorical_cols].fillna("Unknown")

        return df_local

    # Apply missing value handling for physician data
    df = _handle_missing_values_physician(df, physician_col=pk)

    dup_count = df.duplicated(subset=[pk]).sum()
    if dup_count:
        logger.warning("Found %d duplicate physician rows by %s; keeping first occurrence", dup_count, pk)
        df = df.drop_duplicates(subset=[pk], keep="first")

    # Normalize specialty column if present
    for col in ["SPECIALTY", "DEPARTMENT"]:
        if col in df.columns:
            # Ensure string dtype before string operations to avoid dtype warnings
            df[col] = df[col].astype("string").str.title().fillna("Unknown")

    out_dir = _ensure_output_folder(base_path, output_folder)
    out_path = os.path.join(out_dir, "dim_physician_clean.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("Wrote cleaned physician data to %s", out_path)
    return df


def process_transactions(base_path: str, filename: str, output_folder: str, primary_key: str = "PATIENT_ID") -> pd.DataFrame:
    """Ingest and clean transactions (fact) file.

    Steps:
    - Read CSV with encoding fallback
    - Ensure PATIENT_ID exists (as per user requirement PATIENT_ID is primary key for transactions)
    - Convert common numeric/date columns
    - Remove rows with missing patient id
    - Deduplicate if exact duplicates exist
    - Write cleaned CSV to output and return DataFrame
    """
    file_path = os.path.join(base_path, filename)
    df = _read_csv_with_fallback(file_path)
    df = _standardize_columns(df)

    # Normalize strings and parse dates early
    df = _trim_string_columns(df)

    # Identify and coalesce transaction date columns into a single TXN_DT column
    date_candidates = ["TRANSACTION_DATE", "TXN_DATE", "DATE", "TXN_DT"]
    present_dates = [c for c in date_candidates if c in df.columns]
    if present_dates:
        # Convert present date columns to datetime
        for col in present_dates:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        # Create TXN_DT by taking the first non-null among the ordered candidates
        df["TXN_DT"] = df[present_dates].bfill(axis=1).iloc[:, 0]
    else:
        # If no date available, create empty TXN_DT
        df["TXN_DT"] = pd.NaT

    # Convert numeric amount columns if present
    for col in ["AMOUNT", "TOTAL", "CHARGE", "COST"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    patient_col = "PATIENT_ID"
    if patient_col in df.columns:
        # Drop rows missing patient id as they cannot be grouped
        initial_rows = len(df)
        df = df.dropna(subset=[patient_col])
        dropped = initial_rows - len(df)
        if dropped:
            logger.warning("Dropped %d transaction rows with missing %s", dropped, patient_col)

        # Keep a copy of the original transactions (post-basic-cleaning) so we can
        # compute aggregated values (like NO_OF_CONDN) from the full set before
        # any per-patient deduplication is applied.
        df_original = df.copy()

        # Vectorized selection: keep for each PATIENT_ID a single row.
        # Preference: row(s) with the maximum TXN_DT. If a patient has no TXN_DT values,
        # keep the first row for that patient.
        # 1) compute group-wise max TXN_DT
        group_max = df.groupby(patient_col)["TXN_DT"].transform("max")

        # 2) boolean mask where row TXN_DT equals group max (and group max is not null)
        has_group_date = group_max.notna()
        mask_max_date = has_group_date & (df["TXN_DT"] == group_max)

        # 3) boolean mask for groups with no dates: keep first row per group
        first_in_group = df.groupby(patient_col).cumcount() == 0
        mask_no_date = ~has_group_date & first_in_group

        # 4) combined mask
        mask_keep = mask_max_date | mask_no_date

        # 5) select rows and ensure exactly one row per patient by dropping duplicate patient ids (keep first)
        df = df[mask_keep].drop_duplicates(subset=[patient_col], keep="first").reset_index(drop=True)
    else:
        # If PATIENT_ID not present, fall back to dropping exact duplicates
        dup_count = df.duplicated().sum()
        if dup_count:
            logger.info("PATIENT_ID missing: dropping %d exact duplicate transaction rows", dup_count)
            df = df.drop_duplicates(keep="first")

    # Sub-function: handle missing values except patient id
    def _handle_missing_values(df_local: pd.DataFrame, patient_col: str = "PATIENT_ID") -> pd.DataFrame:
        """Fill missing values across the dataframe except for the patient id column.

        - Numeric columns: fill missing values with 0
        - Categorical/string columns: fill missing values with 'Unknown'
        """
        df_local = df_local.copy()
        cols = df_local.columns.tolist()
        if patient_col in cols:
            cols = [c for c in cols if c != patient_col]

        # Determine numeric columns using pandas dtypes
        numeric_cols = df_local[cols].select_dtypes(include=["number"]).columns.tolist()

        # Remaining columns are treated as categorical/string
        categorical_cols = [c for c in cols if c not in numeric_cols]

        if numeric_cols:
            df_local[numeric_cols] = df_local[numeric_cols].fillna(0)

        if categorical_cols:
            df_local[categorical_cols] = df_local[categorical_cols].fillna("Unknown")

        return df_local

    # Sub-function: add Target column based on TXN_DESC
    def _add_target_column(df_local: pd.DataFrame) -> pd.DataFrame:
        """Add 'Target' column: 1 when TXN_DESC equals 'drug a' (case-insensitive), else 0.

        The column will be integer dtype (0/1).
        """
        desc_col = "TXN_DESC"
        if desc_col in df_local.columns:
            matches = (
                df_local[desc_col]
                .astype("string")
                .str.strip()
                .str.lower()
                .eq("drug a")
            )
            df_local["Target"] = matches.astype("int").fillna(0).astype(int)
        else:
            df_local["Target"] = 0
        return df_local

    # Handle missing values before adding the Target column
    df = _handle_missing_values(df, patient_col=patient_col)
    df = _add_target_column(df)

    # Generic sub-function: aggregate counts of a given TXN_TYPE value per PATIENT_ID
    def _aggregate_txn_type_count(df_local: pd.DataFrame, txn_type_value: str, patient_col: str = "PATIENT_ID", txn_type_col: str = "TXN_TYPE") -> pd.Series:
        """Compute counts of rows where txn_type_col equals txn_type_value (case-insensitive)
        grouped by patient_col. Returns a Series indexed by PATIENT_ID with integer counts.
        If required columns are missing, returns an empty Series.
        """
        if patient_col not in df_local.columns or txn_type_col not in df_local.columns:
            return pd.Series(dtype="int64")

        mask = (
            df_local[txn_type_col]
            .astype("string")
            .str.strip()
            .str.upper()
            .eq(str(txn_type_value).strip().upper())
        )

        counts = mask.groupby(df_local[patient_col]).sum().astype(int)
        return counts

    # Compute aggregated counts for each required TXN_TYPE from the in-memory original transactions
    if patient_col in df.columns:
        cond_counts = _aggregate_txn_type_count(df_original, "CONDITIONS", patient_col=patient_col)
        df["NO_OF_CONDN"] = df[patient_col].map(cond_counts).fillna(0).astype(int)

        sympt_counts = _aggregate_txn_type_count(df_original, "SYMPTOMS", patient_col=patient_col)
        df["NO_OF_SYMPT"] = df[patient_col].map(sympt_counts).fillna(0).astype(int)

        contra_counts = _aggregate_txn_type_count(df_original, "CONTRAINDICATIONS", patient_col=patient_col)
        df["NO_OF_CONTRD"] = df[patient_col].map(contra_counts).fillna(0).astype(int)
    else:
        df["NO_OF_CONDN"] = 0
        df["NO_OF_SYMPT"] = 0
        df["NO_OF_CONTRD"] = 0

    out_dir = _ensure_output_folder(base_path, output_folder)
    out_path = os.path.join(out_dir, "fact_transactions_clean.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("Wrote cleaned transactions to %s", out_path)
    return df


def generate_cleansed_output(base_path: str, output_folder: str, config_path: str = None) -> pd.DataFrame:
    """Read cleaned patient, physician, and transaction files and produce a consolidated Cleansed_Output.csv.

    Merging strategy:
    - Transactions is the left table: all transaction rows are preserved.
    - LEFT JOIN transactions -> patient on PATIENT_ID
    - LEFT JOIN result -> physician on PHYSICIAN_ID (if physician id present)

    Writes Cleansed_Output.csv into the processed output folder and returns the DataFrame.
    """
    out_dir = _ensure_output_folder(base_path, output_folder)

    patient_path = os.path.join(out_dir, "dim_patient_clean.csv")
    physician_path = os.path.join(out_dir, "dim_physician_clean.csv")
    transactions_path = os.path.join(out_dir, "fact_transactions_clean.csv")

    # Read files using the robust reader
    if not os.path.exists(transactions_path):
        raise FileNotFoundError(f"Cleaned transactions file not found: {transactions_path}")

    tx = _read_csv_with_fallback(transactions_path)
    tx = _standardize_columns(tx)
    tx = _trim_string_columns(tx)

    # Attach patient if available
    if os.path.exists(patient_path):
        pat = _read_csv_with_fallback(patient_path)
        pat = _standardize_columns(pat)
        pat = _trim_string_columns(pat)
        if "PATIENT_ID" in tx.columns and "PATIENT_ID" in pat.columns:
            tx = tx.merge(pat.add_prefix("PAT_"), how="left", left_on="PATIENT_ID", right_on="PAT_PATIENT_ID")
            # Optionally drop the duplicated right-side key column
            tx = tx.drop(columns=[c for c in tx.columns if c == "PAT_PATIENT_ID"], errors="ignore")
        else:
            logger.warning("PATIENT_ID missing in either transactions or patient cleansed file; skipping patient join")
    else:
        logger.warning("Patient cleansed file not found at %s; skipping patient join", patient_path)

    # Attach physician if available and if transactions contains PHYSICIAN_ID
    if os.path.exists(physician_path):
        phy = _read_csv_with_fallback(physician_path)
        phy = _standardize_columns(phy)
        phy = _trim_string_columns(phy)
        if "PHYSICIAN_ID" in tx.columns and "PHYSICIAN_ID" in phy.columns:
            tx = tx.merge(phy.add_prefix("PHY_"), how="left", left_on="PHYSICIAN_ID", right_on="PHY_PHYSICIAN_ID")
            tx = tx.drop(columns=[c for c in tx.columns if c == "PHY_PHYSICIAN_ID"], errors="ignore")
        else:
            logger.info("PHYSICIAN_ID not present in transactions or physician file; skipping physician join")
    else:
        logger.info("Physician cleansed file not found at %s; skipping physician join", physician_path)

    # Sub-function: derive patient age from PAT_BIRTH_YEAR column
    def _derive_patient_age(merged_tx: pd.DataFrame, birth_col: str = "PAT_BIRTH_YEAR") -> pd.DataFrame:
        """Derive PAT_AGE by subtracting birth year from the current year.

        merged_tx: the merged transactions dataframe (transactions joined with patient data).
        If birth_col is missing or non-numeric, PAT_AGE will be set to 0.
        Returns a new DataFrame with PAT_AGE as an integer column.
        """
        df_out = merged_tx.copy()
        if birth_col in df_out.columns:
            # Coerce to numeric (year); if a full date is present, attempt to extract year
            vals = df_out[birth_col]
            # Robust handling: prefer numeric year values when they look like a year
            current_year = pd.Timestamp.now().year

            # Try to coerce to numeric first (handles plain year values like 1987)
            numeric_years = pd.to_numeric(vals, errors="coerce")
            valid_numeric_mask = (numeric_years >= 1900) & (numeric_years <= current_year)
            years = numeric_years.where(valid_numeric_mask)

            # For entries not covered by valid numeric years, try parsing as dates
            if years.isna().any():
                parsed = pd.to_datetime(vals, errors="coerce", infer_datetime_format=True)
                parsed_years = parsed.dt.year
                years = years.fillna(parsed_years)

            # Cast to pandas nullable integer dtype for consistency
            years = years.astype("Int64")

            age = current_year - years
            df_out["PAT_AGE"] = age.fillna(0).replace([pd.NA], 0).astype(int)
        else:
            df_out["PAT_AGE"] = 0
        return df_out

    # Compute PAT_AGE and append to the consolidated transactions
    tx = _derive_patient_age(tx, birth_col="PAT_BIRTH_YEAR")

    # Impute missing values across the transactions dataframe
    # Behavior is configurable via config.ini keys in DEFAULT section:
    #   IMPUTE_NUMERIC_VALUE (default 0)
    #   IMPUTE_CATEGORICAL_VALUE (default UNKNOWN)
    #   IMPUTE_EXCLUDE_COLS (comma-separated list of columns to exclude; default PATIENT_ID)
    #   IMPUTE_BLANK_AS_NA (boolean, default True)
    try:
        cfg_section = None
        try:
            cfg_section = load_config(config_path)["DEFAULT"]
        except Exception:
            cfg_section = None

        # Read config-driven values with sensible defaults
        impute_numeric_raw = None
        impute_categorical_value = "UNKNOWN"
        impute_exclude = ["PATIENT_ID"]
        impute_blank_as_na = True

        if cfg_section is not None:
            try:
                impute_numeric_raw = cfg_section.get("IMPUTE_NUMERIC_VALUE", None)
            except Exception:
                impute_numeric_raw = None
            try:
                impute_categorical_value = cfg_section.get("IMPUTE_CATEGORICAL_VALUE", impute_categorical_value)
            except Exception:
                impute_categorical_value = impute_categorical_value
            try:
                excl = cfg_section.get("IMPUTE_EXCLUDE_COLS", None)
                if excl:
                    impute_exclude = [c.strip().upper() for c in (excl.split(",") if isinstance(excl, str) else excl) if str(c).strip()]
            except Exception:
                impute_exclude = impute_exclude
            try:
                impute_blank_as_na = cfg_section.getboolean("IMPUTE_BLANK_AS_NA", fallback=True)
            except Exception:
                impute_blank_as_na = True

        # Parse numeric impute value, default to 0 when not parseable
        impute_numeric_value = 0
        if impute_numeric_raw is not None:
            try:
                if isinstance(impute_numeric_raw, str) and "." in impute_numeric_raw:
                    impute_numeric_value = float(impute_numeric_raw)
                else:
                    impute_numeric_value = int(impute_numeric_raw)
            except Exception:
                try:
                    impute_numeric_value = float(impute_numeric_raw)
                except Exception:
                    impute_numeric_value = 0

        # Identify numeric columns
        numeric_cols = tx.select_dtypes(include=["number"]).columns.tolist()
        # Exclude configured columns (uppercase normalized)
        exclude_upper = [c.upper() for c in impute_exclude]
        numeric_cols = [c for c in numeric_cols if c.upper() not in exclude_upper]

        # Fill numeric missing values with configured numeric value
        if numeric_cols:
            tx[numeric_cols] = tx[numeric_cols].fillna(impute_numeric_value)

        # Categorical columns: everything not numeric and not datetime and not excluded
        cat_cols = [
            c
            for c in tx.columns
            if c not in numeric_cols and c.upper() not in exclude_upper and not pd.api.types.is_datetime64_any_dtype(tx[c])
        ]

        if cat_cols:
            if impute_blank_as_na:
                try:
                    tx.loc[:, cat_cols] = tx.loc[:, cat_cols].replace(r"^\s*$", pd.NA, regex=True)
                except Exception:
                    pass

            # Cast to string dtype for consistent fill where possible
            for c in cat_cols:
                try:
                    tx[c] = tx[c].astype("string")
                except Exception:
                    pass

            tx.loc[:, cat_cols] = tx.loc[:, cat_cols].fillna(impute_categorical_value)
    except Exception:
        logger.exception("Failed to impute missing values in transactions; continuing")

    # Sub-function: write a filtered output based on INCLUSION_COLS config
    def _write_inclusion_output(tx_local: pd.DataFrame, cfg_section: configparser.SectionProxy, out_dir: str) -> None:
        """Write a subset of tx_local containing only columns listed in INCLUSION_COLS.

        cfg_section: the DEFAULT section from config.ini
        out_dir: directory where the output file should be written
        The output filename is taken from FINAL_DATAPROCESS_OUTPUT in the config.
        """
        # Be defensive: cfg_section may be a SectionProxy, dict, or other mapping-like object.
        cols_raw = ""
        try:
            get_fn = getattr(cfg_section, "get", None)
            if callable(get_fn):
                cols_raw = get_fn("INCLUSION_COLS", "")
            elif isinstance(cfg_section, dict):
                cols_raw = cfg_section.get("INCLUSION_COLS", "")
            else:
                # Fallback: attempt dictionary-like access
                try:
                    cols_raw = cfg_section["INCLUSION_COLS"]
                except Exception:
                    cols_raw = ""
        except Exception as exc:
            logger.exception("Error fetching INCLUSION_COLS from config: %s", exc)
            cols_raw = ""

        if not cols_raw:
            logger.debug("No INCLUSION_COLS configured; skipping inclusion output")
            return

        # Normalize configured column names to uppercase and trim.
        # Support formats like: comma-separated string or JSON-like list (e.g. ["A","B"]).
        inclusion = []
        try:
            s = cols_raw.strip()
            if s.startswith("[") and s.endswith("]"):
                # Try to parse JSON list
                parsed = json.loads(s)
                if isinstance(parsed, (list, tuple)):
                    inclusion = [str(x).strip().upper() for x in parsed if str(x).strip()]
                else:
                    # fallback to simple split
                    inclusion = [c.strip().upper() for c in s.strip('[]').split(',') if c.strip()]
            else:
                inclusion = [c.strip().upper() for c in s.split(',') if c.strip()]
        except Exception as exc:
            logger.debug("Failed to parse INCLUSION_COLS as JSON: %s; falling back to comma-split. Error: %s", cols_raw, exc)
            inclusion = [c.strip().upper() for c in cols_raw.split(',') if c.strip()]
        if not inclusion:
            logger.debug("INCLUSION_COLS empty after parsing; skipping inclusion output")
            return

        # Select only the columns that actually exist in the dataframe
        present = [c for c in inclusion if c in tx_local.columns]
        if not present:
            logger.warning("None of the configured INCLUSION_COLS found in dataframe: %s", inclusion)
            return

        # Prefer explicit final output name for inclusion output. Support legacy keys:
        # 1) ML_INPUT_DATAFILE (used by some tests/projects to name the ML input file)
        # 2) FINAL_DATAPROCESS_OUTPUT (explicit)
        # 3) OUTPUT_FILENAME (legacy)
        out_fname = (
            cfg_section.get("ML_INPUT_DATAFILE")
            or cfg_section.get("FINAL_DATAPROCESS_OUTPUT")
            or cfg_section.get("OUTPUT_FILENAME", "Final_DataProcess_Output.csv")
        )
        out_path2 = os.path.join(out_dir, out_fname)
        try:
            tx_local[present].to_csv(out_path2, index=False, encoding="utf-8")
            logger.info("Wrote inclusion-based output to %s (columns: %s)", out_path2, present)
        except Exception as exc:
            logger.exception("Failed writing inclusion-based output to %s: %s", out_path2, exc)

    # Determine output filename from config if available
    try:
        cfg = load_config(config_path)["DEFAULT"]
        output_filename = cfg.get("OUTPUT_FILENAME", "Cleansed_Output.csv")
        # If configured, write an inclusion-based final data process output
        try:
            _write_inclusion_output(tx, cfg, out_dir)
        except Exception:
            logger.exception("Failed to write inclusion-based final data process output")
    except Exception:
        output_filename = "Cleansed_Output.csv"

    # Persist the consolidated output
    out_path = os.path.join(out_dir, output_filename)
    tx.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("Wrote consolidated cleansed output to %s", out_path)
    return tx


def run_all(config_path: str = None) -> None:
    """Orchestrate the processing of all configured files."""
    config = load_config(config_path)
    cfg = config["DEFAULT"]
    base_path = cfg.get("DATAFILE_LOCATION_PATH")
    if not base_path:
        raise ValueError("DATAFILE_LOCATION_PATH must be set in config.ini")

    patient_file = cfg.get("PATIENT_FILE", "dim_patient.csv")
    physician_file = cfg.get("PHYSICIAN_FILE", "dim_physician.csv")
    transactions_file = cfg.get("TRANSACTIONS_FILE", "Fact_Transactions.csv")
    output_folder = cfg.get("OUTPUT_FOLDER", "CleansedData")

    logger.info("Starting processing with base path: %s", base_path)

    # Process each file with error handling so a failure in one doesn't prevent others
    try:
        process_patient(base_path, patient_file, output_folder, primary_key="PATIENT_ID")
    except Exception as exc:
        logger.exception("Failed processing patient file: %s", exc)

    try:
        process_physician(base_path, physician_file, output_folder, primary_key="PHYSICIAN_ID")
    except Exception as exc:
        logger.exception("Failed processing physician file: %s", exc)

    try:
        process_transactions(base_path, transactions_file, output_folder, primary_key="PATIENT_ID")
    except Exception as exc:
        logger.exception("Failed processing transactions file: %s", exc)

    # After individual files processed, generate a consolidated cleansed output
    try:
        generate_cleansed_output(base_path, output_folder)
    except Exception as exc:
        logger.exception("Failed generating consolidated cleansed output: %s", exc)

    logger.info("Data preprocessing run complete")


if __name__ == "__main__":
    # When executed directly, run with the local config.ini
    run_all()

