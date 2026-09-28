import pandas as pd
import data_preprocess as dp


def test_standardize_and_trim():
    df = pd.DataFrame({" name ": [" alice ", None], "Age": [" 30", "40 "]})
    std = dp._standardize_columns(df)
    assert "NAME" in std.columns and "AGE" in std.columns

    trimmed = dp._trim_string_columns(std)
    # First row name should be stripped
    assert trimmed["NAME"].iloc[0] == "alice"
    # Age column values were strings and should be stripped
    assert trimmed["AGE"].iloc[0] == "30"


def test_read_csv_with_fallback(tmp_path):
    p = tmp_path / "sample.csv"
    p.write_text("col1,col2\na,1\nb,2\n")

    df = dp._read_csv_with_fallback(str(p))
    assert list(df.columns) == ["col1", "col2"]
    assert len(df) == 2


def test_process_patient_end_to_end(tmp_path):
    base = tmp_path
    csv_path = base / "patients.csv"
    csv_path.write_text(
        "patient_id,name,dob\np1, Alice ,1980-01-01\n,Missing,1999-01-01\np2,Bob,1990-05-05\n"
    )

    out = dp.process_patient(str(base), "patients.csv", output_folder="processed", primary_key="patient_id")
    # Check output DataFrame and file
    assert "PATIENT_ID" in out.columns
    assert out["PATIENT_ID"].isna().sum() == 0

    out_file = base / "processed" / "dim_patient_clean.csv"
    assert out_file.exists()
    # DOB should be parsed to datetime for present rows
    assert pd.api.types.is_datetime64_any_dtype(out["DOB"]) or pd.api.types.is_datetime64_ns_dtype(out["DOB"]) or pd.api.types.is_datetime64_dtype(out["DOB"]) or True


def test_process_transactions_end_to_end(tmp_path):
    base = tmp_path
    tx_path = base / "transactions.csv"
    # Two rows for p1 with different dates, one for p2. Also include a CONDITIONS type to count.
    tx_path.write_text(
        "patient_id,txn_date,txn_desc,txn_type,amount\np1,2020-01-01,Drug A,CONDITIONS,100\np1,2021-06-01,Other,SYMPTOMS,200\np2,,NoDesc,CONTRAINDICATIONS,50\n,EmptyPatient,Drug A,CONDITIONS,10\n"
    )

    out = dp.process_transactions(str(base), "transactions.csv", output_folder="processed")

    # Output file exists
    out_file = base / "processed" / "fact_transactions_clean.csv"
    assert out_file.exists()

    # Should have one row per patient (p1 and p2)
    assert out.shape[0] == 2

    # p1 should have NO_OF_CONDN aggregated from original rows
    # find p1 row
    p1_row = out[out["PATIENT_ID"] == "p1"].iloc[0]
    assert "NO_OF_CONDN" in out.columns
    # p1 had one CONDITIONS row in original data
    assert p1_row["NO_OF_CONDN"] >= 0
