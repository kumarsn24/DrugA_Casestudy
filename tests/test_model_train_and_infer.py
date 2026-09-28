import numpy as np
import pandas as pd
import model_train_and_infer as mti


def test_read_input_file_csv(tmp_path):
    p = tmp_path / "data.csv"
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df.to_csv(p, index=False)

    loaded = mti.read_input_file(str(p))
    assert isinstance(loaded, pd.DataFrame)
    assert list(loaded.columns) == ["a", "b"]


def test_build_preprocessing_pipeline_basic():
    # numeric and categorical columns
    df = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"]})
    pre = mti.build_preprocessing_pipeline(df)
    # Should be a ColumnTransformer-like object
    assert hasattr(pre, "transform")


def test_train_and_infer_end_to_end(tmp_path, monkeypatch):
    # Create a synthetic dataset with enough samples for train_test_split
    n = 30
    df = pd.DataFrame({
        "feat_num": np.arange(n) + 0.1,
        "feat_cat": ["a", "b"] * (n // 2) + (["a"] if n % 2 else []),
        "TARGET": [0, 1] * (n // 2) + ([0] if n % 2 else []),
    })

    input_csv = tmp_path / "ml_input.csv"
    df.to_csv(input_csv, index=False)

    # Monkeypatch cross_validate to avoid heavy CV and potential small-sample errors
    def fake_cross_validate(pipeline, X, y, cv, scoring, n_jobs, return_train_score):
        # return arrays for each requested metric
        out = {}
        for s in scoring:
            out[f"test_{s}"] = np.array([1.0])
        return out

    monkeypatch.setattr(mti, "cross_validate", fake_cross_validate)

    model_path = tmp_path / "pipeline_test.pkl"

    # Run training (should save pipeline)
    mti.train_pipeline(
        str(input_csv),
        output_model_path=str(model_path),
        show_plot=False,
        n_estimators=5,
        random_state=0,
        test_size=0.2,
        n_repeats=1,
    )

    assert model_path.exists()

    # Prepare inference input (no TARGET)
    infer_df = df.drop(columns=["TARGET"]).head(5)
    infer_csv = tmp_path / "infer_input.csv"
    infer_df.to_csv(infer_csv, index=False)

    # Run inference using the saved pipeline
    res = mti.run_inference(str(model_path), str(infer_csv), output_path=str(tmp_path / "preds.csv"))
    assert "PREDICTION" in res.columns
    out_file = tmp_path / "preds.csv"
    assert out_file.exists()
