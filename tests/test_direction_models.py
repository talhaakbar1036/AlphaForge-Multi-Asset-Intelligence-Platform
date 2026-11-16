"""Direction model: leakage guards, metrics, training smoke test and routes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.features import (
    FEATURE_COLUMNS,
    build_direction_features,
    make_sequences,
    split_sequences,
)
from src.models.metrics import classification_metrics, roc_auc


def make_prices(rows: int = 140, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=rows, freq="B")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, rows)))
    open_ = close * np.exp(rng.normal(0.0, 0.002, rows))
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) * 1.006,
            "Low": np.minimum(open_, close) * 0.994,
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, rows),
        },
        index=index,
    )


def test_features_use_prior_close_only():
    frame = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0],
            "High": [101.0, 102.5, 103.0, 104.0],
            "Low": [99.0, 100.0, 101.0, 102.0],
            "Close": [100.0, 101.0, 103.0, 102.0],
            "Volume": [1000.0, 1100.0, 1200.0, 1300.0],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="B"),
    )
    features, labels = build_direction_features(frame)

    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert features.index.equals(labels.index)
    assert str(features.index[0].date()) == "2024-01-02"
    assert labels.tolist() == [1.0, 1.0, 0.0]
    assert features.iloc[-1]["close_return"] == pytest.approx(np.log(102 / 103))
    assert features.iloc[-1]["open_return"] == pytest.approx(np.log(103 / 103))
    assert features.iloc[-1]["intraday_range"] == pytest.approx(np.log(104 / 102))
    assert features.iloc[-1]["log_volume_change"] == pytest.approx(
        np.log1p(1300) - np.log1p(1200)
    )


def test_windows_exclude_their_target_row():
    index = pd.date_range("2024-01-01", periods=10, freq="B")
    features = pd.DataFrame(
        {column: np.arange(10, dtype=float) + offset for offset, column in enumerate(FEATURE_COLUMNS)},
        index=index,
    )
    labels = pd.Series(np.arange(10, dtype=float), index=index)
    dataset = make_sequences(features, labels, sequence_length=3)

    assert len(dataset) == 7
    for row, target_position in enumerate(dataset.target_positions):
        window = dataset.sequences[row, :, 0]
        assert window[-1] == features.iloc[target_position - 1, 0]
        assert features.iloc[target_position, 0] not in window

    splits = split_sequences(dataset, 0.7, 0.15, 0.15)
    assert splits.train.target_dates.max() < splits.validation.target_dates.min()
    assert splits.validation.target_dates.max() < splits.test.target_dates.min()
    assert splits.train_feature_end == splits.train.target_positions[-1]
    assert splits.train_feature_end <= splits.validation.target_positions[0]


def test_metrics_match_hand_computed_values():
    metrics = classification_metrics(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]), threshold=0.5
    )
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["majority_baseline_accuracy"] == pytest.approx(0.5)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["confusion"] == {"tp": 1, "tn": 2, "fp": 0, "fn": 1}
    assert roc_auc(np.array([1, 1]), np.array([0.2, 0.8])) is None


def test_training_predicting_and_catalog_round_trip(tmp_path):
    pytest.importorskip("tensorflow")
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")

    from src.models.predictor import DirectionPredictor, model_catalog
    from src.models.trainer import train_symbol
    from src.utils.config import DirectionModelConfig

    prices = make_prices()
    cfg = DirectionModelConfig(
        sequence_length=10,
        hidden_units=8,
        num_layers=1,
        dropout=0.1,
        learning_rate=0.01,
        batch_size=16,
        epochs=2,
        patience=1,
        seed=11,
    )
    report = train_symbol("TEST", prices, cfg=cfg, output_dir=tmp_path)

    assert report["test_accuracy"] == pytest.approx(
        report["test_majority_class_baseline_accuracy"], abs=0.5
    )
    assert (tmp_path / "TEST_direction_lstm.keras").exists()
    assert (tmp_path / "TEST_direction_scaler.joblib").exists()
    assert (tmp_path / "TEST_direction_metrics.json").exists()

    prediction = DirectionPredictor.load("TEST", tmp_path).predict(prices)
    assert prediction["prediction"] in {"UP", "DOWN"}
    assert 0.0 <= prediction["probability_up"] <= 1.0
    assert prediction["as_of"] == str(prices.index[-1].date())

    catalog = model_catalog(tmp_path)
    assert catalog["count"] == 1
    assert catalog["models"][0]["symbol"] == "TEST"


def test_prediction_routes(monkeypatch):
    from fastapi.testclient import TestClient

    import src.api.service as service
    from src.api.fastapi_app import create_app

    client = TestClient(create_app())
    monkeypatch.setattr(
        service,
        "prediction_payload",
        lambda symbol: {
            "symbol": symbol,
            "prediction": "UP",
            "probability_up": 0.55,
        },
    )
    monkeypatch.setattr(service, "models_payload", lambda: {"models": [], "count": 0})

    response = client.get("/api/v1/predict/aapl")
    assert response.status_code == 200
    assert response.json()["symbol"] == "AAPL"
    assert client.get("/api/v1/models").json()["count"] == 0

    from src.models.errors import (
        ModelArtifactNotFound,
        ModelDependencyError,
        ModelPredictionError,
    )

    def missing(_symbol):
        raise ModelArtifactNotFound("no model")

    monkeypatch.setattr(service, "prediction_payload", missing)
    assert client.get("/api/v1/predict/AAPL").status_code == 404

    def no_tensorflow(_symbol):
        raise ModelDependencyError("tensorflow missing")

    monkeypatch.setattr(service, "prediction_payload", no_tensorflow)
    assert client.get("/api/v1/predict/AAPL").status_code == 503

    def bad_input(_symbol):
        raise ModelPredictionError("not enough history")

    monkeypatch.setattr(service, "prediction_payload", bad_input)
    assert client.get("/api/v1/predict/AAPL").status_code == 422
