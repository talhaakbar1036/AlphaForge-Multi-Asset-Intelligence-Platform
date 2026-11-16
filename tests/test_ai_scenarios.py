"""Scenario simulator: deterministic math, validation, honest labelling."""
from __future__ import annotations

import pytest

from src.ai.errors import ScenarioError
from src.ai.scenarios import SCENARIOS, run_scenario

# Real local data exists for these symbols (data/processed/*_features.parquet).
SYMBOL = "NVDA"
PEER = "AAPL"


class TestCatalogue:
    def test_supported_scenarios_listed(self):
        assert set(SCENARIOS) == {
            "price_shock", "volatility_shock", "weight_change", "rate_shock",
        }

    def test_unknown_scenario_rejected(self):
        with pytest.raises(ScenarioError):
            run_scenario("moon_shot", {})


class TestPriceShock:
    def test_single_asset_ten_percent_drop(self):
        out = run_scenario("price_shock", {"symbol": SYMBOL, "shock_pct": -10})
        assert out["kind"] == "hypothetical"
        assert "disclaimer" in out and "not a forecast" in out["disclaimer"]
        assert out["results"]["portfolio_value_change_pct"] == pytest.approx(-10.0, abs=0.01)
        assert out["results"]["weights_after_pct"][SYMBOL] == pytest.approx(100.0)

    def test_two_asset_allocation_reweights(self):
        out = run_scenario(
            "price_shock",
            {
                "symbol": SYMBOL,
                "shock_pct": -10,
                "allocation": {SYMBOL: 50.0, PEER: 50.0},
            },
        )
        # portfolio loses 5% and NVDA's weight falls below 50
        assert out["results"]["portfolio_value_change_pct"] == pytest.approx(-5.0, abs=0.01)
        assert out["results"]["weights_after_pct"][SYMBOL] < 50.0
        assert out["results"]["weights_after_pct"][PEER] > 50.0
        # the gaining asset now exceeds 50% -> concentration increases
        assert out["results"]["concentration_hhi_after"] > out["results"]["concentration_hhi_before"]

    def test_validation_errors(self):
        with pytest.raises(ScenarioError):
            run_scenario("price_shock", {"symbol": "NOTREAL", "shock_pct": -5})
        with pytest.raises(ScenarioError):
            run_scenario("price_shock", {"symbol": SYMBOL, "shock_pct": -80})
        with pytest.raises(ScenarioError):
            run_scenario("price_shock", {"symbol": SYMBOL, "shock_pct": -5,
                                         "allocation": {SYMBOL: -10}})
        with pytest.raises(ScenarioError):
            run_scenario("price_shock", {"symbol": SYMBOL, "shock_pct": "ten"})
        with pytest.raises(ScenarioError):
            run_scenario("price_shock", {"symbol": ""})


class TestVolatilityShock:
    def test_scale_multiplies_volatility(self):
        out = run_scenario("volatility_shock", {"symbol": SYMBOL, "scale": 1.5})
        before = out["results"]["annualized_volatility_before"]
        after = out["results"]["annualized_volatility_after"]
        assert after == pytest.approx(before * 1.5, rel=1e-6)
        # drawdown on a scaled path is deeper
        assert out["results"]["max_drawdown_after"] <= out["results"]["max_drawdown_before"]
        assert out["results"]["observations"] > 50

    def test_scale_bounds(self):
        with pytest.raises(ScenarioError):
            run_scenario("volatility_shock", {"symbol": SYMBOL, "scale": 0.0})
        with pytest.raises(ScenarioError):
            run_scenario("volatility_shock", {"symbol": SYMBOL, "scale": 99})


class TestWeightChange:
    def test_move_weight_rebalances_stats(self):
        out = run_scenario(
            "weight_change",
            {
                "symbol": SYMBOL,
                "to_symbol": PEER,
                "delta_pct": 10.0,
                "weights": {SYMBOL: 60.0, PEER: 40.0},
            },
        )
        assert out["results"]["weights_after_pct"][SYMBOL] == pytest.approx(50.0)
        assert out["results"]["weights_after_pct"][PEER] == pytest.approx(50.0)
        assert out["results"]["before"]["concentration_hhi"] > out["results"]["after"]["concentration_hhi"]
        assert out["results"]["history_observations"] > 50
        assert any("not forecasts" in lim for lim in out["limitations"])

    def test_validation_errors(self):
        weights = {SYMBOL: 60.0, PEER: 40.0}
        with pytest.raises(ScenarioError):
            run_scenario("weight_change", {
                "symbol": SYMBOL, "to_symbol": SYMBOL, "delta_pct": 5, "weights": weights,
            })
        with pytest.raises(ScenarioError):
            run_scenario("weight_change", {
                "symbol": SYMBOL, "to_symbol": PEER, "delta_pct": 90, "weights": weights,
            })
        with pytest.raises(ScenarioError):
            run_scenario("weight_change", {
                "symbol": "TSLA", "to_symbol": PEER, "delta_pct": 5, "weights": weights,
            })
        with pytest.raises(ScenarioError):
            run_scenario("weight_change", {
                "symbol": SYMBOL, "to_symbol": PEER, "delta_pct": 0, "weights": weights,
            })


class TestRateShock:
    def test_estimates_from_local_macro_data(self):
        out = run_scenario("rate_shock", {"symbol": SYMBOL, "shock_bps": 100})
        results = out["results"]
        assert results["sample_months"] >= 24
        assert results["estimated_beta_return_per_bp"] != 0.0
        assert results["implied_return_estimate"] == pytest.approx(
            results["estimated_beta_return_per_bp"] * 100, abs=1e-5
        )
        assert "interpretation" in results
        assert out["assumptions"]["shock_bps"] == 100
        assert any("OLS" in lim or "historical" in lim for lim in out["limitations"])

    def test_bps_bounds(self):
        with pytest.raises(ScenarioError):
            run_scenario("rate_shock", {"symbol": SYMBOL, "shock_bps": 5000})


class TestNoFabrication:
    def test_results_carry_honest_labels(self):
        for kind, params in (
            ("price_shock", {"symbol": SYMBOL, "shock_pct": -5}),
            ("volatility_shock", {"symbol": SYMBOL, "scale": 1.2}),
        ):
            out = run_scenario(kind, params)
            assert out["kind"] == "hypothetical"
            assert set(out) >= {"scenario", "assumptions", "results", "limitations", "disclaimer"}
            assert "forecast" in out["disclaimer"]

    def test_params_must_be_mapping(self):
        with pytest.raises(ScenarioError):
            run_scenario("price_shock", ["not", "a", "dict"])
