"""Print next-session direction probabilities from trained artifacts.

Usage:
    python scripts/predict_direction.py AAPL MSFT
    python scripts/predict_direction.py NVDA --json

The output is a research signal, not a tradable recommendation. Requires the
optional ML extra (TensorFlow + scikit-learn + joblib).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.data import load_price_history  # noqa: E402
from src.models.predictor import DirectionPredictor  # noqa: E402
from src.utils.config import CONFIG  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict next-session direction")
    parser.add_argument("symbols", nargs="+", help="ticker symbols")
    parser.add_argument("--artifacts-dir", type=Path, default=CONFIG.model_artifacts_dir)
    parser.add_argument("--json", action="store_true", help="emit a JSON array")
    args = parser.parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    payloads = []
    failures = 0
    for symbol in args.symbols:
        symbol = symbol.upper()
        try:
            prices = load_price_history(symbol)
            payload = DirectionPredictor.load(symbol, args.artifacts_dir).predict(prices)
        except Exception as exc:  # noqa: BLE001 - report per symbol and continue
            failures += 1
            log.error("%s: %s", symbol, exc)
            continue
        payloads.append(payload)
        if not args.json:
            print(
                f"{symbol}: {payload['prediction']} "
                f"(P(up)={payload['probability_up']:.3f}, "
                f"as_of={payload['as_of']})"
            )

    if args.json:
        print(json.dumps(payloads, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
