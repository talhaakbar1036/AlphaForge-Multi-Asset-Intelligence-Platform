"""Train next-session direction LSTMs on AlphaForge price artifacts.

Examples:
    python scripts/train_models.py --symbols AAPL MSFT
    python scripts/train_models.py --epochs 30 --output-dir artifacts/direction_lstm

Requires the optional ML extra:
    pip install -e ".[ml]"   # or: pip install -r requirements-ml.txt

Existing artifacts are replaced atomically per symbol. Metrics always include
the majority-class baseline so a near-coin-flip model is reported honestly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.data import load_price_history  # noqa: E402
from src.models.errors import ModelDependencyError  # noqa: E402
from src.models.trainer import train_symbol, training_summary  # noqa: E402
from src.utils.config import CONFIG, TICKERS  # noqa: E402
from src.utils.helpers import sanitize_symbol  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)


def _default_symbols() -> list[str]:
    return [
        symbol
        for symbol in TICKERS["stocks"]
        if (CONFIG.data_processed / f"{sanitize_symbol(symbol)}_features.parquet").exists()
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AlphaForge direction LSTMs")
    parser.add_argument("--symbols", nargs="+", help="symbols to train (default: all with features)")
    parser.add_argument("--data-dir", type=Path, default=None, help="processed feature-table directory")
    parser.add_argument("--output-dir", type=Path, default=CONFIG.model_artifacts_dir)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    symbols = [s.upper() for s in (args.symbols or _default_symbols())]
    if not symbols:
        log.error("no symbols with processed feature tables; run scripts/run_pipeline.py first")
        return 1

    cfg = replace(
        CONFIG.direction_model,
        **{
            key: value
            for key, value in {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "sequence_length": args.sequence_length,
                "seed": args.seed,
            }.items()
            if value is not None
        },
    )
    try:
        cfg.validate()
    except ValueError as exc:
        log.error("invalid configuration: %s", exc)
        return 2

    reports: list[dict] = []
    failures: dict[str, str] = {}
    for symbol in symbols:
        try:
            prices = load_price_history(symbol, data_dir=args.data_dir)
            report = train_symbol(symbol, prices, cfg=cfg, output_dir=args.output_dir)
        except ModelDependencyError as exc:
            log.error("%s", exc)
            return 2
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop a run
            log.error("%s failed: %s", symbol, exc)
            failures[symbol] = str(exc)
            continue
        reports.append(report)
        log.info(
            "%s: accuracy %.4f vs baseline %.4f, AUC %s",
            symbol,
            report["test_accuracy"],
            report["test_majority_class_baseline_accuracy"],
            report["test_roc_auc"],
        )

    summary = training_summary(reports, failures, cfg, args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log.info("wrote %d model(s) and %d failure(s) to %s", len(reports), len(failures), summary_path)
    return 0 if reports else 1


if __name__ == "__main__":
    raise SystemExit(main())
