"""AlphaForge AI end-to-end pipeline runner.

Usage:
    python scripts/run_pipeline.py --help
    python scripts/run_pipeline.py                       # full pipeline (collect -> serve-ready)
    python scripts/run_pipeline.py --skip-collect        # offline: reuse cached raw data
    python scripts/run_pipeline.py --symbols NVDA AAPL   # subset of the universe

Steps: collect -> preprocess -> features.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.data_fetcher import collect_all  # noqa: E402
from src.data.data_processor import (  # noqa: E402
    align_macro_daily,
    build_interim,
    load_raw_macro,
    load_raw_news,
    save_interim,
)
from src.data.feature_engineering import build_feature_table, macro_features, make_targets, save_processed  # noqa: E402
from src.utils.config import CONFIG, DATA_START, TICKERS  # noqa: E402
from src.utils.helpers import sanitize_symbol  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)


def step_collect(args) -> None:
    if args.skip_collect:
        log.info("skipping collection (offline mode - using cached raw data)")
        return
    # TICKERS only holds price groups; macro + news have dedicated sources
    groups = list(TICKERS.keys()) + ["macro", "news"]
    try:
        collect_all(groups=groups, start=DATA_START)
    except Exception as exc:  # noqa: BLE001
        log.warning("live collection failed (%s); continuing with cached data", exc)


def step_preprocess(symbols: list[str]) -> None:
    for sym in symbols:
        bundle = build_interim(sym, "stocks")
        save_interim(bundle, sym)
    try:
        macro_m = load_raw_macro()
        price_path = CONFIG.data_interim / f"{sanitize_symbol(symbols[0])}_price.parquet"
        ref = pd.read_parquet(price_path).index if price_path.exists() else build_interim(symbols[0], "stocks")["price"].index
        daily = macro_features(align_macro_daily(macro_m, ref))
        CONFIG.data_interim.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(CONFIG.data_interim / "macro_daily.parquet")
    except Exception as exc:  # noqa: BLE001
        log.warning("macro alignment skipped: %s", exc)


def step_features(symbols: list[str]) -> dict[str, int]:
    summary = {}
    for sym in symbols:
        news_df = load_raw_news(sym)
        if news_df is None:
            log.info("no raw news for %s - sentiment feature will be 0", sym)
        table = build_feature_table(sym, news_df=news_df)
        table = make_targets(table)
        path = save_processed(table, sym)
        summary[sym] = len(table)
        log.info("features for %s: %d rows x %d cols -> %s", sym, len(table), table.shape[1], path.name)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaForge AI pipeline")
    parser.add_argument("--skip-collect", action="store_true", help="use cached raw data only (offline)")
    parser.add_argument("--symbols", nargs="*", default=None, help="subset of stock tickers")
    args = parser.parse_args()

    CONFIG.ensure_dirs()
    symbols = args.symbols or TICKERS["stocks"]

    t0 = time.time()
    log.info("=== AlphaForge AI pipeline | symbols=%s ===", symbols)

    step_collect(args)
    step_preprocess(symbols)
    step_features(symbols)

    log.info("pipeline finished in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
