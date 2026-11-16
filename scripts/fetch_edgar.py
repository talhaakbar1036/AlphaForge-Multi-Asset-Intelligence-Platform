#!/usr/bin/env python
"""Fetch recent SEC EDGAR filings for a ticker and index them into RAG.

Usage::

    python scripts/fetch_edgar.py NVDA
    python scripts/fetch_edgar.py AAPL --forms 10-K 10-Q --count 3
    python scripts/fetch_edgar.py MSFT --json

Fair-access compliance is enforced by construction: a descriptive User-Agent
with contact info (``ALPHAFORGE_EDGAR_USER_AGENT``), a minimum interval
between requests (default 0.2s, ``ALPHAFORGE_EDGAR_MIN_INTERVAL_S``), official
SEC endpoints only, and no API key requirement.
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ticker", help="stock ticker, e.g. NVDA")
    parser.add_argument(
        "--forms",
        nargs="+",
        default=None,
        help="EDGAR form types to fetch (default: configured, e.g. 10-K 10-Q)",
    )
    parser.add_argument("--count", type=int, default=None, help="max filings to ingest")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args(argv)

    from src.ai.config import AIConfig
    from src.ai.edgar import ingest_latest_filings
    from src.ai.errors import AIError
    from src.ai.runtime import build_runtime

    cfg = AIConfig.from_env()
    if not cfg.enabled:
        print("AI copilot is disabled - set ALPHAFORGE_AI_ENABLED=true", file=sys.stderr)
        return 2
    if not cfg.edgar_enabled:
        print(
            "EDGAR ingestion is disabled - set ALPHAFORGE_EDGAR_ENABLED=true",
            file=sys.stderr,
        )
        return 2

    runtime = build_runtime()
    try:
        result = ingest_latest_filings(
            runtime.ingestor,
            cfg,
            args.ticker,
            forms=args.forms,
            count=args.count,
        )
    except AIError as exc:
        print(f"EDGAR fetch failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        print(f"ticker  : {result.ticker} (CIK {result.cik})")
        print(
            f"filings : {result.filings_found} found, "
            f"{result.ingested_count} ingested, {result.skipped_count} skipped"
        )
        for item in result.details:
            mark = "ok" if item.get("ingested") else "skip"
            extra = "" if item.get("ingested") else f" - {item.get('error', '')}"
            print(f"  [{mark}] {item.get('form')} {item.get('accession')} {item.get('title')}{extra}")
        for line in result.skipped:
            print(f"  skipped: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
