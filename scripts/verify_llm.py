#!/usr/bin/env python
"""Run live verification probes against the configured LLM provider.

Usage::

    python scripts/verify_llm.py                 # text report, exit 1 if failing
    python scripts/verify_llm.py --json          # machine-readable report
    python scripts/verify_llm.py --stream        # include a streaming probe
    python scripts/verify_llm.py --timeout 15    # per-probe timeout override

This performs **real HTTP calls** to the configured provider (Ollama or a
hosted API). Nothing is mocked: that is the point - it answers "does the LLM
actually respond in this deployment?". No API keys or request bodies are ever
printed, only pass/fail, latency and provider metadata.
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument("--stream", action="store_true", help="include the streaming probe")
    parser.add_argument("--timeout", type=float, default=None, help="per-probe timeout in seconds")
    args = parser.parse_args(argv)

    from src.ai.config import AIConfig
    from src.ai.runtime import build_runtime
    from src.ai.verification import verify_provider

    cfg = AIConfig.from_env()
    if not cfg.enabled:
        print("AI copilot is disabled - set ALPHAFORGE_AI_ENABLED=true", file=sys.stderr)
        return 2

    runtime = build_runtime()
    report = verify_provider(
        runtime.provider, cfg, timeout_s=args.timeout, include_streaming=args.stream
    )

    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(f"provider : {report.provider}")
        print(f"ok       : {report.ok}")
        if report.error:
            print(f"error    : {report.error}")
        for probe in report.probes:
            status = "PASS" if probe.ok else "FAIL"
            tag = "" if probe.required else " (advisory)"
            latency = f"{probe.latency_ms:.0f} ms" if probe.latency_ms else "-"
            print(f"  [{status}] {probe.name:<22} {latency:>10}  {probe.detail}{tag}")
        if not report.ok:
            print(
                "\nverification FAILED - check the provider is running and "
                "ALPHAFORGE_LLM_* is configured (see docs/FINANCIAL_AI_COPILOT.md)",
                file=sys.stderr,
            )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
