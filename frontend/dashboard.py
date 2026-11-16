"""AlphaForge AI - Streamlit dashboard.

Run:  streamlit run frontend/dashboard.py

Per-symbol view: current price, investment outlook, risk, confidence and
BUY/HOLD/SELL recommendation with drill-downs into Technical / Market /
Macro / News / Risk / Historical Performance, plus an AI Copilot tab
(RAG chat over indexed documents + verified engine metrics).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.api.service as service  # noqa: E402

st.set_page_config(page_title="AlphaForge AI", page_icon="📈", layout="wide")

DISCLAIMER = (
    "⚠️ AlphaForge AI is decision-support, not a guaranteed-profit predictor. "
    "Outputs ship with confidence scores and explicit risks; this is not financial advice."
)


@st.cache_data(ttl=600, show_spinner=False)
def cached_decision(symbol: str) -> dict | None:
    try:
        return service.decision_payload(symbol)
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def cached_risk(symbol: str) -> dict | None:
    try:
        return service.risk_payload(symbol)
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def cached_backtest(symbol: str) -> dict | None:
    try:
        return service.backtest_payload(symbol)
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def cached_prediction(symbol: str) -> dict | None:
    try:
        return service.prediction_payload(symbol)
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def cached_models() -> dict | None:
    try:
        return service.models_payload()
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def copilot_runtime():
    """One shared AI runtime per process (embedding model loads once)."""
    try:
        from src.ai.runtime import get_runtime

        return get_runtime()
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def copilot_health() -> dict | None:
    runtime = copilot_runtime()
    if runtime is None:
        return None
    try:
        return runtime.health()
    except Exception:
        return None


def render_copilot_tab(symbol: str) -> None:
    """Chat with the Financial AI Copilot (RAG + quantitative engine)."""
    health = copilot_health()
    runtime = copilot_runtime()
    if runtime is None or health is None:
        st.info(
            "The AI copilot is unavailable in this environment. Install the "
            "optional dependencies (`pip install -r requirements-ai.txt`) and "
            "make sure `ALPHAFORGE_AI_ENABLED=true`."
        )
        return

    cfg = health.get("config", {})
    comps = health.get("components", {})
    llm = comps.get("llm", {})
    ollama_ok = bool((llm.get("ollama") or {}).get("reachable"))
    mode = "LLM + retrieval" if ollama_ok or cfg.get("hosted_provider_configured") else "retrieval-only (no LLM reachable)"
    st.caption(
        f"Provider mode: **{mode}** · index: **{comps.get('vector_store', '?')}** "
        f"({comps.get('indexed_documents', 0)} docs / {comps.get('indexed_chunks', 0)} chunks) · "
        f"embedding: `{comps.get('embedder', '?')}`"
    )
    if health.get("status") == "disabled":
        st.warning("The AI copilot is disabled (`ALPHAFORGE_AI_ENABLED=false`).")
        return
    if mode.startswith("retrieval-only"):
        st.caption(
            "Start Ollama (`ollama serve` + `ollama pull qwen2.5:3b`) or set "
            "`ALPHAFORGE_LLM_API_KEY` for hosted inference to enable generated answers."
        )

    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = []

    for message in st.session_state.ai_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_answer_details(message.get("payload"))

    opt_col1, opt_col2, _rest = st.columns([1, 1, 2])
    stream_on = opt_col1.toggle(
        "Stream answer",
        value=True,
        help="Yield answer text token-by-token instead of waiting for the full reply.",
    )

    question = st.chat_input(
        f"Ask about {symbol} (e.g. risks, metrics, filings, backtest assumptions)",
        max_chars=int(cfg.get("chat_question_max_chars", 4000)),
    )
    if question:
        st.session_state.ai_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            payload = {"error": "not answered"}
            try:
                if stream_on:
                    events: list[dict] = []

                    def _deltas():
                        for event in runtime.copilot.ask_stream(question, ticker=symbol):
                            events.append(event)
                            if event.get("type") == "delta" and event.get("text"):
                                yield event["text"]

                    with st.spinner("Gathering evidence..."):
                        st.write_stream(_deltas)
                    final = next(
                        (e for e in reversed(events) if e.get("type") == "final"), None
                    )
                    if final is not None:
                        payload = final
                        st.markdown(final.get("answer", ""))
                        _render_answer_details(final)
                    else:
                        err = next(
                            (e for e in events if e.get("type") == "error"), None
                        )
                        message = (err or {}).get("message", "stream ended without an answer")
                        st.error(f"Copilot error: {message}")
                        payload = {"error": message}
                else:
                    with st.spinner("Gathering evidence..."):
                        answer = runtime.copilot.ask(question, ticker=symbol)
                        payload = answer.as_dict()
                        st.markdown(answer.answer)
                        _render_answer_details(payload)
            except Exception as exc:  # noqa: BLE001 - surface, never crash the app
                st.error(f"Copilot error: {exc}")
                payload = {"error": str(exc)}
        st.session_state.ai_messages.append(
            {"role": "assistant", "content": payload.get("answer", "(no answer)"), "payload": payload}
        )

    with st.expander("Live provider verification"):
        st.caption(
            "Sends real probe prompts to the configured LLM (Ollama or hosted) "
            "and reports pass/fail + latency per probe. Nothing is mocked; this "
            "is the definitive check that generation works in this deployment."
        )
        if opt_col2.button("Run live verification"):
            with st.spinner("Probing provider..."):
                try:
                    from src.ai.verification import verify_runtime

                    report = verify_runtime(runtime).as_dict()
                    if report.get("ok"):
                        st.success(
                            f"Provider `{report.get('provider')}` verified "
                            f"({len(report.get('probes', []))} probes, model `{report.get('model')}`)"
                        )
                    else:
                        st.warning(report.get("error") or "verification failed")
                    for probe in report.get("probes", []):
                        label = "PASS" if probe.get("ok") else "FAIL"
                        kind = "" if probe.get("required") else " · advisory"
                        latency = (
                            f" · {probe.get('latency_ms')} ms"
                            if probe.get("latency_ms")
                            else ""
                        )
                        line = f"`[{label}]` **{probe.get('name')}**{latency}{kind} — {probe.get('detail')}"
                        (st.success if probe.get("ok") else st.warning)(line)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Verification failed: {exc}")

    with st.expander("Fetch latest SEC EDGAR filings (auto-index)"):
        st.caption(
            f"Downloads recent filings ({cfg.get('edgar_forms', '10-K,10-Q,8-K')}) for a "
            "ticker from official SEC endpoints (respecting their fair-access rules) "
            "and indexes them into RAG. Requires the admin token when configured."
        )
        edgar_col1, edgar_col2 = st.columns(2)
        edgar_ticker = edgar_col1.text_input("Ticker", value=symbol or "", key="edgar_ticker")
        edgar_count = edgar_col2.number_input(
            "Filings", min_value=1, max_value=20,
            value=int(cfg.get("edgar_max_filings", 5)), key="edgar_count",
        )
        if not cfg.get("edgar_enabled", False):
            st.caption("EDGAR auto-fetch is disabled (`ALPHAFORGE_EDGAR_ENABLED=false`).")
        elif st.button("Fetch & index filings", disabled=not edgar_ticker.strip()):
            with st.spinner("Contacting SEC EDGAR..."):
                try:
                    from src.ai.edgar import ingest_latest_filings

                    fetched = ingest_latest_filings(
                        runtime.ingestor,
                        runtime.cfg,
                        edgar_ticker.strip(),
                        count=int(edgar_count),
                    )
                    st.success(
                        f"{fetched.ticker}: {fetched.filings_found} found, "
                        f"{fetched.ingested_count} indexed, {fetched.skipped_count} skipped"
                    )
                    for item in fetched.details:
                        mark = "ok" if item.get("ingested") else "skip"
                        st.markdown(
                            f"- `[{mark}]` **{item.get('form')}** {item.get('filing_date')} "
                            f"— {item.get('title')}"
                        )
                    for line in fetched.skipped:
                        st.warning(line)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"EDGAR fetch failed: {exc}")

    with st.expander("Index documents (RAG)"):
        st.caption(
            "PDF, TXT, Markdown, HTML, JSON, CSV. Files are validated, stored "
            "locally and indexed; hosted LLMs only ever see retrieved snippets "
            "you explicitly send with a question."
        )
        upload = st.file_uploader(
            "Document", type=["pdf", "txt", "md", "markdown", "html", "htm", "json", "csv"]
        )
        up_col1, up_col2 = st.columns(2)
        up_ticker = up_col1.text_input("Ticker (optional)", value=symbol or "")
        up_type = up_col2.text_input("Document type (optional)", value="")
        if st.button("Index document", disabled=upload is None) and upload is not None:
            try:
                result = runtime.ingestor.ingest_bytes(
                    upload.getvalue(),
                    upload.name or "upload",
                    ticker=up_ticker.strip() or None,
                    doc_type=up_type.strip() or None,
                )
                st.success(
                    f"{result.status}: `{result.filename}` → {result.chunks} chunk(s) "
                    f"({result.document_id})"
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Ingestion failed: {exc}")


def _render_answer_details(payload: dict | None) -> None:
    """Sources, engine metrics and limitations for one copilot answer."""
    if not payload or payload.get("error"):
        return
    meta1, meta2, meta3 = st.columns(3)
    meta1.metric("Mode", payload.get("generated_by", "?"))
    meta2.metric("Category", payload.get("category", "?"))
    meta3.metric("Provider", payload.get("provider", "?"))

    sources = payload.get("sources") or []
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for src in sources:
                page = f", p.{src['page_number']}" if src.get("page_number") else ""
                date = f" ({src['publication_date']})" if src.get("publication_date") else ""
                st.markdown(
                    f"**[{src.get('retrieval_rank') or src.get('score')}]** "
                    f"{src.get('title') or src.get('filename') or src.get('document_id')}"
                    f"{date}{page} · score {src.get('score')}"
                )
                if src.get("text_preview"):
                    st.text(src["text_preview"][:400])

    metrics = payload.get("market_metrics") or {}
    decision = metrics.get("decision")
    risk = metrics.get("risk")
    if decision or risk:
        with st.expander("Verified engine metrics"):
            if decision:
                st.markdown(
                    f"**Decision ({decision.get('symbol')}, {decision.get('as_of')}):** "
                    f"{decision.get('recommendation')} · score {decision.get('investment_score')} "
                    f"· confidence {decision.get('confidence')}"
                )
            if risk:
                cols = st.columns(4)
                cols[0].metric("Volatility", f"{(risk.get('volatility') or float('nan')):.1%}")
                cols[1].metric("Max DD", f"{(risk.get('max_drawdown') or float('nan')):.1%}")
                cols[2].metric("Sharpe", f"{(risk.get('sharpe') or float('nan')):.2f}")
                cols[3].metric("Risk band", str(risk.get("risk_band", "?")))

    limitations = payload.get("limitations") or []
    if limitations:
        with st.expander(f"Limitations & caveats ({len(limitations)})"):
            for item in limitations:
                st.warning(item)


def main() -> None:
    st.title("AlphaForge AI - Multi-Asset Intelligence")
    st.caption(DISCLAIMER)

    ready = service.available_symbols()
    if not ready:
        st.warning("No processed data found. Run `python scripts/run_pipeline.py --skip-collect` first.")
        return

    symbol = st.selectbox("Select Company", ready)
    decision = cached_decision(symbol)
    if decision is None:
        st.error(f"Could not build a decision for {symbol}.")
        return

    score = decision["investment_score"]
    outlook = "Bullish" if score > 0.05 else ("Bearish" if score < -0.05 else "Neutral")

    # ---- headline row ------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"${decision['last_close']:,.2f}")
    c2.metric("Outlook", outlook, f"score {score:+.2f}")
    c3.metric("Risk", f"{decision['risk_band']} ({decision['risk_score']}/100)")
    c4.metric("Confidence", f"{decision['confidence']*100:.0f}%")

    color = {"BUY": "green", "SELL": "red"}.get(decision["recommendation"], "orange")
    st.markdown(f"## Recommendation: :{color}[{decision['recommendation']}]")

    left, right = st.columns(2)
    with left:
        st.markdown("### Why")
        for reason in decision["why"]:
            st.markdown(f"- ✅ {reason}")
    with right:
        st.markdown("### Risks")
        for risk in decision["risks"]:
            st.markdown(f"- ⚠️ {risk}")

    with st.expander("Score components"):
        st.dataframe(pd.DataFrame([decision["components"]]).T.rename(columns={0: "score"}).round(3))

    # ---- drill-downs -------------------------------------------------------
    tab_tech, tab_mkt, tab_macro, tab_news, tab_risk, tab_ml, tab_hist, tab_ai = st.tabs(
        [
            "Technical Analysis",
            "Market Analysis",
            "Macro Analysis",
            "News Sentiment",
            "Risk Analysis",
            "ML Direction",
            "Historical Performance",
            "AI Copilot",
        ]
    )

    table = service.load_feature_table(symbol)
    with tab_tech:
        if table is not None:
            recent = table["Close"].tail(252)
            st.line_chart(recent, height=260)
            cols = [c for c in ("rsi_14", "macd_hist", "bb_pctb", "realized_vol_21") if c in table.columns]
            st.dataframe(table[cols].tail(5).round(3).T.rename(columns=lambda c: str(c.date())))

    with tab_mkt:
        ctx_cols = [c for c in (table.columns if table is not None else []) if c.startswith("ctx_")]
        if ctx_cols:
            latest = table[ctx_cols].tail(1).T.rename(columns={table.index[-1]: "value"})
            latest["group"] = [c.split("_")[1] for c in latest.index]
            st.bar_chart(latest["value"].astype(float))
            st.dataframe(latest.round(4))
        else:
            st.info("No cross-asset context features available.")

    with tab_macro:
        macro_path = service.CONFIG.data_interim / "macro_daily.parquet"
        if macro_path.exists():
            macro = pd.read_parquet(macro_path).tail(504)
            show = [c for c in ("fed_funds_rate", "cpi_yoy", "unemployment", "dollar_index") if c in macro.columns]
            if show:
                st.line_chart(macro[show], height=280)
            st.dataframe(macro.tail(3).round(3))
        else:
            st.info("Macro data not processed yet.")

    with tab_news:
        sent = table["news_sentiment"].tail(120) if table is not None and "news_sentiment" in table.columns else None
        if sent is not None:
            st.area_chart(sent, height=240)
            st.metric("Latest daily sentiment", f"{sent.iloc[-1]:+.2f}", f"{sent.diff().iloc[-1]:+.2f} vs prev")
        else:
            st.info("No news sentiment data.")

    with tab_risk:
        rp = cached_risk(symbol) or {}
        if rp:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Volatility (ann.)", f"{rp.get('volatility', float('nan')):.1%}")
            m2.metric("Max Drawdown", f"{rp.get('max_drawdown', float('nan')):.1%}")
            m3.metric("Sharpe", f"{rp.get('sharpe', float('nan')):.2f}")
            m4.metric("Risk Score", f"{rp.get('risk_score', 0)}/100 ({rp.get('risk_band', '?')})")

    with tab_ml:
        pred = cached_prediction(symbol)
        if pred:
            mc1, mc2, mc3, mc4 = st.columns(4)
            direction = pred.get("prediction", "?")
            mc1.metric("Direction", direction)
            mc2.metric("P(up)", f"{pred.get('probability_up', 0):.1%}")
            mc3.metric("Confidence", f"{pred.get('confidence', 0):.0%}")
            mc4.metric("As of", pred.get("as_of", "?"))
            st.progress(float(pred.get("probability_up", 0.5)))
            st.caption(
                "Next-session direction only. The probability is a research "
                "output and is not blended into the recommendation above."
            )
            test = pred.get("training_metrics") or {}
            st.dataframe(
                pd.DataFrame(
                    {
                        "value": {
                            "Test accuracy": test.get("accuracy"),
                            "Majority baseline": test.get("majority_baseline_accuracy"),
                            "ROC AUC": test.get("roc_auc"),
                            "Test window": f"{test.get('test_start', '?')} to {test.get('test_end', '?')}",
                        }
                    }
                )
            )
            st.warning(pred.get("disclaimer", ""))
        else:
            catalog = cached_models() or {}
            trained = [m.get("symbol") for m in catalog.get("models", [])]
            st.info(
                "No ML direction prediction is available for this symbol. "
                f"Pretrained models exist for: {', '.join(trained) if trained else 'none'}. "
                "Train with `python scripts/train_models.py --symbols "
                f"{symbol}` (requires the optional `ml` extra)."
            )
            if not catalog.get("dependencies", {}).get("tensorflow", True):
                st.caption("TensorFlow is not installed in this environment.")

    with tab_hist:
        bt = cached_backtest(symbol)
        if bt and "comparison" in bt:
            st.caption(f"Strategy: {bt.get('strategy', '?')} | window {bt.get('window', {}).get('start', '?')} → {bt.get('window', {}).get('end', '?')}")
            st.dataframe(pd.DataFrame(bt["comparison"]).T)
        else:
            st.info("Backtest unavailable for this symbol.")

    with tab_ai:
        render_copilot_tab(symbol)


if __name__ == "__main__":
    main()
