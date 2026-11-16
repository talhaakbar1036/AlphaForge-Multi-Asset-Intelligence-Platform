"""SEC EDGAR auto-fetch: download public filings and feed them into RAG.

Uses only official SEC endpoints:

* ticker -> CIK map: ``https://www.sec.gov/files/company_tickers.json``
  (cached on disk for 7 days);
* submissions: ``https://data.sec.gov/submissions/CIK##########.json``;
* filing documents: ``https://www.sec.gov/Archives/edgar/data/...``.

SEC fair-access rules are respected: a descriptive ``User-Agent`` with
contact info (``ALPHAFORGE_EDGAR_USER_AGENT``) and a client-side minimum
interval between requests (default 0.2 s, well under the 10 req/s cap).
No API key is required for these public endpoints.

Downloaded filings are treated like any other upload: validated, extracted
with the existing HTML/text extractors, chunked, embedded and indexed with
full provenance (source URL, filing date, form type, accession number).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from src.ai.config import AIConfig
from src.ai.errors import EdgarError
from src.ai.ingest import DocumentIngestor, IngestResult
from src.utils.logger import get_logger

log = get_logger(__name__)

_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodot}/{document}"

_TICKER_MAP_TTL_S = 7 * 24 * 3600
_MAP_FILENAME = "company_tickers.json"

# EDGAR form -> copilot doc_type
_FORM_DOC_TYPES = {
    "10-K": "10-K",
    "10-Q": "10-Q",
    "8-K": "8-K",
    "S-1": "filing",
    "424B5": "filing",
}


@dataclass
class EdgarFiling:
    """One recent filing discovered via the submissions API."""

    ticker: str
    cik: int
    form: str
    accession: str
    filing_date: str
    report_date: str | None = None
    primary_document: str | None = None
    primary_document_url: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "cik": self.cik,
            "form": self.form,
            "accession": self.accession,
            "filing_date": self.filing_date,
            "report_date": self.report_date,
            "primary_document": self.primary_document,
            "primary_document_url": self.primary_document_url,
        }


@dataclass
class EdgarFetchResult:
    """Outcome of fetching + ingesting filings for one ticker."""

    ticker: str
    cik: int
    filings_found: int = 0
    ingested: list[IngestResult] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    details: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ingested_count(self) -> int:
        return len(self.ingested)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "cik": self.cik,
            "filings_found": self.filings_found,
            "ingested": [r.to_dict() for r in self.ingested],
            "ingested_count": self.ingested_count,
            "skipped": self.skipped,
            "skipped_count": self.skipped_count,
            "details": self.details,
        }


class _MinIntervalThrottle:
    """Enforce a minimum wall-clock interval between requests."""

    def __init__(self, min_interval_s: float):
        self.min_interval = float(min_interval_s)
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


class EdgarClient:
    """Read-only SEC EDGAR client (no auth, public data only)."""

    def __init__(self, cfg: AIConfig):
        self.cfg = cfg
        self._throttle = _MinIntervalThrottle(cfg.edgar_min_interval_s)
        self._ticker_map: dict[str, int] | None = None

    # ------------------------------------------------------------ helpers --
    def _headers(self) -> dict[str, str]:
        return {"User-Agent": self.cfg.edgar_user_agent, "Accept-Encoding": "gzip, deflate"}

    def _get(self, url: str) -> bytes:
        if not self.cfg.edgar_enabled:
            raise EdgarError(
                "EDGAR fetching is disabled (set ALPHAFORGE_EDGAR_ENABLED=true)"
            )
        self._throttle.wait()
        try:
            response = requests.get(url, headers=self._headers(), timeout=self.cfg.edgar_timeout_s)
        except requests.Timeout as exc:
            raise EdgarError(f"SEC EDGAR timed out fetching {url}") from exc
        except requests.RequestException as exc:
            raise EdgarError(f"SEC EDGAR is unreachable: {exc}") from exc
        if response.status_code == 404:
            raise EdgarError(f"SEC EDGAR has no resource at {url} (404)")
        if not response.ok:
            raise EdgarError(
                f"SEC EDGAR answered HTTP {response.status_code} for {url} - "
                "check ALPHAFORGE_EDGAR_USER_AGENT (SEC requires contact info)"
            )
        return response.content

    # --------------------------------------------------------- ticker->CIK --
    def _load_ticker_map(self, force: bool = False) -> dict[str, int]:
        if self._ticker_map is not None and not force:
            return self._ticker_map
        cache = self.cfg.edgar_cache_dir / _MAP_FILENAME
        if not force and cache.is_file() and (time.time() - cache.stat().st_mtime) < _TICKER_MAP_TTL_S:
            try:
                payload = json.loads(cache.read_text(encoding="utf-8"))
                self._ticker_map = _parse_ticker_map(payload)
                return self._ticker_map
            except (ValueError, OSError) as exc:
                log.warning("ignoring corrupt EDGAR ticker cache: %s", exc)
        raw = self._get(_TICKER_MAP_URL)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise EdgarError("SEC EDGAR ticker map was not valid JSON") from exc
        self._ticker_map = _parse_ticker_map(payload)
        try:
            self.cfg.edgar_cache_dir.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(payload), encoding="utf-8")
        except OSError as exc:  # cache is best effort
            log.warning("could not cache EDGAR ticker map: %s", exc)
        return self._ticker_map

    def cik_for(self, ticker: str) -> int:
        cleaned = (ticker or "").strip().upper().lstrip("^")
        if not cleaned:
            raise EdgarError("a ticker is required")
        mapping = self._load_ticker_map()
        cik = mapping.get(cleaned)
        if cik is None:
            raise EdgarError(
                f"'{cleaned}' was not found in SEC company_tickers.json "
                "(foreign issuers and some ETFs are not in this map)"
            )
        return cik

    # ------------------------------------------------------------- filings --
    def list_filings(
        self,
        ticker: str,
        forms: list[str] | None = None,
        count: int | None = None,
    ) -> list[EdgarFiling]:
        """Latest filings for ``ticker`` filtered to the requested forms."""
        cik = self.cik_for(ticker)
        raw = self._get(_SUBMISSIONS_URL.format(cik=cik))
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise EdgarError(f"submissions payload for CIK {cik} was not valid JSON") from exc
        recent = ((payload.get("filings") or {}).get("recent")) or {}
        forms_col = recent.get("form") or []
        accessions = recent.get("accessionNumber") or []
        dates = recent.get("filingDate") or []
        reports = recent.get("reportDate") or []
        docs = recent.get("primaryDocument") or []

        wanted = {f.strip().upper() for f in (forms or self._parse_forms())}
        limit = int(count or self.cfg.edgar_max_filings)
        limit = max(1, min(limit, 20))

        filings: list[EdgarFiling] = []
        for i, form in enumerate(forms_col):
            if str(form).upper() not in wanted:
                continue
            if i >= len(accessions) or i >= len(dates):
                break
            accession = str(accessions[i])
            filing = EdgarFiling(
                ticker=ticker.upper(),
                cik=cik,
                form=str(form),
                accession=accession,
                filing_date=str(dates[i]),
                report_date=str(reports[i]) if i < len(reports) and reports[i] else None,
                primary_document=str(docs[i]) if i < len(docs) and docs[i] else None,
            )
            if filing.primary_document:
                filing.primary_document_url = _ARCHIVES_URL.format(
                    cik=cik,
                    accession_nodot=accession.replace("-", ""),
                    document=filing.primary_document,
                )
            filings.append(filing)
            if len(filings) >= limit:
                break
        return filings

    def _parse_forms(self) -> list[str]:
        return [f.strip().upper() for f in self.cfg.edgar_forms.split(",") if f.strip()]

    def download_filing(self, filing: EdgarFiling) -> bytes:
        if not filing.primary_document_url:
            raise EdgarError(
                f"{filing.form} ({filing.accession}) has no primary document to download"
            )
        return self._get(filing.primary_document_url)


def _parse_ticker_map(payload: Any) -> dict[str, int]:
    """Parse SEC's company_tickers.json: {"fields":..., "data": [[cik, ticker, title]...]}."""
    if isinstance(payload, dict) and "data" in payload:
        rows = payload.get("data") or []
        fields = payload.get("fields") or []
        try:
            cik_idx = fields.index("cik")
            ticker_idx = fields.index("ticker")
        except ValueError:
            cik_idx, ticker_idx = 0, 1
        out: dict[str, int] = {}
        for row in rows:
            try:
                out[str(row[ticker_idx]).upper()] = int(row[cik_idx])
            except (IndexError, TypeError, ValueError):
                continue
        return out
    if isinstance(payload, dict):  # legacy {"0": {"cik_str":..,"ticker":..}, ...}
        out = {}
        for item in payload.values():
            if isinstance(item, dict) and item.get("ticker"):
                try:
                    out[str(item["ticker"]).upper()] = int(item.get("cik_str") or item.get("cik"))
                except (TypeError, ValueError):
                    continue
        return out
    raise EdgarError("unexpected company_tickers.json shape")


# ------------------------------------------------------------------- ingest --
def ingest_latest_filings(
    ingestor: DocumentIngestor,
    cfg: AIConfig,
    ticker: str,
    *,
    forms: list[str] | None = None,
    count: int | None = None,
    client: EdgarClient | None = None,
) -> EdgarFetchResult:
    """Fetch the latest EDGAR filings for ``ticker`` and index them into RAG."""
    client = client or EdgarClient(cfg)
    filings = client.list_filings(ticker, forms=forms, count=count)
    result = EdgarFetchResult(
        ticker=ticker.upper(),
        cik=filings[0].cik if filings else client.cik_for(ticker),
        filings_found=len(filings),
    )
    if not filings:
        result.skipped.append("no filings matched the requested forms")
        return result
    for filing in filings:
        detail = {
            "form": filing.form,
            "accession": filing.accession,
            "filing_date": filing.filing_date,
            "title": f"{filing.ticker} {filing.form} filed {filing.filing_date}",
            "source_url": filing.primary_document_url,
            "ingested": False,
        }
        try:
            data = client.download_filing(filing)
            name = f"{filing.ticker}_{filing.form}_{filing.filing_date}_{filing.accession}.html"
            ingest_result = ingestor.ingest_bytes(
                data,
                name,
                ticker=filing.ticker,
                doc_type=_FORM_DOC_TYPES.get(filing.form.upper(), "filing"),
                title=detail["title"],
                publication_date=filing.filing_date,
                reporting_period=filing.report_date,
                source_url=filing.primary_document_url,
                replace=True,
            )
            result.ingested.append(ingest_result)
            detail["ingested"] = True
            detail["document_id"] = ingest_result.document_id
            log.info(
                "EDGAR: indexed %s %s (%s) -> %s",
                filing.ticker,
                filing.form,
                filing.filing_date,
                ingest_result.document_id,
            )
        except EdgarError as exc:
            result.skipped.append(f"{filing.form} {filing.accession}: {exc}")
            detail["error"] = str(exc)
        except Exception as exc:  # noqa: BLE001 - one bad filing must not kill the batch
            result.skipped.append(f"{filing.form} {filing.accession}: {type(exc).__name__}: {exc}")
            detail["error"] = f"{type(exc).__name__}: {exc}"
            log.warning("EDGAR ingest failed for %s: %s", filing.accession, exc)
        result.details.append(detail)
    return result
