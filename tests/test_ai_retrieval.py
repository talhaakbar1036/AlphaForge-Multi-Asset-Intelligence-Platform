"""Retrieval: relevance, metadata filters, citations, empty results."""
from __future__ import annotations

import pytest

from src.ai.errors import RetrievalError
from src.ai.retrieval import RetrievalFilters

NVDA_DOC = (
    "NVIDIA risk factors annual report. Data-center demand is concentrated "
    "among a few hyperscale customers. Export controls restrict sales of "
    "advanced GPUs to certain markets. Foundry supply constraints may limit "
    "shipment volumes. Competition in accelerated computing is increasing."
)
AAPL_DOC = (
    "Apple quarterly earnings release. Services revenue grew double digits. "
    "Gross margin expanded. The company announced an increased share "
    "repurchase program and raised the dividend."
)
MSFT_DOC = (
    "Microsoft annual report risk factors. Regulatory scrutiny of cloud "
    "services and artificial intelligence products continues to grow "
    "across multiple jurisdictions. Cybersecurity incidents remain a risk."
)


@pytest.fixture
def indexed(ai_runtime):
    """Three documents with distinct tickers, types and dates."""
    a = ai_runtime.ingestor.ingest_text(
        NVDA_DOC,
        "NVDA_10-K_2025.txt",
        ticker="NVDA",
        doc_type="10-K",
        publication_date="2025-02-26",
        title="NVIDIA 10-K 2025",
    )
    b = ai_runtime.ingestor.ingest_text(
        AAPL_DOC,
        "AAPL_earnings_Q2.txt",
        ticker="AAPL",
        doc_type="earnings",
        publication_date="2024-05-02",
        title="Apple Q2 Earnings",
    )
    c = ai_runtime.ingestor.ingest_text(
        MSFT_DOC,
        "MSFT_10-K_2024.txt",
        ticker="MSFT",
        doc_type="10-K",
        publication_date="2024-07-30",
        title="Microsoft 10-K 2024",
    )
    return ai_runtime, [a, b, c]


class TestRelevance:
    def test_relevant_document_ranks_first(self, indexed):
        runtime, _ = indexed
        hits = runtime.retriever.retrieve("export controls on advanced GPUs", top_k=3)
        assert hits
        assert hits[0].ticker == "NVDA"
        assert hits[0].rank == 1

    def test_citation_metadata_complete(self, indexed):
        runtime, results = indexed
        hits = runtime.retriever.retrieve("share repurchase dividend", top_k=3)
        top = hits[0]
        assert top.document_id == results[1].document_id
        assert top.title == "Apple Q2 Earnings"
        assert top.doc_type == "earnings"
        assert top.publication_date == "2024-05-02"
        assert top.chunk_id.startswith(top.document_id)
        assert top.citation  # human-readable source string
        assert "Apple" in top.citation
        payload = top.as_dict()
        assert payload["text_preview"]
        assert payload["score"] <= 1.0

    def test_empty_query_returns_nothing(self, indexed):
        runtime, _ = indexed
        assert runtime.retriever.retrieve("   ") == []

    def test_empty_index_returns_nothing(self, ai_runtime):
        assert ai_runtime.retriever.retrieve("anything at all") == []

    def test_top_k_respected(self, indexed):
        runtime, _ = indexed
        hits = runtime.retriever.retrieve("risk", top_k=2)
        assert len(hits) <= 2


class TestFilters:
    def test_ticker_filter(self, indexed):
        runtime, _ = indexed
        hits = runtime.retriever.retrieve(
            "risk factors", filters=RetrievalFilters(tickers=["NVDA"])
        )
        assert hits
        assert {h.ticker for h in hits} == {"NVDA"}

    def test_doc_type_filter(self, indexed):
        runtime, _ = indexed
        hits = runtime.retriever.retrieve(
            "report", filters=RetrievalFilters(doc_types=["earnings"])
        )
        assert hits
        assert {h.doc_type for h in hits} == {"earnings"}

    def test_date_range_filter(self, indexed):
        runtime, _ = indexed
        hits = runtime.retriever.retrieve(
            "report revenue",
            filters=RetrievalFilters(published_from="2025-01-01", published_to="2025-12-31"),
        )
        assert hits
        assert {h.publication_date for h in hits} == {"2025-02-26"}

    def test_date_filter_excludes_out_of_range(self, indexed):
        runtime, _ = indexed
        hits = runtime.retriever.retrieve(
            "report",
            filters=RetrievalFilters(published_from="2026-01-01"),
        )
        assert hits == []

    def test_invalid_date_raises(self, indexed):
        runtime, _ = indexed
        with pytest.raises(RetrievalError):
            runtime.retriever.retrieve(
                "report", filters=RetrievalFilters(published_from="not-a-date")
            )

    def test_no_match_returns_empty(self, indexed):
        runtime, _ = indexed
        hits = runtime.retriever.retrieve(
            "quantum blockchain metaverse", filters=RetrievalFilters(tickers=["TSLA"])
        )
        # TSLA has no document; filter excludes everything
        assert hits == []

    def test_private_documents_hidden_by_default(self, ai_runtime):
        ai_runtime.ingestor.ingest_text(
            "confidential internal merger discussion", "memo.txt", visibility="private"
        )
        assert ai_runtime.retriever.retrieve("merger discussion") == []
        hits = ai_runtime.retriever.retrieve(
            "merger discussion", filters=RetrievalFilters(include_private=True)
        )
        assert len(hits) == 1

    def test_filter_describe_is_serialisable(self):
        import json

        payload = RetrievalFilters(tickers=["nvda"]).describe()
        assert json.dumps(payload)
        assert payload["tickers"] == ["nvda"]


class TestHybrid:
    def test_keyword_overlap_breaks_ties(self, ai_runtime):
        # Two near-identical semantic docs; only one contains the rare token.
        ai_runtime.ingestor.ingest_text(
            "general market commentary about equities and bonds", "generic.txt"
        )
        ai_runtime.ingestor.ingest_text(
            "general market commentary about equities and bonds, "
            "with a special note on zebraCorp restructuring",
            "zebra.txt",
        )
        hybrid = ai_runtime.retriever.retrieve("zebraCorp restructuring", hybrid=True)
        pure = ai_runtime.retriever.retrieve("zebraCorp restructuring", hybrid=False)
        assert hybrid[0].filename == "zebra.txt"
        assert pure  # semantic search also returns something
