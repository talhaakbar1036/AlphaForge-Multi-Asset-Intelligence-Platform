"""Document ingestion: extraction, chunking, duplicates, updates, deletion."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.ai.chunking import Chunk, chunk_text
from src.ai.embeddings import HashingEmbedder
from src.ai.errors import (
    DocumentError,
    DocumentTooLargeError,
    EmptyDocumentError,
    ExtractionError,
    InvalidFilenameError,
    UnsupportedFormatError,
)
from src.ai.extractors import extract_bytes
from src.ai.ingest import make_document_id

NVDA_RISK_TEXT = (
    "NVIDIA Corporation - Annual Report 2025\n\n"
    "Item 1A. Risk Factors. Our results depend on demand for data-center GPUs "
    "and on continued hyperscaler capital expenditure. Supply chain "
    "constraints at foundry partners could limit our ability to ship "
    "products. Export restrictions on advanced semiconductors to certain "
    "markets may materially reduce our revenue.\n\n"
    "Competition in accelerated computing is intensifying from both "
    "incumbent and custom-silicon vendors."
)

AAPL_TEXT = (
    "Apple Inc. quarterly report. Services revenue reached a new record "
    "while iPhone revenue was flat year over year. The company increased "
    "the quarterly dividend and continued share repurchases."
)


# --------------------------------------------------------------- chunking --
class TestChunking:
    def test_short_text_single_chunk(self):
        chunks = chunk_text("hello world", chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0][1] == "hello world"

    def test_long_text_overlaps_and_covers(self):
        text = ("Sentence about risk factors. " * 80).strip()
        chunks = chunk_text(text, chunk_size=200, overlap=40)
        assert len(chunks) > 1
        joined = " ".join(piece for _, piece, _ in chunks)
        assert "risk factors" in joined
        # every chunk respects the size cap
        assert all(len(piece) <= 200 for _, piece, _ in chunks)

    def test_invalid_overlap_rejected(self):
        with pytest.raises(ValueError):
            chunk_text("x" * 500, chunk_size=100, overlap=100)
        with pytest.raises(ValueError):
            chunk_text("x" * 500, chunk_size=100, overlap=-1)

    def test_empty_text_yields_nothing(self):
        assert chunk_text("", chunk_size=100, overlap=10) == []
        assert chunk_text("   \n ", chunk_size=100, overlap=10) == []

    def test_chunk_ids_are_stable(self):
        assert Chunk.make_id("doc_abc", 3) == "doc_abc::0003"


# ------------------------------------------------------------- extractors --
class TestExtractors:
    def test_text_extraction_normalises(self):
        doc = extract_bytes(b"Line one.\r\n\r\n\r\n\r\nLine two.\x01", "note.txt")
        assert "Line one." in doc.text
        assert "\x01" not in doc.text

    def test_html_strips_scripts(self):
        html = b"<html><head><title>T</title><script>alert(1)</script></head><body><p>Body text</p></body></html>"
        doc = extract_bytes(html, "page.html")
        assert "Body text" in doc.text
        assert "alert" not in doc.text
        assert doc.title == "T"

    def test_json_text_field_preferred(self):
        doc = extract_bytes(b'{"title": "Memo", "text": "Risk discussion"}', "m.json")
        assert doc.text == "Risk discussion"
        assert doc.title == "Memo"

    def test_csv_flattened(self):
        doc = extract_bytes(b"a,b\n1,2\n3,4\n", "t.csv")
        assert "a | b" in doc.text
        assert "1 | 2" in doc.text

    def test_unreadable_pdf_raises_extraction_error(self):
        with pytest.raises(ExtractionError):
            extract_bytes(b"%PDF-1.4 this is not a real pdf", "broken.pdf")


# ----------------------------------------------------------------- ingest --
class TestIngest:
    def test_ingest_text_file(self, ai_runtime):
        result = ai_runtime.ingestor.ingest_text(
            NVDA_RISK_TEXT,
            "NVDA_10-K_2025.txt",
            ticker="NVDA",
            publication_date="2025-02-26",
        )
        assert result.status == "indexed"
        assert result.chunks >= 1
        assert result.ticker == "NVDA"
        assert result.doc_type == "10-K"
        record = ai_runtime.registry.get(result.document_id)
        assert record is not None
        assert record.publication_date == "2025-02-26"
        assert record.chunk_count == result.chunks
        assert ai_runtime.store.count() == result.chunks

    def test_ingest_rejects_unsupported_extension(self, ai_runtime):
        with pytest.raises(UnsupportedFormatError):
            ai_runtime.ingestor.ingest_bytes(b"MZ payload", "malware.exe")

    def test_ingest_rejects_empty_payload(self, ai_runtime):
        with pytest.raises(InvalidFilenameError):
            ai_runtime.ingestor.ingest_bytes(b"", "empty.txt")

    def test_ingest_rejects_text_without_content(self, ai_runtime):
        with pytest.raises(EmptyDocumentError):
            ai_runtime.ingestor.ingest_text("   ", "blank.txt")

    def test_ingest_rejects_oversize_payload(self, ai_runtime):
        big = b"x" * (ai_runtime.cfg.max_upload_bytes + 1)
        with pytest.raises(DocumentTooLargeError):
            ai_runtime.ingestor.ingest_bytes(big, "big.txt")

    def test_duplicate_content_is_not_reindexed(self, ai_runtime):
        first = ai_runtime.ingestor.ingest_text(NVDA_RISK_TEXT, "nvda_risks.txt", ticker="NVDA")
        second = ai_runtime.ingestor.ingest_text(NVDA_RISK_TEXT, "nvda_risks_copy.txt", ticker="NVDA")
        assert second.status == "duplicate"
        assert second.document_id == first.document_id
        assert ai_runtime.store.count() == first.chunks

    def test_identical_reupload_is_noop(self, ai_runtime):
        first = ai_runtime.ingestor.ingest_text(NVDA_RISK_TEXT, "nvda_risks.txt", ticker="NVDA")
        again = ai_runtime.ingestor.ingest_text(NVDA_RISK_TEXT, "nvda_risks.txt", ticker="NVDA")
        assert again.status == "duplicate"
        assert "unchanged" in " ".join(again.warnings)
        assert ai_runtime.store.count() == first.chunks

    def test_update_replaces_chunks_in_sync(self, ai_runtime):
        v1 = ai_runtime.ingestor.ingest_text(NVDA_RISK_TEXT * 5, "doc.txt", ticker="NVDA")
        v2_text = "Completely different content about dividends and buybacks. " * 40
        v2 = ai_runtime.ingestor.ingest_text(v2_text, "doc.txt", ticker="NVDA")
        assert v2.status == "updated"
        assert v2.document_id == v1.document_id
        record = ai_runtime.registry.get(v1.document_id)
        assert record.chunk_count == v2.chunks
        assert ai_runtime.store.count() == v2.chunks  # stale chunks pruned
        assert record.content_hash == v2.content_hash

    def test_update_refused_without_replace_flag(self, ai_runtime):
        ai_runtime.ingestor.ingest_text(NVDA_RISK_TEXT, "doc.txt", ticker="NVDA")
        with pytest.raises(DocumentError):
            ai_runtime.ingestor.ingest_text(
                "different content entirely", "doc.txt", ticker="NVDA", replace=False
            )

    def test_delete_removes_index_registry_and_file(self, ai_runtime):
        result = ai_runtime.ingestor.ingest_text(NVDA_RISK_TEXT, "doc.txt", ticker="NVDA")
        record = ai_runtime.registry.get(result.document_id)
        stored = record.stored_path
        assert stored and ai_runtime.cfg.documents_dir in Path(stored).parents

        assert ai_runtime.ingestor.delete(result.document_id) is True
        assert ai_runtime.registry.get(result.document_id) is None
        assert ai_runtime.store.count() == 0
        assert not Path(stored).exists()

    def test_delete_unknown_document_is_false(self, ai_runtime):
        assert ai_runtime.ingestor.delete("doc_doesnotexist") is False

    def test_invalid_document_id_rejected(self, ai_runtime):
        from src.ai.registry import validate_document_id

        with pytest.raises(DocumentError):
            validate_document_id("../../etc/passwd")
        with pytest.raises(DocumentError):
            validate_document_id("")

    def test_traversal_filename_is_sanitised(self, ai_runtime):
        result = ai_runtime.ingestor.ingest_text("safe content", "../../evil.txt")
        assert result.filename == "evil.txt"

    def test_null_byte_filename_rejected(self, ai_runtime):
        with pytest.raises(InvalidFilenameError):
            ai_runtime.ingestor.ingest_text("content", "evil\x00.txt")

    def test_list_and_get_document(self, ai_runtime):
        result = ai_runtime.ingestor.ingest_text(
            NVDA_RISK_TEXT, "nvda.txt", ticker="NVDA", doc_type="10-K"
        )
        docs = ai_runtime.ingestor.list_documents(ticker="NVDA")
        assert [d["document_id"] for d in docs] == [result.document_id]
        assert ai_runtime.ingestor.list_documents(ticker="MSFT") == []
        fetched = ai_runtime.ingestor.get_document(result.document_id)
        assert fetched["doc_type"] == "10-K"
        assert "stored_path" not in fetched  # internal path never exposed

    def test_private_visibility_recorded(self, ai_runtime):
        result = ai_runtime.ingestor.ingest_text(
            "internal memo content", "memo.txt", visibility="private"
        )
        assert result.document["visibility"] == "private"
        with pytest.raises(DocumentError):
            ai_runtime.ingestor.ingest_text("x", "memo2.txt", visibility="everyone")

    def test_make_document_id_stable(self):
        a = make_document_id("report.txt", "NVDA")
        b = make_document_id("report.txt", "nvda")
        c = make_document_id("report.txt", "AAPL")
        assert a == b
        assert a != c


# -------------------------------------------------------------- embeddings --
class TestHashingEmbedder:
    def test_deterministic_and_normalised(self):
        emb = HashingEmbedder(dim=64)
        v1 = emb.encode(["risk factors and supply chain"])
        v2 = emb.encode(["risk factors and supply chain"])
        assert v1.shape == (1, 64)
        assert (v1 == v2).all()
        assert abs(float((v1**2).sum()) - 1.0) < 1e-4

    def test_similar_text_scores_higher(self):
        emb = HashingEmbedder(dim=128)
        query = emb.encode(["supply chain risk factors"])[0]
        docs = emb.encode(
            ["supply chain risk factors for NVIDIA", "recipe for chocolate cake"]
        )
        scores = docs @ query
        assert scores[0] > scores[1]

    def test_empty_input(self):
        emb = HashingEmbedder(dim=32)
        assert emb.encode([]).shape == (0, 32)

    def test_too_small_dim_rejected(self):
        from src.ai.errors import EmbeddingError

        with pytest.raises(EmbeddingError):
            HashingEmbedder(dim=8)
