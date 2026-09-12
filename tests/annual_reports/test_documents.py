from __future__ import annotations

import io
from itertools import pairwise

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from backend.annual_reports import documents
from backend.annual_reports.documents import (
    CHUNK_SIZE,
    AnnualReportError,
    AnnualReportStore,
)


def make_pdf(*pages: str) -> bytes:
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=595, height=842)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): writer._add_object(font)}
                )
            }
        )
        if text:
            stream = DecodedStreamObject()
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream.set_data(f"BT /F1 12 Tf 50 750 Td ({escaped}) Tj ET".encode("ascii"))
            page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_text_is_persistent_idempotent_and_scoped(tmp_path):
    content = "# 经营情况\n营业收入为 100 亿元。\f# 现金流量表\n经营活动现金流为 30 亿元。".encode()
    store = AnnualReportStore(tmp_path)
    original = store.ingest_bytes(content, "2024年报.md", "示例制造", 2024)
    duplicate = store.ingest_bytes(content, "renamed.md", "示例制造", 2024)
    next_year = store.ingest_bytes(content, "2025年报.md", "示例制造", 2025)
    other_company = store.ingest_bytes(content, "其他公司.md", "其他制造", 2024)
    assert duplicate["id"] == original["id"]
    assert duplicate["deduplicated"] is True
    assert duplicate["filename"] == "2024年报.md"
    assert len({original["id"], next_year["id"], other_company["id"]}) == 3
    restarted = AnnualReportStore(tmp_path)
    assert len(restarted.list_documents()) == 3
    restored = restarted.get_document(original["id"])
    assert restored["page_count"] == 2
    assert restored["created_at"] == original["created_at"]
    assert "不能代表原 PDF 页码" in restored["warnings"][0]
    page = restarted.get_page(original["id"], 2)
    assert page["company"] == "示例制造" and page["year"] == 2024
    assert "经营活动现金流为 30 亿元" in page["text"]
    assert restarted.get_page(original["id"], 3) is None
    assert restarted.get_page(original["id"], 0) is None
    assert restarted.get_document("missing") is None


def test_real_pdf_keeps_physical_pages(tmp_path):
    store = AnnualReportStore(tmp_path)
    document = store.ingest_bytes(
        make_pdf("Revenue 100 million", "Operating cash 30 million"),
        "report.pdf",
        "Example",
        2024,
    )
    assert document["page_count"] == 2
    assert document["warnings"] == []
    hit = store.search("Operating cash", [document["id"]], mode="bm25")[0]
    assert hit["page"] == 2
    assert hit["text"] in store.get_page(document["id"], 2)["text"]
    assert "Revenue" not in store.get_page(document["id"], 2)["text"]


def test_chinese_pdf_text_is_actually_extracted(tmp_path):
    pytest.importorskip("reportlab")
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfgen import canvas

    output = io.BytesIO()
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    pdf = canvas.Canvas(output)
    pdf.setFont("STSong-Light", 12)
    pdf.drawString(50, 750, "营业收入为一百亿元，经营现金流下降。")
    pdf.showPage()
    pdf.setFont("STSong-Light", 12)
    pdf.drawString(50, 750, "应收账款增加导致回款放缓。")
    pdf.save()
    store = AnnualReportStore(tmp_path)
    document = store.ingest_bytes(output.getvalue(), "中文年报.pdf", "中文示例", 2024)
    assert "营业收入" in store.get_page(document["id"], 1)["text"]
    assert store.search("应收账款", [document["id"]], mode="bm25")[0]["page"] == 2


def test_headings_and_financial_table_rows_preserved(tmp_path):
    table = "# 合并现金流量表\n\n| 项目 | 2024年 | 2023年 |\n| 经营活动现金流量净额 | 35.25 | 48.61 |\n| 应收账款 | 150.00 | 90.00 |"
    store = AnnualReportStore(tmp_path)
    document = store.ingest_bytes(table.encode(), "报表.md", "制造公司", 2024)
    hit = store.search("现金流量净额", [document["id"]], mode="bm25")[0]
    assert hit["section"] == "合并现金流量表"
    assert "| 经营活动现金流量净额 | 35.25 | 48.61 |" in hit["text"]
    assert hit["page"] == 1


def test_long_paragraphs_are_bounded_with_overlap(tmp_path):
    text = "# 经营情况\n" + "收入增长来自新产品交付，现金回款主要受客户账期影响。" * 100
    store = AnnualReportStore(tmp_path)
    document = store.ingest_bytes(text.encode(), "long.txt", "制造公司", 2024)
    hits = store.search("现金回款", [document["id"]], mode="bm25", top_k=50)
    assert document["chunk_count"] > 2
    assert all(len(hit["text"]) <= CHUNK_SIZE for hit in hits)
    assert all(hit["page"] == 1 and hit["section"] == "经营情况" for hit in hits)
    chunks = documents._chunk_pages([text], "id")
    body = [chunk["text"] for chunk in chunks if len(chunk["text"]) > 100]
    assert any(left[-100:] == right[:100] for left, right in pairwise(body))


@pytest.mark.parametrize(
    "filename",
    [
        "../report.txt",
        "..\\report.txt",
        "C:\\report.txt",
        "/report.txt",
        "report.txt:evil",
        "report\x00.txt",
    ],
)
def test_path_filenames_are_rejected(tmp_path, filename):
    with pytest.raises(AnnualReportError, match="文件名"):
        AnnualReportStore(tmp_path).ingest_bytes(b"content", filename, "Company", 2024)
    assert not (tmp_path.parent / "report.txt").exists()


@pytest.mark.parametrize(
    ("content", "filename", "message"),
    [
        (b"", "empty.txt", "为空"),
        (b"\x00\x01binary", "binary.txt", "二进制"),
        (b"\xff\xfe\x01", "nonutf8.txt", "UTF-8"),
        (b"not a pdf", "invalid.pdf", "损坏"),
        (b"%PDF-1.7\ncorrupt", "corrupt.pdf", "损坏"),
        (b"%PDF-1.7\ncontent", "pretend.txt", "二进制"),
        (b"hello", "script.exe", "仅支持"),
        (b"\n\t  ", "blank.txt", "没有可检索"),
    ],
)
def test_invalid_uploads_do_not_leave_documents(tmp_path, content, filename, message):
    store = AnnualReportStore(tmp_path)
    with pytest.raises(AnnualReportError, match=message):
        store.ingest_bytes(content, filename, "Company", 2024)
    assert store.list_documents() == []


def test_scanned_pdf_is_not_faked_as_ocr(tmp_path):
    store = AnnualReportStore(tmp_path)
    with pytest.raises(AnnualReportError, match="扫描件.*未启用 OCR"):
        store.ingest_bytes(make_pdf("", ""), "scanned.pdf", "Company", 2024)
    assert store.list_documents() == []


def test_partial_empty_pdf_warns_without_shifting_page_numbers(tmp_path):
    store = AnnualReportStore(tmp_path)
    doc = store.ingest_bytes(
        make_pdf("", "Cash flow 50 million"), "mixed.pdf", "Company", 2024
    )
    assert "第 1 页" in doc["warnings"][0]
    assert store.search("Cash flow", [doc["id"]], mode="bm25")[0]["page"] == 2


def test_encrypted_pdf_requires_decryption(tmp_path):
    writer = PdfWriter()
    writer.add_blank_page(595, 842)
    writer.encrypt("secret")
    output = io.BytesIO()
    writer.write(output)
    with pytest.raises(AnnualReportError, match="加密"):
        AnnualReportStore(tmp_path).ingest_bytes(
            output.getvalue(), "encrypted.pdf", "Company", 2024
        )


def test_size_and_page_limits(tmp_path, monkeypatch):
    store = AnnualReportStore(tmp_path)
    monkeypatch.setattr(documents, "MAX_FILE_BYTES", 5)
    with pytest.raises(AnnualReportError, match="20 MB"):
        store.ingest_bytes(b"123456", "large.txt", "Company", 2024)
    monkeypatch.setattr(documents, "MAX_FILE_BYTES", 20 * 1024 * 1024)
    monkeypatch.setattr(documents, "MAX_PAGES", 1)
    with pytest.raises(AnnualReportError, match="页数"):
        store.ingest_bytes(make_pdf("first", "second"), "many.pdf", "Company", 2024)
    with pytest.raises(AnnualReportError, match="超过 1 页"):
        store.ingest_bytes(b"first\fsecond", "many.txt", "Company", 2024)


@pytest.mark.parametrize(
    ("company", "year"),
    [("", 2024), ("A\x00B", 2024), ("A", True), ("A", "2024"), ("A", 1800)],
)
def test_invalid_metadata_rejected(tmp_path, company, year):
    with pytest.raises(AnnualReportError):
        AnnualReportStore(tmp_path).ingest_bytes(b"report", "r.txt", company, year)


def test_simultaneous_duplicate_uploads_are_atomic(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    store = AnnualReportStore(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda _: store.ingest_bytes(
                    "经营现金流为 100 亿元。".encode(), "annual.txt", "甲公司", 2024
                ),
                range(4),
            )
        )
    assert len({item["id"] for item in results}) == 1
    assert sum(not item["deduplicated"] for item in results) == 1
    assert len(store.list_documents()) == 1
    assert len(store.search("现金流", [results[0]["id"]], mode="bm25")) == 1


def test_split_financial_tables_repeat_their_own_year_header(tmp_path):
    header = "2024 年    2023 年    本年比上年增减    2022 年"
    balance_header = "2024 年末    2023 年末    本年末比上年末增减"
    rows = [
        f"财务明细项目{index}（千元）"
        + " " * 35
        + "407,149,600    372,037,280    9.44%    343,917,531"
        for index in range(12)
    ]
    text = (
        "# 财务指标\n"
        + header
        + "\n"
        + "\n".join(rows)
        + "\n经营活动产生的现金流量净额（千元）    60,511,572    57,902,611    4.51%    34,657,828\n"
        + balance_header
        + "\n资产总额（千元）    10,000    9,000    11.11%\f# 其他事项\n董事会完成讨论。"
    )
    store = AnnualReportStore(tmp_path)
    document = store.ingest_bytes(text.encode(), "年度财务.txt", "制造公司", 2024)
    cash = store.search("经营活动产生的现金流量净额", [document["id"]], mode="bm25")[0]
    assert header in cash["text"]
    assert balance_header not in cash["text"]
    assert "60,511,572" in cash["text"] and cash["page"] == 1
    asset = store.search("资产总额", [document["id"]], mode="bm25")[0]
    assert balance_header in asset["text"]
    other = store.search("董事会", [document["id"]], mode="bm25")[0]
    assert "2024" not in other["text"] and other["page"] == 2
    assert all(
        len(chunk["text"]) <= CHUNK_SIZE
        for chunk in documents._chunk_pages([text], "id")
    )
