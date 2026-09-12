"""Source-preserving annual-report ingestion and isolated, persistent retrieval."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

from .retrieval import (
    SemanticEmbedder,
    SemanticUnavailable,
    bm25_rank,
    reciprocal_rank_fusion,
    semantic_rank,
)

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_PAGES = 1000
MAX_TEXT_CHARS = 5_000_000
CHUNK_SIZE = 900
CHUNK_OVERLAP = 100


class AnnualReportError(ValueError):
    """An uploaded report or retrieval request cannot be safely processed."""


class AnnualReportStorageError(RuntimeError):
    """Report storage failed independently of the submitted document."""


def _validate_upload(
    content: bytes, filename: str, company: str, year: int
) -> tuple[str, str, int]:
    if not isinstance(content, bytes) or not content:
        raise AnnualReportError("上传文件为空。")
    if len(content) > MAX_FILE_BYTES:
        raise AnnualReportError("文件超过 20 MB 上限。")
    if not isinstance(filename, str) or not filename.strip() or len(filename) > 240:
        raise AnnualReportError("文件名为空或过长。")
    if (
        any(char in filename for char in ("/", "\\", "\x00", ":"))
        or PureWindowsPath(filename).is_absolute()
    ):
        raise AnnualReportError("文件名不能包含路径。")
    filename = filename.strip()
    if filename in {".", ".."} or re.search(r"[\x00-\x1f\x7f]", filename):
        raise AnnualReportError("文件名包含非法字符。")
    if Path(filename).suffix.lower() not in {".pdf", ".txt", ".md"}:
        raise AnnualReportError("仅支持 PDF、UTF-8 TXT 或 Markdown 文件。")
    if (
        not isinstance(company, str)
        or not company.strip()
        or len(company.strip()) > 120
    ):
        raise AnnualReportError("请提供长度在 1 至 120 字符内的公司名称。")
    company = company.strip()
    if re.search(r"[\x00-\x1f\x7f]", company):
        raise AnnualReportError("公司名称包含非法字符。")
    if isinstance(year, bool) or not isinstance(year, int) or not 1900 <= year <= 2100:
        raise AnnualReportError("报告年度必须是 1900 至 2100 之间的整数。")
    return filename, company, year


def _normalize_text(text: str) -> str:
    return "\n".join(
        line.rstrip()
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ).strip()


def _extract_pages(content: bytes, filename: str) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    if Path(filename).suffix.lower() == ".pdf":
        if not content.lstrip().startswith(b"%PDF-"):
            raise AnnualReportError("PDF 文件已损坏或不是有效的 PDF。")
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise AnnualReportError(
                "PDF 解析依赖 pypdf 未安装，请安装年报服务依赖。"
            ) from exc
        try:
            reader = PdfReader(io.BytesIO(content), strict=False)
            if reader.is_encrypted:
                raise AnnualReportError("暂不支持加密 PDF，请上传解密后的文件。")
            page_count = len(reader.pages)
            if not page_count or page_count > MAX_PAGES:
                raise AnnualReportError(f"PDF 页数必须在 1 至 {MAX_PAGES} 页之间。")
            pages: list[str] = []
            total_chars = 0
            for page in reader.pages:
                # Layout extraction keeps financial table rows/columns together.
                text = (
                    _normalize_text(page.extract_text(extraction_mode="layout") or "")
                    if page.get_contents() is not None
                    else ""
                )
                total_chars += len(text)
                if total_chars > MAX_TEXT_CHARS:
                    raise AnnualReportError("报告文本超过 500 万字符上限。")
                pages.append(text)
        except AnnualReportError:
            raise
        except Exception as exc:
            raise AnnualReportError(
                "PDF 文件已损坏或文本无法解析，请重新导出后上传。"
            ) from exc
        empty = [
            str(index + 1)
            for index, text in enumerate(pages)
            if not re.search(r"[\w\u3400-\u9fff]", text)
        ]
        if len(empty) == len(pages):
            raise AnnualReportError(
                "PDF 未提取到可检索文本，可能是扫描件；当前未启用 OCR，请上传含文字层的 PDF。"
            )
        if empty:
            warnings.append(
                f"第 {', '.join(empty[:20])}{' 等' if len(empty) > 20 else ''} 页未提取到文本，可能是扫描页；未执行 OCR。"
            )
        return pages, warnings

    try:
        text = content.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise AnnualReportError(
            "文本文件必须使用 UTF-8 编码，不能上传二进制文件。"
        ) from exc
    if text.lstrip().startswith("%PDF-") or re.search(
        r"[\x00-\x08\x0b\x0e-\x1f\x7f]", text
    ):
        raise AnnualReportError("检测到二进制或不支持的控制字符，请上传 UTF-8 文本。")
    if len(text) > MAX_TEXT_CHARS:
        raise AnnualReportError("报告文本超过 500 万字符上限。")
    pages = [_normalize_text(page) for page in text.split("\f")]
    if len(pages) > MAX_PAGES:
        raise AnnualReportError(f"文档超过 {MAX_PAGES} 页上限。")
    if not any(re.search(r"[\w\u3400-\u9fff]", page) for page in pages):
        raise AnnualReportError("文件没有可检索的文本。")
    warnings.append(
        "TXT/Markdown 页码按换页符划分；无换页符时为第 1 页，不能代表原 PDF 页码。"
    )
    return pages, warnings


def _is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 85 or "\t" in line or "|" in line:
        return False
    return bool(
        re.match(r"^#{1,6}\s+\S", line)
        or re.match(r"^第[一二三四五六七八九十百零〇\d]+[章节篇部]\s*\S", line)
        or re.match(r"^[一二三四五六七八九十]+[、．.]\s*\S", line)
        or re.match(r"^[（(][一二三四五六七八九十]+[）)]\s*\S", line)
        # Financial-note headings such as '(61) 营业收入和营业成本' must
        # terminate the previous table's column context. Exclude numeric rows.
        or (
            re.match(r"^[（(]\s*\d{1,3}\s*[）)]\s*[^\d\s]", line)
            and not re.search(r"\d", re.sub(r"^[（(]\s*\d{1,3}\s*[）)]", "", line))
        )
        or line
        in {
            "管理层讨论与分析",
            "财务报告",
            "财务报表",
            "合并利润表",
            "合并现金流量表",
            "合并资产负债表",
            "主要会计数据和财务指标",
            "经营情况讨论与分析",
        }
    )


def _split_oversized(text: str, max_size: int = CHUNK_SIZE) -> list[str]:
    """Prefer row/sentence boundaries; bound exceptionally long individual rows."""
    if len(text) <= max_size:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_size, len(text))
        if end < len(text):
            boundaries = [
                text.rfind(char, start + max_size // 2, end) for char in "\n。；;！!?"
            ]
            boundary = max(boundaries)
            if boundary >= 0:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end == len(text):
            break
        start = max(start + 1, end - CHUNK_OVERLAP)
    return pieces


def _year_table_header(line: str) -> str | None:
    """Only repeat a clear column header, never infer year columns from prose."""
    years = set(re.findall(r"((?:19|20)\d{2})\s*年", line))
    stripped = line.strip()
    if (
        2 <= len(years) <= 8
        and len(stripped) <= 300
        and re.search(r"\s{2,}|\t|\|", stripped)
    ):
        return stripped
    return None


def _chunk_pages(pages: list[str], document_id: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    section = "正文"
    for page_number, page_text in enumerate(pages, 1):
        pending: list[str] = []
        table_header: str | None = None

        def flush(
            current_section: str, header: str | None, source_page: int = page_number
        ) -> None:
            nonlocal pending
            text = "\n".join(pending).strip()
            if text:
                reserve = len(header) + 1 if header else 0
                for piece in _split_oversized(text, CHUNK_SIZE - reserve):
                    if header and header not in piece:
                        piece = header + "\n" + piece
                    index = len(chunks)
                    chunks.append(
                        {
                            "id": f"{document_id}:p{source_page}:c{index}",
                            "document_id": document_id,
                            "page": source_page,
                            "section": current_section,
                            "text": piece,
                            "ordinal": index,
                        }
                    )
            pending = []

        for line in page_text.split("\n"):
            if _is_heading(line):
                flush(section, table_header)
                section = re.sub(r"^#{1,6}\s+", "", line.strip())
                table_header = None
                pending.append(line.strip())
            elif new_header := _year_table_header(line):
                flush(section, table_header)
                table_header = new_header
                pending.append(new_header)
            elif not line.strip():
                # Paragraph boundaries separate evidence without losing table rows.
                if sum(map(len, pending)) >= CHUNK_SIZE // 2:
                    flush(section, table_header)
                elif pending:
                    pending.append("")
            else:
                if (
                    sum(len(part) + 1 for part in pending) + len(line) > CHUNK_SIZE
                    and pending
                ):
                    previous = "\n".join(pending)
                    flush(section, table_header)
                    # Preserve whole trailing lines where possible (table-safe overlap).
                    tail_lines: list[str] = []
                    tail_length = 0
                    for tail in reversed(previous.split("\n")):
                        if tail_length + len(tail) + 1 > CHUNK_OVERLAP:
                            break
                        tail_lines.insert(0, tail)
                        tail_length += len(tail) + 1
                    pending.extend(tail_lines)
                pending.append(line)
        flush(section, table_header)
    return chunks


def _same_page_table_context(page_text: str, chunk_text: str) -> str:
    """Recover only literal same-page context lost at an existing chunk boundary.

    This does not rewrite persisted chunks or embeddings. The anchor is an exact
    source line; a table header and unit must precede it on the same physical page.
    Ambiguous adjustment/subcolumn headers are retained so parsers can refuse them.
    """
    lines = [line.strip() for line in page_text.splitlines()]
    anchors = [line.strip() for line in chunk_text.splitlines()
               if len(line.strip()) >= 12 and re.search(r"[\u4e00-\u9fff]", line)
               and re.search(r"\d{1,3}(?:[,，]\d{3})+|\d+\.\d+", line)
               and not _year_table_header(line)]
    indexes = [lines.index(line) for line in anchors if line in lines]
    if not indexes:
        return ""
    first = min(indexes)
    before = lines[:first]
    header_index = next((index for index in range(len(before) - 1, -1, -1)
                         if _year_table_header(before[index])
                         or re.search(r"本期数\s+上年同期数", before[index])), None)
    if header_index is None:
        return ""
    # An intervening statement/note heading terminates the previous table.
    if any(_is_heading(line) for line in before[header_index + 1:]):
        return ""
    context = []
    title = next((line for line in lines[:5] if re.search(r"(?:19|20)\d{2}\s*年\s*年度报告", line)), None)
    if title:
        context.append(title)
    preceding = before[:header_index]
    statement = next((line for line in reversed(preceding)
                      if re.search(r"(?:合并|母公司)(?:利润表|现金流量表|资产负债表)", line)), None)
    if statement:
        context.append(statement)
    # Explicit unit lines only; never infer a unit from unrelated metric values.
    units = [(index, line) for index, line in enumerate(preceding)
             if re.search(r"单位\s*[:：]\s*(?:人民币)?\s*(?:百万元|亿元|万元|千元|元)", line)]
    if units:
        index, unit = units[-1]
        if not any(_is_heading(line) for line in preceding[index + 1:]):
            context.append(unit)
    context.append(before[header_index])
    # Header continuations carry percent units or adjustment labels. Do not copy
    # unrelated financial values into another evidence chunk.
    for line in before[header_index + 1:]:
        if not line:
            continue
        if re.search(r"\d", line) or len(line) > 200:
            break
        context.append(line)
        if len(context) >= 8:
            break
    return "\n".join(dict.fromkeys(context))


class AnnualReportStore:
    """SQLite-backed report library with explicit document-scoped retrieval.

    Files are parsed from memory, never written using the supplied filename.
    Document IDs are content+company+year hashes, so duplicates are idempotent
    while an identical report assigned to another company/year remains isolated.
    """

    def __init__(
        self, data_dir: str | Path | None = None, *, embedder: Any = None
    ) -> None:
        self.data_dir = Path(
            data_dir
            or os.getenv("ANNUAL_REPORT_DATA_DIR")
            or Path(__file__).resolve().parents[2] / "data" / "annual_reports"
        )
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "annual_reports.sqlite3"
        self._embedder = SemanticEmbedder(embedder)
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, filename TEXT NOT NULL, company TEXT NOT NULL,
                    year INTEGER NOT NULL, page_count INTEGER NOT NULL, chunk_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL, warnings TEXT NOT NULL, content_sha256 TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pages (
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    page INTEGER NOT NULL, text TEXT NOT NULL, PRIMARY KEY(document_id, page)
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    page INTEGER NOT NULL, section TEXT NOT NULL, text TEXT NOT NULL, ordinal INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id, ordinal);
                CREATE INDEX IF NOT EXISTS documents_scope ON documents(company, year);
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    model TEXT NOT NULL, vector TEXT NOT NULL, PRIMARY KEY(chunk_id, model)
                );
            """)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        try:
            connection = sqlite3.connect(self.db_path, timeout=30)
        except sqlite3.Error as exc:
            raise AnnualReportStorageError(
                "年报数据库无法打开，请检查数据目录与访问权限。"
            ) from exc
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                yield connection
        except sqlite3.Error as exc:
            raise AnnualReportStorageError(
                "年报数据库读写失败，请检查存储空间后重试。"
            ) from exc
        finally:
            connection.close()

    @staticmethod
    def _document(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        result["warnings"] = json.loads(result["warnings"])
        result.pop("content_sha256", None)
        return result

    def ingest_bytes(
        self, content: bytes, filename: str, company: str, year: int
    ) -> dict[str, Any]:
        filename, company, year = _validate_upload(content, filename, company, year)
        content_hash = hashlib.sha256(content).hexdigest()
        identity = json.dumps(
            [content_hash, company, year], ensure_ascii=False, separators=(",", ":")
        )
        document_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
        existing = self.get_document(document_id)
        if existing:
            return {**existing, "deduplicated": True}
        pages, warnings = _extract_pages(content, filename)
        chunks = _chunk_pages(pages, document_id)
        if not chunks:
            raise AnnualReportError("报告没有可检索的段落。")
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            # Two simultaneous uploads are serialized before checking/inserting.
            connection.execute("BEGIN IMMEDIATE")
            existing_row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if existing_row:
                return {**self._document(existing_row), "deduplicated": True}
            connection.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    document_id,
                    filename,
                    company,
                    year,
                    len(pages),
                    len(chunks),
                    created_at,
                    json.dumps(warnings, ensure_ascii=False),
                    content_hash,
                ),
            )
            connection.executemany(
                "INSERT INTO pages VALUES (?, ?, ?)",
                [(document_id, index, text) for index, text in enumerate(pages, 1)],
            )
            connection.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        chunk["id"],
                        document_id,
                        chunk["page"],
                        chunk["section"],
                        chunk["text"],
                        chunk["ordinal"],
                    )
                    for chunk in chunks
                ],
            )
        return {
            "id": document_id,
            "filename": filename,
            "company": company,
            "year": year,
            "page_count": len(pages),
            "chunk_count": len(chunks),
            "created_at": created_at,
            "warnings": warnings,
            "deduplicated": False,
        }

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY created_at DESC, id"
            ).fetchall()
        return [self._document(row) for row in rows]

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            return self._document(
                connection.execute(
                    "SELECT * FROM documents WHERE id = ?", (document_id,)
                ).fetchone()
            )

    def get_page(self, document_id: str, page: int) -> dict[str, Any] | None:
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            return None
        with self._connect() as connection:
            row = connection.execute(
                """SELECT p.document_id, p.page, p.text, d.filename, d.company, d.year
                                      FROM pages p JOIN documents d ON d.id = p.document_id
                                      WHERE p.document_id = ? AND p.page = ?""",
                (document_id, page),
            ).fetchone()
        return dict(row) if row else None

    def retrieval_status(self) -> dict[str, Any]:
        return self._embedder.status

    def _semantic_ranking(
        self, query: str, chunks: list[dict[str, Any]]
    ) -> list[tuple[str, float]]:
        query_vector = self._embedder.encode([query], query=True)[0]
        model_key = self._embedder.cache_key
        chunk_ids = [chunk["id"] for chunk in chunks]
        vectors: dict[str, list[float]] = {}
        with self._connect() as connection:
            for start in range(0, len(chunk_ids), 500):
                batch = chunk_ids[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"SELECT chunk_id, vector FROM embeddings WHERE model = ? AND chunk_id IN ({placeholders})",
                    [model_key, *batch],
                ).fetchall()
                for row in rows:
                    try:
                        vector = json.loads(row["vector"])
                        if isinstance(vector, list) and len(vector) == len(
                            query_vector
                        ):
                            vector = [float(value) for value in vector]
                            norm = math.sqrt(sum(value * value for value in vector))
                            if (
                                math.isfinite(norm)
                                and norm > 0
                                and all(math.isfinite(value) for value in vector)
                            ):
                                vectors[row["chunk_id"]] = [
                                    value / norm for value in vector
                                ]
                    except (ValueError, TypeError):
                        pass  # Recompute corrupt cache rows from the original source.
        missing = [chunk for chunk in chunks if chunk["id"] not in vectors]
        for start in range(0, len(missing), 64):
            batch = missing[start : start + 64]
            encoded = self._embedder.encode(
                [f"{chunk['section']}\n{chunk['text']}" for chunk in batch]
            )
            with self._connect() as connection:
                for chunk, vector in zip(batch, encoded):
                    vectors[chunk["id"]] = vector
                    connection.execute(
                        "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)",
                        (chunk["id"], model_key, json.dumps(vector, allow_nan=False)),
                    )
        return semantic_rank(query_vector, vectors)

    def search(
        self,
        query: str,
        document_ids: list[str],
        years: list[int] | None = None,
        top_k: int = 8,
        mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        if mode not in {"hybrid", "bm25", "semantic", "vector"}:
            raise AnnualReportError("检索模式必须是 hybrid、bm25 或 semantic。")
        if not isinstance(query, str) or not query.strip() or not document_ids:
            return []
        if len(query) > 4000:
            raise AnnualReportError("检索问题不能超过 4000 字符。")
        if not isinstance(document_ids, (list, tuple)) or any(
            not isinstance(item, str) for item in document_ids
        ):
            raise AnnualReportError("请提供明确的文档 ID 列表。")
        document_ids = list(dict.fromkeys(document_ids))
        if len(document_ids) > 100:
            raise AnnualReportError("一次最多检索 100 份报告。")
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= 50
        ):
            raise AnnualReportError("top_k 必须在 1 至 50 之间。")
        if years is not None:
            if not isinstance(years, (list, tuple)) or any(
                isinstance(year, bool) or not isinstance(year, int) for year in years
            ):
                raise AnnualReportError("年度过滤必须是整数列表。")
            years = list(dict.fromkeys(years))
            if not years:
                return []
            if len(years) > 100:
                raise AnnualReportError("年度过滤条件过多。")
        placeholders = ",".join("?" for _ in document_ids)
        sql = f"""SELECT c.*, d.filename, d.company, d.year FROM chunks c JOIN documents d ON d.id = c.document_id
                  WHERE c.document_id IN ({placeholders})"""
        parameters: list[Any] = list(document_ids)
        if years is not None:
            sql += " AND d.year IN (" + ",".join("?" for _ in years) + ")"
            parameters.extend(years)
        sql += " ORDER BY c.document_id, c.ordinal"
        with self._connect() as connection:
            chunks = [
                dict(row) for row in connection.execute(sql, parameters).fetchall()
            ]
        if not chunks:
            return []
        lexical = bm25_rank(query, chunks)
        warning: str | None = None
        if mode == "bm25":
            ranked, actual_mode = lexical, "bm25"
        else:
            try:
                semantic = self._semantic_ranking(query, chunks)
                if mode == "hybrid":
                    ranked, actual_mode = (
                        reciprocal_rank_fusion(lexical, semantic),
                        "hybrid",
                    )
                else:
                    ranked, actual_mode = semantic, "semantic"
            except SemanticUnavailable as exc:
                ranked, actual_mode = lexical, "bm25_fallback"
                warning = str(exc)
        by_id = {chunk["id"]: chunk for chunk in chunks}
        selected_pages = {(by_id[chunk_id]["document_id"], by_id[chunk_id]["page"])
                          for chunk_id, _score in ranked[:top_k]}
        page_texts = {}
        with self._connect() as connection:
            for document_id, page in selected_pages:
                row = connection.execute("SELECT text FROM pages WHERE document_id = ? AND page = ?",
                                         (document_id, page)).fetchone()
                if row:
                    page_texts[(document_id, page)] = row["text"]
        hits = []
        for chunk_id, score in ranked[:top_k]:
            hit = {
                **by_id[chunk_id],
                "score": round(score, 8),
                "retrieval_mode": actual_mode,
            }
            hit.pop("ordinal", None)
            context = _same_page_table_context(page_texts.get((hit["document_id"], hit["page"]), ""), hit["text"])
            if context:
                hit["source_context"] = context
            if warning:
                hit["retrieval_warning"] = warning
            hits.append(hit)
        return hits
