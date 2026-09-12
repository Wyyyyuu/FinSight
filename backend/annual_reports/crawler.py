"""Small, rate-limited CNINFO annual-report collector with auditable provenance.

Uses the site's public announcement search, then validates the exact security,
fiscal year, full-report title, PDF header and cover before importing anything.
No login, browser cookies, challenge bypass or external generation is involved.
"""
from __future__ import annotations

import hashlib
import html
import io
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlsplit

import httpx
from pypdf import PdfReader

QUERY_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
PDF_ORIGIN = "https://static.cninfo.com.cn/"
MAX_PDF_BYTES = 20 * 1024 * 1024
USER_AGENT = "FinSight-AnnualReports/1.0 (public annual-report research)"


class CollectionError(ValueError):
    """A source could not be discovered, downloaded or verified."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_title(value: str) -> str:
    return re.sub(r"\s+", "", html.unescape(re.sub(r"<[^>]*>", "", value)))


def is_full_report(title: str, year: int) -> bool:
    title = clean_title(title)
    if re.search(r"摘要|英文|英语|取消|撤销|关于|提示|说明|审计|ESG|社会责任", title, re.IGNORECASE):
        return False
    # Full revised reports are preferred over stale originals, but a correction
    # announcement is not itself a replacement financial report.
    return bool(re.search(
        rf"(?<!\d){year}年年度报告(?:全文)?(?:[（(](?:修订版|修订后|更新版|更新后|更正后|更正版)[）)])?$",
        title,
    ))


def validate_security(code: str, year: int) -> None:
    if not re.fullmatch(r"[036]\d{5}", code):
        raise CollectionError("股票代码须为沪深 A 股六位代码（0、3 或 6 开头）。")
    if isinstance(year, bool) or not isinstance(year, int) or not 2000 <= year < datetime.now(timezone.utc).astimezone().year:
        raise CollectionError("报告年度须在 2000 年至上一年之间。")


def pdf_url(adjunct_path: str) -> str:
    if not re.fullmatch(r"finalpage/\d{4}-\d{2}-\d{2}/\d+\.[pP][dD][fF]", adjunct_path):
        raise CollectionError("公告附件地址不是可验证的巨潮 PDF 路径。")
    return PDF_ORIGIN + adjunct_path


@dataclass(frozen=True)
class Announcement:
    stock_code: str
    company: str
    report_year: int
    title: str
    announcement_id: str
    published_at: str
    pdf_url: str
    query_url: str = QUERY_URL


def select_announcement(rows: list[dict[str, Any]], code: str, year: int) -> Announcement:
    candidates = []
    for row in rows:
        title = str(row.get("announcementTitle") or "")
        if str(row.get("secCode")) != code or not is_full_report(title, year):
            continue
        if str(row.get("adjunctType", "PDF")).upper() != "PDF":
            continue
        try:
            url = pdf_url(str(row.get("adjunctUrl") or ""))
            stamp = int(row["announcementTime"])
            published = datetime.fromtimestamp(stamp / 1000, timezone.utc).isoformat()
            announcement_id = str(row["announcementId"])
            company = clean_title(str(row.get("secName") or ""))
            if not company or not announcement_id.isdigit():
                continue
        except (KeyError, ValueError, OverflowError, OSError):
            continue
        candidates.append((stamp, announcement_id, Announcement(
            code, company, year, clean_title(title), announcement_id, published, url,
        )))
    if not candidates:
        raise CollectionError(f"未找到 {code} 的 {year} 年完整中文年报；摘要、英文版及非对应公司公告均已过滤。")
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def verify_pdf(content: bytes, report: Announcement) -> dict[str, Any]:
    if not content or len(content) > MAX_PDF_BYTES or not content.lstrip().startswith(b"%PDF-"):
        raise CollectionError("下载内容不是有效 PDF，或超过 20 MB 上限。")
    try:
        reader = PdfReader(io.BytesIO(content), strict=False)
        if reader.is_encrypted or not 20 <= len(reader.pages) <= 1000:
            raise CollectionError("年报须为未加密的 20–1000 页完整 PDF。")
        cover = clean_title("\n".join(page.extract_text() or "" for page in reader.pages[:5]))
    except CollectionError:
        raise
    except Exception as exc:
        raise CollectionError("下载的 PDF 无法解析。") from exc
    company = report.company.replace("*ST", "").removeprefix("ST")
    if report.stock_code not in cover and company not in cover:
        raise CollectionError("PDF 前五页未能核实对应公司或股票代码。")
    if not re.search(rf"{report.report_year}年?(?:年度报告|年报)", cover):
        raise CollectionError("PDF 前五页未能核实所请求的年度报告。")
    return {"page_count": len(reader.pages), "byte_count": len(content),
            "content_sha256": hashlib.sha256(content).hexdigest()}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class CninfoCollector:
    def __init__(self, client: httpx.Client | None = None, *, interval: float = 1.0,
                 max_pages: int = 5) -> None:
        self.client = client or httpx.Client(
            timeout=httpx.Timeout(60, connect=15), follow_redirects=False,
            headers={"User-Agent": USER_AGENT, "Referer": "https://www.cninfo.com.cn/"},
        )
        self._owns_client = client is None
        self.interval = max(1.0, interval)
        self.max_pages = max(1, min(max_pages, 10))
        self._last_request = 0.0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        if self._owns_client:
            self.client.close()

    def _pace(self) -> None:
        delay = self.interval - (time.monotonic() - self._last_request)
        if delay > 0:
            time.sleep(delay)
        self._last_request = time.monotonic()

    def discover(self, code: str, year: int) -> Announcement:
        validate_security(code, year)
        rows: list[dict[str, Any]] = []
        for page in range(1, self.max_pages + 1):
            data = {"pageNum": str(page), "pageSize": "30",
                    "column": "sse" if code.startswith("6") else "szse", "tabName": "fulltext",
                    "searchkey": f"{code} {year}年年度报告", "isHLtitle": "false",
                    "seDate": f"{year + 1}-01-01~{datetime.now(timezone.utc).astimezone().date().isoformat()}"}
            response = self._query(data)
            try:
                payload = response.json()
                batch = payload.get("announcements") or []
                if not isinstance(batch, list) or any(not isinstance(row, dict) for row in batch):
                    raise ValueError("invalid announcements")
            except (ValueError, AttributeError) as exc:
                raise CollectionError("巨潮公告查询未返回预期 JSON，可能正在维护。") from exc
            rows.extend(batch)
            if not payload.get("hasMore"):
                return select_announcement(rows, code, year)
        raise CollectionError("公告页数达到上限，未假定不完整搜索中的报告为最新版本。")

    def _query(self, data: dict[str, str]) -> httpx.Response:
        for attempt in range(3):
            self._pace()
            try:
                response = self.client.post(QUERY_URL, data=data, follow_redirects=False)
                if response.status_code in (401, 403, 429):
                    raise CollectionError(f"公告网站拒绝或限制访问（{response.status_code}），已停止请求。")
                if response.status_code >= 500 and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                response.raise_for_status()
                return response
            except httpx.TransportError as exc:
                if attempt == 2:
                    raise CollectionError("公告查询网络连接失败，请稍后重试。") from exc
                time.sleep(2 ** attempt)
            except httpx.HTTPStatusError as exc:
                raise CollectionError(f"公告查询失败（HTTP {exc.response.status_code}）。") from exc
        raise CollectionError("公告查询失败。")  # defensive, all attempts return or raise

    def download(self, report: Announcement, directory: Path) -> dict[str, Any]:
        # Revalidate even dataclasses supplied by callers; no arbitrary URL fetch.
        parsed = urlsplit(report.pdf_url)
        if parsed.scheme != "https" or parsed.netloc != "static.cninfo.com.cn":
            raise CollectionError("只允许下载巨潮官方 HTTPS 附件。")
        if pdf_url(parsed.path.lstrip("/")) != report.pdf_url:
            raise CollectionError("附件 URL 包含非预期参数。")
        validate_security(report.stock_code, report.report_year)
        if not report.announcement_id.isdigit():
            raise CollectionError("公告编号无效。")
        path = directory / report.stock_code / str(report.report_year) / f"{report.announcement_id}.pdf"
        cached = path.is_file()
        fetched_at = None
        sidecar = path.with_suffix(".source.json")
        if cached:
            if path.stat().st_size > MAX_PDF_BYTES:
                raise CollectionError("本地 PDF 缓存超过大小上限。")
            content = path.read_bytes()
            if sidecar.is_file():
                try:
                    previous = json.loads(sidecar.read_text(encoding="utf-8"))
                    if previous.get("content_sha256") == hashlib.sha256(content).hexdigest():
                        fetched_at = previous.get("fetched_at")
                except (OSError, ValueError, AttributeError):
                    pass
        else:
            content = self._download_bytes(report.pdf_url)
            fetched_at = utc_now()
        verified = verify_pdf(content, report)
        if not cached:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".pdf.tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
        result = {**asdict(report), **verified, "local_path": str(path.resolve()),
                  "fetched_at": fetched_at, "verified_at": utc_now(), "download_cached": cached}
        write_json(sidecar, result)
        return result

    def _download_bytes(self, url: str) -> bytes:
        for attempt in range(3):
            self._pace()
            try:
                with self.client.stream("GET", url, follow_redirects=False) as response:
                    if response.status_code in (401, 403, 429):
                        raise CollectionError(f"附件网站拒绝或限制访问（{response.status_code}），已停止请求。")
                    if response.status_code >= 500 and attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                    # Redirects are deliberately not followed to other domains.
                    response.raise_for_status()
                    length = response.headers.get("content-length", "")
                    if length.isdigit() and int(length) > MAX_PDF_BYTES:
                        raise CollectionError("公告附件超过 20 MB 上限。")
                    content = bytearray()
                    for chunk in response.iter_bytes(64 * 1024):
                        content.extend(chunk)
                        if len(content) > MAX_PDF_BYTES:
                            raise CollectionError("公告附件超过 20 MB 上限。")
                    return bytes(content)
            except httpx.TransportError as exc:
                if attempt == 2:
                    raise CollectionError("附件下载网络连接失败，请稍后重试。") from exc
                time.sleep(2 ** attempt)
            except httpx.HTTPStatusError as exc:
                raise CollectionError(f"附件下载失败（HTTP {exc.response.status_code}）。") from exc
        raise CollectionError("附件下载失败。")
