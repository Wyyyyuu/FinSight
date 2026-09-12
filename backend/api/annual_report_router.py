"""HTTP boundary for the annual-report workspace.

Without the existing FinSight authentication this module is loopback-only.
Authenticated users receive separate SQLite workspaces; client supplied document
IDs and session headers never determine another user's storage directory.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from backend.annual_reports.demo import DEMO_COMPANY, demo_reports
from backend.annual_reports.documents import AnnualReportStorageError, AnnualReportStore
from backend.annual_reports.workflow import analyze_reports

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
logger = logging.getLogger(__name__)


class StorageSafeRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def guarded(request: Request):
            try:
                return await handler(request)
            except AnnualReportStorageError as exc:
                logger.error(
                    "Annual report storage unavailable: %s", type(exc).__name__
                )
                raise HTTPException(
                    503, "年报存储暂时不可用，请检查磁盘空间与目录权限后重试。"
                ) from exc

        return guarded


annual_report_router = APIRouter(
    prefix="/api/annual-reports", tags=["Annual reports"], route_class=StorageSafeRoute
)


def _workspace_root() -> Path:
    return Path(
        os.getenv("ANNUAL_REPORT_DATA_DIR")
        or Path(__file__).resolve().parents[2] / "data" / "annual_reports"
    )


@lru_cache(maxsize=32)
def _store_at(directory: str) -> AnnualReportStore:
    return AnnualReportStore(directory)


def get_report_store(request: Request) -> AnnualReportStore:
    identity = getattr(request.state, "rag_authenticated_user", None)
    user_id = str(identity.get("user_id") or "") if isinstance(identity, dict) else ""
    if user_id:
        scope = "user_" + hashlib.sha256(user_id.encode()).hexdigest()[:32]
    else:
        host = request.client.host if request.client else ""
        loopback = {"127.0.0.1", "::1", "localhost"}
        origin = request.headers.get("origin")
        try:
            origin_host = urlsplit(origin).hostname if origin else None
        except ValueError:
            origin_host = None
        proxy_headers = (
            "forwarded",
            "x-forwarded-for",
            "x-real-ip",
            "cf-connecting-ip",
        )
        if (
            host not in loopback
            or request.url.hostname not in loopback
            or (origin is not None and origin_host not in loopback)
            or any(key in request.headers for key in proxy_headers)
        ):
            raise HTTPException(401, "年报工作区需要登录；无登录模式仅允许本机访问。")
        scope = "local"
    return _store_at(str(_workspace_root() / scope))


ReportStoreDep = Annotated[AnnualReportStore, Depends(get_report_store)]


class AnalysisRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    document_ids: list[str] = Field(min_length=1, max_length=30)
    years: list[int] | None = Field(default=None, max_length=15)
    max_retries: int = Field(default=1, ge=0, le=2)
    mode: Literal["hybrid", "bm25"] = "hybrid"

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2:
            raise ValueError("请输入至少两个字的问题。")
        return value

    @field_validator("document_ids")
    @classmethod
    def valid_document_ids(cls, values: list[str]) -> list[str]:
        if any(not value or len(value) > 128 for value in values):
            raise ValueError("文档编号无效。")
        return list(dict.fromkeys(values))

    @field_validator("years")
    @classmethod
    def valid_years(cls, values: list[int] | None) -> list[int] | None:
        if values is not None and (
            not values
            or any(
                v < 1900 or v > datetime.now(timezone.utc).astimezone().year + 1
                for v in values
            )
        ):
            raise ValueError("年度范围无效。")
        return sorted(set(values)) if values else None


@annual_report_router.get("/health")
def health(store: ReportStoreDep) -> dict:
    return {
        "status": "ok",
        "document_count": len(store.list_documents()),
        "max_upload_mb": 20,
        "embedding_provider": os.getenv("ANNUAL_REPORT_EMBEDDING_PROVIDER", "bm25"),
        "retrieval": store.retrieval_status(),
        "llm_configured": all(
            os.getenv(k)
            for k in (
                "ANNUAL_REPORT_LLM_BASE_URL",
                "ANNUAL_REPORT_LLM_API_KEY",
                "ANNUAL_REPORT_LLM_MODEL",
            )
        ),
    }


@annual_report_router.get("/documents")
def list_documents(store: ReportStoreDep) -> dict:
    return {"documents": store.list_documents()}


@annual_report_router.post("/documents", status_code=201)
async def upload_document(
    file: Annotated[UploadFile, File()],
    company: Annotated[str, Form(min_length=1, max_length=120)],
    year: Annotated[int, Form()],
    store: ReportStoreDep,
) -> dict:
    if (
        not company.strip()
        or year < 1900
        or year > datetime.now(timezone.utc).astimezone().year + 1
    ):
        raise HTTPException(422, "请输入公司名称与有效年度。")
    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "文件不能超过 20 MB。")
        return await run_in_threadpool(
            store.ingest_bytes, content, file.filename or "", company.strip(), year
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        await file.close()


@annual_report_router.get("/documents/{document_id}/pages/{page}")
def get_page(document_id: str, page: int, store: ReportStoreDep) -> dict:
    document = store.get_document(document_id)
    if not document or page < 1:
        raise HTTPException(404, "未找到该文档页。")
    result = store.get_page(document_id, page)
    if result is None:
        raise HTTPException(404, "未找到该文档页。")
    return {
        **result,
        "document_id": document_id,
        "page": page,
        "company": document["company"],
        "year": document["year"],
        "filename": document["filename"],
    }


@annual_report_router.post("/demo", status_code=201)
def load_demo(store: ReportStoreDep) -> dict:
    return {
        "documents": [
            store.ingest_bytes(data, filename, DEMO_COMPANY, year)
            for filename, year, data in demo_reports()
        ]
    }


@annual_report_router.post("/analyze")
def analyze(payload: AnalysisRequest, store: ReportStoreDep) -> dict:
    if any(store.get_document(doc_id) is None for doc_id in payload.document_ids):
        raise HTTPException(404, "所选文档不存在或不属于当前工作区，请刷新资料列表。")
    try:
        return analyze_reports(
            store,
            payload.question,
            payload.document_ids,
            years=payload.years,
            max_retries=payload.max_retries,
            mode=payload.mode,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
