"""Lightweight local entry point; shares the exact router with FinSight's API."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.annual_report_router import annual_report_router


def create_app() -> FastAPI:
    app = FastAPI(title="FinSight 中文年报工作台", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            f"http://{host}:{port}"
            for host in ("localhost", "127.0.0.1")
            for port in (5173, 5174, 8001)
        ],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization", "X-Session-Id"],
    )
    app.include_router(annual_report_router)
    dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "annual-reports"}

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        if path.startswith("api/"):
            raise HTTPException(404)
        target = (dist / path).resolve()
        if target.is_relative_to(dist.resolve()) and target.is_file():
            return FileResponse(target)
        if (dist / "index.html").is_file():
            return FileResponse(dist / "index.html")
        return {
            "message": "请先运行 npm run build --prefix frontend，然后打开 /annual-reports。",
            "docs": "/docs",
        }

    return app


app = create_app()
