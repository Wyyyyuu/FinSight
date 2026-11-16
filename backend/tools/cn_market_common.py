"""Response contracts for public A-share data (no model calls)."""
from datetime import datetime, timedelta, timezone
from typing import Any
import requests

from backend.tools.http import _http_get_no_retry

CHINA_TZ = timezone(timedelta(hours=8))


class CNMarketSourceError(ValueError):
    pass


def market_http_get(url: str, *, timeout: int = 12, **kwargs):
    """One bounded retry for intermittent public-source transport failures.

    Disable the shared adapter's three implicit retries. Each attempt has a
    three-second connection and at most five-second read timeout.
    """
    for attempt in range(2):
        try:
            response = _http_get_no_retry(url, timeout=(3, min(timeout, 5)), **kwargs)
        except requests.RequestException:
            if attempt == 1:
                raise
            continue
        if response.status_code >= 500 and attempt == 0:
            response.close()
            continue
        return response


def response_json(response) -> dict[str, Any]:
    if getattr(response, 'status_code', 0) != 200:
        raise CNMarketSourceError('行情源暂时不可用，请稍后重试。')
    payload = response.json()
    if not isinstance(payload, dict) or payload.get('success') is False or payload.get('rc', 0) != 0:
        raise CNMarketSourceError('行情源返回异常，请稍后重试。')
    return payload


def market_result(items: list[dict[str, Any]], source: str, **extra) -> dict[str, Any]:
    return {'success': True, 'items': items, 'count': len(items), 'source': source,
            'market': 'CN', 'retrieved_at': datetime.now(CHINA_TZ).isoformat(timespec='seconds'), **extra}


def market_failure(source: str, exc: Exception) -> dict[str, Any]:
    message = str(exc) if isinstance(exc, CNMarketSourceError) else '行情源连接失败或超时，请稍后重试。'
    return market_result([], source, success=False, error=message)
