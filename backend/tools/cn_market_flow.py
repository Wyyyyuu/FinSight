from __future__ import annotations

import logging
import os
from typing import Any

from backend.tools.cn_market_common import market_http_get as _http_get
from backend.tools.cn_market_common import CNMarketSourceError, response_json, market_result, market_failure
from backend.utils.quote import safe_float

logger = logging.getLogger(__name__)

_EASTMONEY_USER_AGENT = os.getenv("EASTMONEY_USER_AGENT", "Mozilla/5.0 (FinSight)")
_EASTMONEY_TIMEOUT = int(os.getenv("EASTMONEY_TIMEOUT", "12"))
_EASTMONEY_LIST_URL = "https://push2.eastmoney.com/api/qt/clist/get"


def _eastmoney_list(*, fs: str, fields: str, limit: int = 20, sort: str = 'f3') -> list[dict[str, Any]]:
    params = {
        "pn": "1",
        "pz": str(max(1, min(int(limit), 200))),
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": sort,
        "fs": fs,
        "fields": fields,
    }
    resp = _http_get(_EASTMONEY_LIST_URL, params=params, timeout=_EASTMONEY_TIMEOUT,
                    headers={"User-Agent": _EASTMONEY_USER_AGENT})
    data = response_json(resp).get('data')
    rows = data.get('diff') if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise CNMarketSourceError('行情源未返回有效列表，请稍后重试。')
    return rows


def _build_symbol(row: dict[str, Any]) -> str:
    code = str(row.get("f12") or "").strip()
    market_id = str(row.get("f13", "")).strip()
    if not code:
        return ""
    if market_id == "1":
        return f"{code}.SH"
    if market_id == "0":
        return f"{code}.SZ"
    return code


def fetch_fund_flow(*, limit: int = 20) -> dict[str, Any]:
    """Fetch A-share fund flow list from Eastmoney snapshot endpoint."""
    try:
        rows = _eastmoney_list(
            fs="m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
            fields="f12,f13,f14,f2,f3,f62,f184", limit=limit, sort='f62',
        )
    except Exception as exc:
        logger.info('fetch_fund_flow failed: %s', exc)
        return market_failure('eastmoney_clist', exc)
    items: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = _build_symbol(row)
        if not symbol:
            continue
        items.append(
            {
                "symbol": symbol,
                "name": str(row.get("f14") or "").strip() or symbol,
                "last_price": safe_float(row.get("f2")),
                "change_percent": safe_float(row.get("f3")),
                "main_net_inflow": safe_float(row.get("f62")),
                "main_net_inflow_ratio": safe_float(row.get("f184")),
            }
        )

    return market_result(items, 'eastmoney_clist', note='沪深 A 股按主力净流入排序；非交易时段为最近行情快照。')


def fetch_northbound(*, limit: int = 20) -> dict[str, Any]:
    """Published daily northbound turnover; net buying is no longer disclosed."""
    source = 'eastmoney_northbound_daily'
    params = {'reportName': 'RPT_MUTUAL_DEALAMT', 'columns': 'ALL',
              'pageSize': str(max(1, min(int(limit), 200))), 'pageNumber': '1',
              'sortColumns': 'TRADE_DATE', 'sortTypes': '-1', 'source': 'WEB', 'client': 'WEB'}
    try:
        payload = response_json(_http_get('https://datacenter-web.eastmoney.com/api/data/v1/get',
            params=params, timeout=_EASTMONEY_TIMEOUT, headers={'User-Agent': _EASTMONEY_USER_AGENT}))
        result = payload.get('result')
        rows = result.get('data') if isinstance(result, dict) else None
        if not isinstance(rows, list):
            raise CNMarketSourceError('北向成交数据格式异常，请稍后重试。')
        items = []
        for row in rows:
            if not isinstance(row, dict) or not row.get('TRADE_DATE'):
                continue
            item = {'trade_date': str(row['TRADE_DATE'])[:10]}
            # Eastmoney reports turnover in million CNY. Keep missing values missing.
            for field, target in (('SSC_DEAL_AMT', 'sh_turnover'), ('ST_DEAL_AMT', 'sz_turnover'), ('NF_DEAL_AMT', 'total_turnover')):
                value = safe_float(row.get(field))
                item[target] = value * 1_000_000 if value is not None else None
            items.append(item)
        return market_result(items, source, as_of=items[0]['trade_date'] if items else None,
            note='北向成交额按交易日盘后披露，非实时净流入；不再展示未披露的买入、卖出或净买入数据。')
    except Exception as exc:
        logger.info('fetch_northbound failed: %s', exc)
        return market_failure(source, exc)


__all__ = ["fetch_fund_flow", "fetch_northbound"]
