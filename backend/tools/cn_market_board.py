from __future__ import annotations

import logging
import os
from typing import Any

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from backend.tools.cn_market_common import market_http_get as _http_get
from backend.tools.cn_market_common import CHINA_TZ, CNMarketSourceError, response_json, market_result, market_failure
from backend.tools.cn_market_flow import _build_symbol
from backend.utils.quote import safe_float

logger = logging.getLogger(__name__)

_EASTMONEY_USER_AGENT = os.getenv("EASTMONEY_USER_AGENT", "Mozilla/5.0 (FinSight)")
_EASTMONEY_TIMEOUT = int(os.getenv("EASTMONEY_TIMEOUT", "12"))
_EASTMONEY_DATA_CENTER_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"


def _limit_pool(direction: str, trade_date: str, limit: int) -> list[dict[str, Any]]:
    endpoint, order = ('getTopicZTPool', 'fbt:asc') if direction == 'up' else ('getTopicDTPool', 'fund:asc')
    params = {'ut': '7eea3edcaed734bea9cbfc24409ed989', 'dpt': 'wz.ztzt',
              'Pageindex': '0', 'pagesize': str(max(1, min(int(limit), 200))),
              'date': trade_date.replace('-', ''), 'sort': order}
    payload = response_json(_http_get(f'https://push2ex.eastmoney.com/{endpoint}',
        params=params, timeout=_EASTMONEY_TIMEOUT, headers={'User-Agent': _EASTMONEY_USER_AGENT}))
    data = payload.get('data')
    if data is None:
        return []  # No pool is published before trading or on non-trading days.
    rows = data.get('pool') if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise CNMarketSourceError('涨跌停数据格式异常，请稍后重试。')
    items = []
    for row in rows:
        if not isinstance(row, dict) or not row.get('c'):
            continue
        price = safe_float(row.get('p'))
        items.append({'symbol': _build_symbol({'f12': row['c'], 'f13': row.get('m')}),
            'name': row.get('n') or row['c'], 'last_price': price / 1000 if price is not None else None,
            'change_percent': safe_float(row.get('zdp')), 'turnover_rate': safe_float(row.get('hs')),
            'amount': safe_float(row.get('amount')), 'limit_side': direction, 'trade_date': trade_date})
    return items


def fetch_limit_board(*, limit: int = 20) -> dict[str, Any]:
    """Return genuine limit-up/down pools; prices are milli-yuan upstream."""
    trade_date = datetime.now(CHINA_TZ).date().isoformat()
    items, errors = [], []
    # Both pools run concurrently, so an unavailable supplier does not double latency.
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = [(side, executor.submit(_limit_pool, side, trade_date, limit)) for side in ('up', 'down')]
        for side, future in pending:
            try:
                items.extend(future.result())
            except Exception as exc:
                logger.info('limit pool %s failed: %s', side, exc)
                errors.append('涨停池' if side == 'up' else '跌停池')
    if len(errors) == 2:
        return market_failure('eastmoney_limit_pool', CNMarketSourceError('涨跌停行情源暂时不可用，请稍后重试。'))
    return market_result(items, 'eastmoney_limit_pool', as_of=trade_date,
        warning=f"{'、'.join(errors)}获取失败，当前仅显示部分数据。" if errors else None,
        note=f'涨停、跌停各最多 {limit} 条；盘前或非交易日可能暂无数据。')


def fetch_lhb(*, limit: int = 20) -> dict[str, Any]:
    """Fetch published LongHuBang entries, most recent trading dates first."""
    params = {
        "reportName": "RPT_DAILYBILLBOARD_DETAILSNEW",
        "columns": "ALL",
        "pageNumber": "1",
        "pageSize": str(max(1, min(int(limit), 100))),
        "sortTypes": "-1",
        "sortColumns": "TRADE_DATE",
        "source": "WEB",
        "client": "WEB",
    }
    items: list[dict[str, Any]] = []
    try:
        resp = _http_get(
            _EASTMONEY_DATA_CENTER_URL,
            params=params,
            timeout=_EASTMONEY_TIMEOUT,
            headers={"User-Agent": _EASTMONEY_USER_AGENT},
        )
        payload = response_json(resp)
        result = payload.get("result")
        rows = result.get("data") if isinstance(result, dict) else None
        if not isinstance(rows, list):
            raise CNMarketSourceError('龙虎榜数据格式异常，请稍后重试。')
        if rows:
            for row in rows:
                if not isinstance(row, dict):
                    continue
                symbol = str(row.get("SECURITY_CODE") or "").strip()
                if not symbol:
                    continue
                items.append(
                    {
                        "symbol": symbol,
                        "name": str(row.get("SECURITY_NAME_ABBR") or "").strip() or symbol,
                        "trade_date": row.get("TRADE_DATE"),
                        "close_price": safe_float(row.get("CLOSE_PRICE")),
                        "change_percent": safe_float(row.get("CHANGE_RATE")),
                        "net_buy": safe_float(row.get("BILLBOARD_NET_AMT")),
                        "buy_amt": safe_float(row.get("BILLBOARD_BUY_AMT")),
                        "sell_amt": safe_float(row.get("BILLBOARD_SELL_AMT")),
                        "reason": row.get("EXPLANATION") or row.get("EXPLAIN"),
                    }
                )
    except Exception as exc:
        logger.info("fetch_lhb failed: %s", exc)
        return market_failure('eastmoney_datacenter', exc)

    return market_result(items, 'eastmoney_datacenter',
        as_of=str(items[0]['trade_date'])[:10] if items else None,
        note='龙虎榜为盘后披露数据；各条目按上榜日期排序。')


__all__ = ["fetch_limit_board", "fetch_lhb"]
