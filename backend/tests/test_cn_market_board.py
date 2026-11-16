# -*- coding: utf-8 -*-
from __future__ import annotations

import pytest
from backend.tools import cn_market_board


class _DummyResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_real_limit_pools_price_units_market_and_dates(monkeypatch):
    calls = []
    def get(url, params, **kwargs):
        calls.append(url)
        down = url.endswith('getTopicDTPool')
        return _DummyResponse(200, {'rc': 0, 'data': {'qdate': 19990101, 'pool': [{
            'c': '000668' if down else '000504', 'm': 0, 'n': '测试股票', 'p': 12110,
            'zdp': -9.97 if down else 9.99, 'hs': 1.78, 'amount': 71297625}]}})
    monkeypatch.setattr(cn_market_board, '_http_get', get)
    result = cn_market_board.fetch_limit_board(limit=2)
    assert result['success'] is True and result['count'] == 2
    assert {row['limit_side'] for row in result['items']} == {'up', 'down'}
    assert result['items'][0]['last_price'] == 12.11
    assert result['items'][0]['symbol'] == '000504.SZ'
    assert result['items'][0]['trade_date'] == result['as_of'] != '1999-01-01'
    assert all('push2ex.eastmoney.com/getTopic' in url for url in calls)
    assert all('main_net_inflow' not in row for row in result['items'])


def test_limit_pool_partial_failure_is_labelled(monkeypatch):
    def get(url, **kwargs):
        if url.endswith('getTopicDTPool'):
            raise TimeoutError()
        return _DummyResponse(200, {'rc': 0, 'data': {'pool': [{'c': '600001', 'm': 1, 'p': 15000}]}})
    monkeypatch.setattr(cn_market_board, '_http_get', get)
    result = cn_market_board.fetch_limit_board()
    assert result['success'] is True and result['count'] == 1
    assert '跌停池' in result['warning']


def test_limit_pool_empty_nontrading_day(monkeypatch):
    monkeypatch.setattr(cn_market_board, '_http_get', lambda *a, **k: _DummyResponse(200, {'rc': 0, 'data': None}))
    result = cn_market_board.fetch_limit_board()
    assert result['success'] is True and result['items'] == []


@pytest.mark.parametrize('fn', [cn_market_board.fetch_limit_board, cn_market_board.fetch_lhb])
def test_board_source_failure_is_explicit(monkeypatch, fn):
    monkeypatch.setattr(cn_market_board, '_http_get', lambda *a, **k: _DummyResponse(503, {}))
    result = fn()
    assert result['success'] is False and result['error']


def test_fetch_lhb_parses_rows(monkeypatch):
    def _fake_get(_url: str, params: dict, timeout: int, headers: dict):
        return _DummyResponse(
            200,
            {
                "result": {
                    "data": [
                        {
                            "SECURITY_CODE": "002594",
                            "SECURITY_NAME_ABBR": "比亚迪",
                            "TRADE_DATE": "2026-02-28",
                            "CLOSE_PRICE": "200",
                            "CHANGE_RATE": "5.0",
                            "BILLBOARD_NET_AMT": "1000000",
                            "BILLBOARD_BUY_AMT": "2000000",
                            "BILLBOARD_SELL_AMT": "1000000",
                            "EXPLANATION": "日涨幅偏离值达7%",
                        }
                    ]
                }
            },
        )

    monkeypatch.setattr(cn_market_board, "_http_get", _fake_get)

    result = cn_market_board.fetch_lhb(limit=20)

    assert result["success"] is True
    assert result["count"] == 1
    assert result["items"][0]["symbol"] == "002594"
    assert result["items"][0]["change_percent"] == 5.0

    assert result['items'][0]['net_buy'] == 1000000
    assert result['items'][0]['buy_amt'] == 2000000
    assert result['items'][0]['sell_amt'] == 1000000
    assert result['items'][0]['reason'] == '日涨幅偏离值达7%'
