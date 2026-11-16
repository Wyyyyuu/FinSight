# -*- coding: utf-8 -*-
from __future__ import annotations

import pytest
from backend.tools import cn_market_flow, concept_map


class _DummyResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_fetch_fund_flow_parses_rows(monkeypatch):
    def _fake_get(_url: str, params: dict, timeout: int, headers: dict):
        assert params['fid'] == 'f62'
        return _DummyResponse(
            200,
            {
                "data": {
                    "diff": [
                        {"f12": "000001", "f13": 0, "f14": "贵州茅台", "f2": "1680.5", "f3": "2.15", "f62": "1000", "f184": "3.2"}
                    ]
                }
            },
        )

    monkeypatch.setattr(cn_market_flow, "_http_get", _fake_get)

    result = cn_market_flow.fetch_fund_flow(limit=10)

    assert result["success"] is True
    assert result["count"] == 1
    assert result["items"][0]["symbol"] == "000001.SZ"
    assert result["items"][0]["change_percent"] == 2.15


def test_northbound_daily_turnover_units_and_missing_values(monkeypatch):
    def get(url, params, **kwargs):
        assert params['reportName'] == 'RPT_MUTUAL_DEALAMT'
        return _DummyResponse(200, {'success': True, 'result': {'data': [{
            'TRADE_DATE': '2026-09-21 00:00:00', 'NF_DEAL_AMT': 283911.86,
            'SSC_DEAL_AMT': 133832.25, 'ST_DEAL_AMT': None}]}})
    monkeypatch.setattr(cn_market_flow, '_http_get', get)
    result = cn_market_flow.fetch_northbound(limit=5)
    assert result['success'] is True
    assert result['as_of'] == '2026-09-21'
    row = result['items'][0]
    assert row['total_turnover'] == pytest.approx(283911860000)
    assert row['sh_turnover'] == 133832250000
    assert row['sz_turnover'] is None
    assert 'northbound_net' not in row and 'main_net_inflow' not in row


@pytest.mark.parametrize('module,fn', [
    (cn_market_flow, cn_market_flow.fetch_fund_flow),
    (cn_market_flow, cn_market_flow.fetch_northbound),
    (concept_map, concept_map.fetch_concept_map),
])
@pytest.mark.parametrize('failure', ['http', 'json', 'shape', 'timeout'])
def test_source_failure_is_not_successful_empty_list(monkeypatch, module, fn, failure):
    def get(*args, **kwargs):
        if failure == 'timeout':
            raise TimeoutError('supplier down')
        if failure == 'json':
            raise ValueError('invalid json')
        return _DummyResponse(503 if failure == 'http' else 200, {})
    monkeypatch.setattr(module, '_http_get', get)
    result = fn(limit=2)
    assert result['success'] is False
    assert result['items'] == []
    assert result['error']


def test_concept_filter_and_valid_empty(monkeypatch):
    monkeypatch.setattr(concept_map, '_http_get', lambda *a, **k: _DummyResponse(200, {
        'data': {'diff': [{'f12': 'BK1169', 'f14': 'Kimi概念', 'f3': 3.57, 'f62': None}]}}))
    result = concept_map.fetch_concept_map(keyword='Kimi')
    assert result['items'][0]['concept_code'] == 'BK1169'
    assert result['items'][0]['main_net_inflow'] is None
    result = concept_map.fetch_concept_map(keyword='not present')
    assert result['success'] is True and result['items'] == []
