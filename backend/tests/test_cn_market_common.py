import pytest
import requests

from backend.tools import cn_market_common as common


def test_transient_timeout_retries_once_with_short_timeouts(monkeypatch):
    calls = []
    response = requests.Response()
    response.status_code = 200
    def get(url, **kwargs):
        calls.append(kwargs['timeout'])
        if len(calls) == 1:
            raise requests.ReadTimeout()
        return response
    monkeypatch.setattr(common, '_http_get_no_retry', get)
    assert common.market_http_get('https://source.test', timeout=12) is response
    assert calls == [(3, 5), (3, 5)]


def test_persistent_timeout_has_only_two_attempts(monkeypatch):
    calls = []
    def get(*args, **kwargs):
        calls.append(1)
        raise requests.ReadTimeout()
    monkeypatch.setattr(common, '_http_get_no_retry', get)
    with pytest.raises(requests.ReadTimeout):
        common.market_http_get('https://source.test')
    assert len(calls) == 2


@pytest.mark.parametrize('status,attempts', [(503, 2), (403, 1)])
def test_server_errors_retry_but_access_denials_do_not(monkeypatch, status, attempts):
    calls = []
    def get(*args, **kwargs):
        calls.append(1)
        response = requests.Response()
        response.status_code = status
        response._content = b''
        response._content_consumed = True
        return response
    monkeypatch.setattr(common, '_http_get_no_retry', get)
    assert common.market_http_get('https://source.test').status_code == status
    assert len(calls) == attempts
