"""Slow vendor I/O must leave the execution stream's event loop responsive."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from backend.agents.base_agent import BaseFinancialAgent
from backend.agents.fundamental_agent import FundamentalAgent
from backend.agents.macro_agent import MacroAgent
from backend.agents.news_agent import NewsAgent
from backend.agents.price_agent import PriceAgent
from backend.agents.risk_agent import RiskAgent
from backend.agents.technical_agent import TechnicalAgent
from backend.graph.cancellation import get_cancel_event, reset_cancel_event, set_cancel_event


class Cache:
    def get(self, key):
        return None

    def set(self, key, value, ttl):
        pass


class SlowVendor:
    def __init__(self, payload):
        self.payload = payload
        self.started = threading.Event()
        self.released = threading.Event()
        self.contexts = []
        self.heartbeat_during_io = False

    def __call__(self, *args, **kwargs):
        self.contexts.append(get_cancel_event())
        self.started.set()
        # If called on the event loop this times out before the heartbeat can run.
        self.heartbeat_during_io = self.released.wait(timeout=0.5)
        return self.payload


async def with_heartbeat(coroutine, vendor):
    cancel_event = asyncio.Event()
    token = set_cancel_event(cancel_event)
    task = asyncio.create_task(coroutine)
    try:
        async def heartbeat():
            while not vendor.started.is_set():
                await asyncio.sleep(0.001)
            vendor.released.set()

        await asyncio.wait_for(asyncio.gather(task, heartbeat()), timeout=3)
        assert vendor.heartbeat_during_io, "synchronous I/O blocked the SSE heartbeat"
        assert vendor.contexts and all(value is cancel_event for value in vendor.contexts)
        return task.result()
    finally:
        vendor.released.set()
        reset_cancel_event(token)
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["price_quote", "price_history", "technical", "fundamental", "macro", "news", "risk"])
async def test_vendor_io_keeps_heartbeat_and_request_context(kind):
    vendor = SlowVendor({})
    if kind.startswith("price"):
        vendor.payload = {"price": 100, "source": "test"} if kind == "price_quote" else {}
        tools = SimpleNamespace(_fetch_with_yfinance=vendor if kind == "price_quote" else lambda _: {"price": 100})
        if kind == "price_history":
            tools.get_stock_historical_data = vendor
        agent = PriceAgent(None, Cache(), tools)
    elif kind == "technical":
        agent = TechnicalAgent(None, Cache(), SimpleNamespace(get_stock_historical_data=vendor))
    elif kind == "fundamental":
        agent = FundamentalAgent(None, Cache(), SimpleNamespace(get_financial_statements=vendor))
    elif kind == "macro":
        agent = MacroAgent(None, Cache(), SimpleNamespace(get_fred_data=vendor))
    elif kind == "news":
        vendor.payload = []
        agent = NewsAgent(None, Cache(), SimpleNamespace(_fetch_with_finnhub_news=vendor))
    else:
        vendor.payload = {"price": 100, "change_percent": 1, "source": "test"}
        agent = RiskAgent(None, Cache(), SimpleNamespace(get_stock_price=vendor))
    operation = agent.research("NVDA", "NVDA") if kind == "risk" else agent._initial_search("NVDA", "NVDA")
    assert await with_heartbeat(operation, vendor) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("call_with", ["query", "ticker", "positions", "none", "fallback"])
async def test_reflection_tools_keep_heartbeat(call_with):
    vendor = SlowVendor("research evidence " * 6)
    agent = BaseFinancialAgent(None, Cache(), None)
    entry = {"func": vendor, "call_with": call_with}
    registry = {"search": entry}
    if call_with == "fallback":
        def fail(*args):
            raise RuntimeError("vendor unavailable")
        registry["specialized"] = {"func": fail, "call_with": "ticker"}
    agent._get_tool_registry = lambda: registry
    result = await with_heartbeat(agent._targeted_search([{"tool": "specialized" if call_with == "fallback" else "search", "query": "outlook"}], "NVDA"), vendor)
    assert result == vendor.payload


@pytest.mark.asyncio
async def test_news_stream_keeps_heartbeat():
    vendor = SlowVendor([])
    agent = NewsAgent(None, Cache(), SimpleNamespace(_fetch_with_finnhub_news=vendor))

    async def consume():
        return [event async for event in agent.analyze_stream("news", "NVDA")]

    assert await with_heartbeat(consume(), vendor)


@pytest.mark.asyncio
async def test_cancel_returns_while_vendor_call_is_pending():
    vendor = SlowVendor({"price": 100})
    agent = PriceAgent(None, Cache(), SimpleNamespace(_fetch_with_yfinance=vendor))
    cancel_event = asyncio.Event()
    token = set_cancel_event(cancel_event)
    task = asyncio.create_task(agent._fetch_realtime_quote("NVDA"))
    try:
        async def wait_started():
            while not vendor.started.is_set():
                await asyncio.sleep(0.001)
        await asyncio.wait_for(wait_started(), timeout=0.2)
        assert not task.done()
        cancel_event.set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.1)
        assert vendor.contexts == [cancel_event]
    finally:
        vendor.released.set()
        reset_cancel_event(token)
        await asyncio.gather(task, return_exceptions=True)
