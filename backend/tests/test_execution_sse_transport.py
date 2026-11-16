import json

import pytest

from backend.api.execution_router import _stream_sse_pipeline


@pytest.mark.asyncio
async def test_large_report_keeps_terminal_frame_across_bounded_writes():
    report = {'summary': '中文证据与结论' * 100000}

    async def events():
        yield {'type': 'token', 'content': '正文'}
        yield {'type': 'done', 'run_id': 'run-test', 'report': report}

    chunks = [chunk async for chunk in _stream_sse_pipeline(events())]
    assert all(len(chunk) <= 16 * 1024 for chunk in chunks)
    frames = b''.join(chunks).decode('utf-8').strip().split('\n\n')
    parsed = [json.loads(frame.removeprefix('data: ')) for frame in frames]
    assert [event['type'] for event in parsed] == ['token', 'done']
    assert parsed[-1]['report'] == report
    assert parsed[-1]['run_id'] == 'run-test'
