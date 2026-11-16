"""Resumable parallel download of the pinned public weight; verify full LFS SHA256."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import time
import requests
from huggingface_hub import get_hf_file_metadata, hf_hub_url

REVISION = '5617a9f61b028005a4858fdac845db406aefb181'
url = hf_hub_url('BAAI/bge-m3', 'pytorch_model.bin', revision=REVISION)
meta = get_hf_file_metadata(url)
assert len(meta.etag) == 64 and meta.size > 0
root = Path('/models/hub/models--BAAI--bge-m3')
chunks = root / 'download-parts'
chunks.mkdir(parents=True, exist_ok=True)
size = 8 * 1024 * 1024


def download(i):
    start, end = i * size, min((i+1)*size, meta.size)-1
    path = chunks / str(i)
    if path.exists() and path.stat().st_size == end-start+1:
        return
    for attempt in range(4):
        try:
            with requests.get(url + f'?download=true&part={i}', headers={'Range': f'bytes={start}-{end}'},
                              stream=True, timeout=(20,60)) as response:
                response.raise_for_status()
                assert response.status_code == 206 and response.headers['Content-Range'] == f'bytes {start}-{end}/{meta.size}'
                with path.with_suffix('.tmp').open('wb') as output:
                    for block in response.iter_content(1024*1024):
                        output.write(block)
            assert path.with_suffix('.tmp').stat().st_size == end-start+1
            os.replace(path.with_suffix('.tmp'), path)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


count = (meta.size + size - 1)//size
started = time.monotonic()
with ThreadPoolExecutor(max_workers=8) as pool:
    for n, future in enumerate(as_completed([pool.submit(download,i) for i in range(count)]),1):
        future.result()
        if n%8 == 0:
            print(f'{n}/{count} parts, {time.monotonic()-started:.1f}s', flush=True)
target = root / 'blobs' / meta.etag
target.parent.mkdir(parents=True, exist_ok=True)
hasher = hashlib.sha256()
with target.with_suffix('.verified-tmp').open('wb') as output:
    for i in range(count):
        with (chunks/str(i)).open('rb') as source:
            while block := source.read(1024*1024):
                hasher.update(block)
                output.write(block)
assert hasher.hexdigest() == meta.etag, 'Downloaded model SHA256 mismatch'
os.replace(target.with_suffix('.verified-tmp'), target)
snapshot = root / 'snapshots' / REVISION / 'pytorch_model.bin'
snapshot.parent.mkdir(parents=True, exist_ok=True)
if not snapshot.exists():
    snapshot.symlink_to('../../blobs/' + meta.etag)
print('Verified', meta.etag, meta.size, flush=True)
