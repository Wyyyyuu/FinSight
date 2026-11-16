"""Check bounded CPU batches against the previous full-batch FP32 reranker."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend.rag.reranker import RerankerService
from experiments.docker_embedding.evaluate import DATA, HERE, save

os.environ['RAG_RERANKER_CACHE_DIR'] = str(ROOT/'experiments/lightrag/models')
data = json.loads(DATA.read_text(encoding='utf-8'))
service = RerankerService(force_backend='bge-reranker-onnx')
service._predict_onnx('warmup', ['warmup'])
rows = []
for lang in ['en', 'zh']:
    q = next(q for q in data['questions'] if q['language']==lang)
    docs = [d['contextual'] for d in data['documents'] if d['collection']==q['collection']][:24]
    start = time.perf_counter()
    before = list(service._onnx_model.rerank(q['question'], docs))
    before_ms = (time.perf_counter()-start)*1000
    start = time.perf_counter()
    after = service._predict_onnx(q['question'], docs)
    after_ms = (time.perf_counter()-start)*1000
    top = lambda scores: sorted(range(len(scores)), key=lambda i:(-scores[i], i))[:8]
    delta = max(abs(a-b) for a,b in zip(before,after))
    assert delta < 0.001 and top(before)==top(after), (lang,delta)
    rows.append(dict(language=lang,documents=len(docs),max_abs_score_diff=delta,
                     exact_top8_order=True,before_ms=before_ms,after_ms=after_ms))
    print(rows[-1], flush=True)
save(HERE/'cpu_batch_assessment.json', {'rows':rows,'batch_size':4,
     'note':'Same CPU FP32 weights and tokenization, length grouping only; two-query parity sample, not load testing.'})
