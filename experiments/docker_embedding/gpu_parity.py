"""Compare identical FP32 reranker weights/tokenization across CPU and CUDA."""
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from backend.rag.reranker import RerankerService
from backend.rag.bm25 import BM25Index
from experiments.docker_embedding.evaluate import HERE, DATA, save, sha


def main():
    data=json.loads(DATA.read_text(encoding='utf-8'))
    os.environ['RAG_RERANKER_CACHE_DIR']=str(ROOT/'experiments/lightrag/models')
    cpu=RerankerService(force_backend='bge-reranker-onnx')
    gpu=RerankerService(force_backend='bge-reranker-http')
    assert 'rerank_score' in cpu.rerank('revenue',[{'content':'revenue increased'}],top_n=1)[0]
    cases=[]
    for language in ['en','zh']:
        cases += sorted([q for q in data['questions'] if q['language']==language and q['split']!='test'],
                        key=lambda q:sha(q['id']))[:2]
    rows=[]
    for q in cases:
        docs=[d for d in data['documents'] if d['collection']==q['collection']]
        lexical=BM25Index([d['contextual'] for d in docs]).scores(q['question'])
        docs=[docs[i] for i in sorted(range(len(docs)),key=lambda i:-lexical[i])[:24]]
        started=time.perf_counter()
        old=cpu.rerank(q['question'],docs,top_n=len(docs),content_key='contextual')
        cpu_ms=(time.perf_counter()-started)*1000
        started=time.perf_counter()
        new=gpu.rerank(q['question'],docs,top_n=len(docs),content_key='contextual')
        gpu_ms=(time.perf_counter()-started)*1000
        scores={d['id']:d['rerank_score'] for d in old}
        difference=max(abs(d['rerank_score']-scores[d['id']]) for d in new)
        same_set={d['id'] for d in old[:8]}=={d['id'] for d in new[:8]}
        rows.append(dict(id=q['id'],language=q['language'],pairs=len(docs),max_absolute_score_difference=difference,
                         top8_same_set=same_set,top8_same_order=[d['id'] for d in old[:8]]==[d['id'] for d in new[:8]],
                         cpu_ms=cpu_ms,gpu_ms=gpu_ms))
        save(HERE/'gpu_parity.json',{'rows':rows,'complete':False})
        print(rows[-1],flush=True)
        assert same_set and difference<.01, 'GPU/CPU reranker parity failed'
    save(HERE/'gpu_parity.json',{'rows':rows,'complete':True,'tolerance':.01})


if __name__=='__main__':
    main()
