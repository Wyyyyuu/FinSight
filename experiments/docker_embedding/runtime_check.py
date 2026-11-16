"""Uncached end-to-end timings through the actual main RAG service, isolated RAM collections."""
import json
import os
from pathlib import Path
from statistics import mean
import sys
import time
import hashlib
import requests

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from backend.rag.embedder import EmbeddingService
from backend.rag.hybrid_service import HybridRAGService, RAGDocument
from backend.rag.reranker import RerankerService
from experiments.docker_embedding.evaluate import HERE, DATA, sha, save, reranker_identity
from experiments.lightrag.metrics import retrieval_metrics, percentile


def main():
    data=json.loads(DATA.read_text(encoding='utf-8'))
    selection=json.loads((HERE/'selection.json').read_text())
    winner=selection['winner']
    live_health=requests.get('http://127.0.0.1:8010/health',timeout=5).json()
    assert live_health==selection['seal']['model']
    assert reranker_identity()==selection['seal']['reranker']
    assert hashlib.sha256(DATA.read_bytes()).hexdigest()==selection['seal']['dataset_sha256']
    seal={'selection':selection,'model':live_health,'reranker':reranker_identity(),
          'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    modes={'m3_dense':'dense_only','m3_sparse_rrf':'legacy','m3_bm25_rrf':'bm25'}
    os.environ['RAG_RERANKER_CACHE_DIR']=str(ROOT/'experiments/lightrag/models')
    reranker=RerankerService(force_backend=os.getenv('RAG_EVAL_RERANKER','bge-reranker-onnx'))
    rows=[]
    for language in ['en','zh']:
        choices=sorted([q for q in data['questions'] if q['language']==language and
                        (q['split']=='test' if language=='en' else True)],key=lambda q:sha(q['id']))
        collection=choices[0]['collection']
        queries=[q for q in choices if q['collection']==collection][:5]
        docs=[d for d in data['documents'] if d['collection']==collection]
        for arm,backend in [('hash_legacy','hash'),(winner,'bge-m3-http')]:
            os.environ['RAG_LEXICAL_MODE']=modes[winner] if arm==winner else 'legacy'
            # Match the old fallback dimension, not force-hash's legacy 96-dim default.
            os.environ['RAG_HASH_DIM']='1024'
            embedder=EmbeddingService(force_backend=backend)
            service=HybridRAGService(backend='memory',vector_dim=1024,rrf_k=60,embedder=embedder)
            started=time.perf_counter()
            service.ingest_documents([RAGDocument(collection='ws:docker-runtime-check',source_id=d['id'],
                                                  content=d['contextual'],scope='ephemeral') for d in docs])
            ingest_ms=(time.perf_counter()-started)*1000
            for q in queries:
                for repeat in range(3):
                    t=time.perf_counter()
                    hits=service.hybrid_search(q['question'],collection='ws:docker-runtime-check',top_k=24)
                    retrieve_ms=(time.perf_counter()-t)*1000
                    t=time.perf_counter()
                    final=reranker.rerank(q['question'],hits,top_n=8)
                    rerank_ms=(time.perf_counter()-t)*1000
                    assert len(final)==min(8,len(hits)) and all('rerank_score' in d for d in final)
                    profile=hits[0]['metadata']['_embedding_profile']
                    assert profile.startswith('hash:1024:' if arm=='hash_legacy' else 'bge-m3:1024:5617a9')
                    rows.append(dict(arm=arm,language=language,query_id=q['id'],repeat=repeat,
                                     corpus_docs=len(docs),profile=profile,ingest_ms=ingest_ms,
                                     retrieve_ms=retrieve_ms,rerank_ms=rerank_ms,total_ms=retrieve_ms+rerank_ms,
                                     top24=retrieval_metrics([d['source_id'] for d in hits],q['required'],24),
                                     top8=retrieval_metrics([d['source_id'] for d in final],q['required'],8)))
            print('checked',language,arm,len(queries),flush=True)
    summary={}
    for arm in ['hash_legacy',winner]:
        measured=[r for r in rows if r['arm']==arm and r['repeat']>0]
        summary[arm]={name:{'p50':percentile([r[name] for r in measured],.5),
                           'p95':percentile([r[name] for r in measured],.95)}
                      for name in ['retrieve_ms','rerank_ms','total_ms']}
        summary[arm]['warm_samples']=len(measured)
    save(HERE/'runtime_assessment.json',{'seal':seal,'rows':rows,'summary':summary,
                                       'note':'Actual service path; no query or rerank score cache. Repeat0 excludes cold load from warm summary. Small single-report latency sample; not load testing.'})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
