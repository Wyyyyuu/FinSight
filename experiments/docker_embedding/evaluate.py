"""Read-only corpus replay: select on dev, then evaluate locked retrieval arms."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sqlite3
from statistics import mean
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import requests
import tiktoken
from backend.rag.embedder import EmbeddingService, _hash_encode_batch
from backend.rag.bm25 import BM25Index, reciprocal_rank_fusion
from backend.rag.evidence_pack import balanced_evidence
from backend.rag.reranker import RerankerService
from experiments.lightrag.metrics import retrieval_metrics, percentile

HERE = Path(__file__).parent
DATA = ROOT / 'experiments/financial_retrieval_v2/dataset.json'
ARMS = ['hash_legacy', 'hash_bm25', 'm3_dense', 'm3_sparse_rrf', 'm3_bm25_rrf']
TOKENS = tiktoken.get_encoding('cl100k_base')


@lru_cache(maxsize=1)
def reranker_identity():
    model_root = ROOT / 'experiments/lightrag/models/models--BAAI--bge-reranker-base/snapshots'
    files = {str(p.relative_to(model_root)): hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
             for p in sorted(model_root.rglob('*')) if p.is_file()}
    assert files, 'Missing reranker model files'
    identity = {'files': files, 'fastembed': importlib.metadata.version('fastembed'),
                'tokenizers':importlib.metadata.version('tokenizers'),
                'onnxruntime':importlib.metadata.version('onnxruntime'),
                'batch_size':int(os.getenv('RAG_RERANKER_BATCH_SIZE','4')),
                'code': hashlib.sha256((ROOT/'backend/rag/reranker.py').read_bytes()).hexdigest()}
    if os.getenv('RAG_EVAL_RERANKER') == 'bge-reranker-http':
        health = requests.get('http://127.0.0.1:8011/health',timeout=5).json()
        assert health['ready'] and health['provider']=='CUDAExecutionProvider'
        identity['runtime'] = {k:health[k] for k in ['model','provider','onnxruntime','weights_sha256','revision','tf32','batch_size','tokenizers','fastembed']}
        identity['server_sha'] = hashlib.sha256((ROOT/'deploy/reranker/server.py').read_bytes()).hexdigest()
    return identity


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def save(path, obj):
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def metric(docs, required, k):
    return retrieval_metrics([d['id'] for d in docs], required, k)


def context(d):
    return f'[{d["id"]}] Report: {d["collection"]}\n{d["safe"]}\n'


@lru_cache(maxsize=20000)
def text_tokens(text):
    return len(TOKENS.encode(text))


class Cache:
    def __init__(self, health):
        (HERE / 'cache').mkdir(exist_ok=True)
        self.db = sqlite3.connect(HERE / 'cache/vectors.sqlite')
        self.db.execute('CREATE TABLE IF NOT EXISTS vectors (id TEXT PRIMARY KEY, dense BLOB, sparse TEXT)')
        self.profile = f'{health["model"]}:{health["dim"]}:{health["revision"]}'
        self.encoder = EmbeddingService(force_backend='bge-m3-http')

    def key(self, text):
        return sha(self.profile + '\n' + text)

    def prepare(self, texts, *, offline=False):
        unique = list(dict.fromkeys(texts))
        missing = [t for t in unique if not self.db.execute('SELECT 1 FROM vectors WHERE id=?', (self.key(t),)).fetchone()]
        if offline and missing:
            raise RuntimeError(f'Offline replay requires {len(missing)} missing embeddings; run --prepare-only first')
        start = time.perf_counter()
        for i in range(0, len(missing), 32):
            batch = missing[i:i+32]
            result = self.encoder.encode(batch)
            assert result.profile == self.profile
            for text, dense, sparse in zip(batch, result.dense, result.sparse):
                self.db.execute('INSERT INTO vectors VALUES (?,?,?)',
                                (self.key(text), np.asarray(dense, dtype=np.float32).tobytes(), json.dumps(sparse.weights)))
            self.db.commit()
            print(f'encoded {min(i+32,len(missing))}/{len(missing)}, elapsed {time.perf_counter()-start:.1f}s', flush=True)
        return {'new_texts': len(missing), 'seconds': time.perf_counter()-start}

    @lru_cache(maxsize=20000)
    def get(self, text):
        row = self.db.execute('SELECT dense,sparse FROM vectors WHERE id=?', (self.key(text),)).fetchone()
        return np.frombuffer(row[0], dtype=np.float32), json.loads(row[1])


class Engine:
    def __init__(self, docs, cache):
        self.docs, self.cache = docs, cache
        self.indexes = {}
        self.hashes = {}

    def hash(self, text):
        if text not in self.hashes:
            r = _hash_encode_batch([text], 1024)
            self.hashes[text] = (np.asarray(r.dense[0]), r.sparse[0].weights)
        return self.hashes[text]

    def ranked(self, query, collections, arm):
        key = tuple(sorted(set(collections)))
        if key not in self.indexes:
            docs = [d for d in self.docs if d['collection'] in key]
            texts = [d['contextual'] for d in docs]
            self.indexes[key] = (docs, BM25Index(texts),
                                 np.asarray([self.cache.get(t)[0] for t in texts]),
                                 [self.cache.get(t)[1] for t in texts],
                                 np.asarray([self.hash(t)[0] for t in texts]),
                                 [self.hash(t)[1] for t in texts])
        docs, bm25, matrix, sparse, hashes, hash_sparse = self.indexes[key]
        if not docs:
            return []
        qd, qs = self.hash(query) if arm.startswith('hash') else self.cache.get(query)
        dense = (hashes if arm.startswith('hash') else matrix) @ qd
        if arm == 'm3_dense':
            order = sorted(range(len(docs)), key=lambda i: (-dense[i], i))
        else:
            if arm.endswith('bm25') or arm == 'm3_bm25_rrf':
                lexical = bm25.scores(query)
            else:
                lexical = [sum(v * d.get(k, 0) for k, v in qs.items())
                           for d in (hash_sparse if arm == 'hash_legacy' else sparse)]
            order = reciprocal_rank_fusion(dense, lexical, rrf_k=60)
        return [docs[i] for i in order]


class Rerank:
    def __init__(self):
        os.environ['RAG_RERANKER_CACHE_DIR'] = str(ROOT / 'experiments/lightrag/models')
        self.service = RerankerService(force_backend=os.getenv('RAG_EVAL_RERANKER','bge-reranker-onnx'))
        self.profile = sha(json.dumps(reranker_identity(), sort_keys=True))
        self.db = sqlite3.connect(HERE / 'cache/rerank.sqlite')
        self.db.execute('CREATE TABLE IF NOT EXISTS scores (id TEXT PRIMARY KEY, score REAL)')

    def run(self, query, docs):
        started = time.perf_counter()
        keys = [sha(self.profile + '\n' + query + '\n' + d['contextual']) for d in docs]
        scores = {k: row[0] for k in keys if (row := self.db.execute('SELECT score FROM scores WHERE id=?', (k,)).fetchone())}
        missing = [d for d, k in zip(docs, keys) if k not in scores]
        if missing:
            result = self.service.rerank(query, missing, top_n=len(missing), content_key='contextual')
            assert len(result) == len(missing) and all(np.isfinite(d['rerank_score']) for d in result), 'Reranker fallback is not a valid test'
            for d in result:
                k = sha(self.profile + '\n' + query + '\n' + d['contextual'])
                scores[k] = d['rerank_score']
                self.db.execute('INSERT OR REPLACE INTO scores VALUES (?,?)', (k, scores[k]))
            self.db.commit()
        indices = sorted(range(len(docs)), key=lambda i: (-scores[keys[i]], i))
        return [docs[i] for i in indices[:8]], (time.perf_counter()-started)*1000, len(missing)


def aggregate(rows):
    return {key: {m: mean(r[key][m] for r in rows) for m in rows[0][key]}
            for key in ('top4', 'top8', 'top24')} if rows else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('split', choices=['dev', 'test', 'regression', 'zh'])
    parser.add_argument('--rerank', action='store_true')
    parser.add_argument('--prepare-only', action='store_true', help='Encode missing inputs without loading the reranker')
    parser.add_argument('--offline', action='store_true', help='Require cached vectors and a previously verified model identity')
    args = parser.parse_args()
    data = json.loads(DATA.read_text(encoding='utf-8'))
    questions = [q for q in data['questions'] if (q['language']=='zh' if args.split=='zh' else q['split']==args.split)]
    tasks = [t for t in data['multi'] if t['split']==args.split]
    cols = {q['collection'] for q in questions} | {c for t in tasks for c in t['collections']}
    docs = [d for d in data['documents'] if d['collection'] in cols]
    health_path = HERE/'cache/embedding_health.json'
    if args.offline:
        health = json.loads(health_path.read_text())
    else:
        health = requests.get('http://127.0.0.1:8010/health', timeout=5).json()
        save(health_path, health)
    assert health['ready'] and health['model']=='bge-m3'
    seal = {'dataset_sha256': hashlib.sha256(DATA.read_bytes()).hexdigest(), 'model': health,
            'field': 'contextual', 'rrf_k': 60, 'candidates': 24, 'final': 8,
            'reranker':reranker_identity(), 'code':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [Path(__file__), ROOT/'backend/rag/embedder.py', ROOT/'backend/rag/bm25.py',
                        ROOT/'backend/rag/lexical.py', ROOT/'backend/rag/evidence_pack.py',
                        ROOT/'experiments/lightrag/metrics.py', ROOT/'deploy/embedding/server.py']}}
    if args.split != 'dev' and not args.prepare_only:
        selection = json.loads((HERE/'selection.json').read_text())
        assert selection['seal'] == seal
    cache = Cache(health)
    texts = [d['contextual'] for d in docs] + [q['question'] for q in questions]
    plans = {t['id']: re.findall(r'^\s*-\s*\[([^\]\n]+)\]\s*(.+)$', t['question'], re.M) for t in tasks}
    texts += [t['question'] for t in tasks] + [q for plan in plans.values() for _, q in plan]
    cost = cache.prepare(texts, offline=args.offline)
    if args.prepare_only:
        print('prepared', args.split, json.dumps(cost), flush=True)
        return
    engine = Engine(docs, cache)
    rows, multi, results = {}, {}, {}
    folder = HERE/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+args.split)
    for arm in ARMS:
        rows[arm] = []
        for q in questions:
            ranked = engine.ranked(q['question'], [q['collection']], arm)
            rows[arm].append({'id': q['id'], 'candidates': [d['id'] for d in ranked[:24]],
                              **{f'top{k}': metric(ranked,q['required'],k) for k in [4,8,24]}})
        results[arm] = aggregate(rows[arm])
        multi[arm] = []
        for t in tasks:
            available = [f for f in t['facets'] if f['answerable']]
            required = list({sid for f in available for sid in f['required']})
            global_ranks = engine.ranked(t['question'], t['collections'], arm)
            per_facet = [engine.ranked(q,[c],arm)[:24] if c in t['collections'] else [] for c,q in plans[t['id']]]
            packed = balanced_evidence(per_facet, all_documents=[d for d in docs if d['collection'] in t['collections']],
                                       token_count=lambda d:text_tokens(context(d)), budget=3000)
            assert all(d['collection'] in t['collections'] for d in packed)
            multi[arm].append({'id':t['id'],'category':t['category'],
                               'global8':metric(global_ranks,required,8),'global24':metric(global_ranks,required,24),
                               'balanced3000':metric(packed,required,max(1,len(packed))),
                               'facet_complete':mean(metric(packed,f['required'],max(1,len(packed)))['all_evidence'] for f in available),
                               'tokens':sum(text_tokens(context(d)) for d in packed),
                               'evidence':[d['id'] for d in packed]})
        print(arm, json.dumps(results[arm]), flush=True)
    if args.split == 'dev':
        assert args.rerank, 'Development selection requires real 24-to-8 reranking'
        winner = max(ARMS[2:], key=lambda a: mean(results[a]['top8'][m] for m in ['recall_at_k','all_evidence','ndcg_at_k']))
        selection = {'winner':winner,'seal':seal,'dev_results':results,
                     'rule':'max mean recall/complete/nDCG@8 on dev100, RRF60 fixed; no test tuning',
                     'created':datetime.now(timezone.utc).isoformat()}
    rerank_rows = {}
    rerank_multi = {}
    if args.rerank:
        rr = Rerank()
        rerank_arms = ['hash_legacy', *ARMS[2:]] if args.split=='dev' else ['hash_legacy', selection['winner']]
        for arm in rerank_arms:
            rerank_rows[arm] = []
            for i,q in enumerate(questions):
                candidates = engine.ranked(q['question'],[q['collection']],arm)[:24]
                ranked, ms, fresh = rr.run(q['question'],candidates)
                rerank_rows[arm].append({'id':q['id'],'top8':metric(ranked,q['required'],8),
                                         'top4':metric(ranked,q['required'],4),'evidence':[d['id'] for d in ranked],
                                         'rerank_ms':ms,'fresh_pairs':fresh})
                if (i+1)%10==0: print('rerank',arm,i+1,len(questions),flush=True)
                save(folder/'rerank-progress.json',rerank_rows)
            rerank_multi[arm] = []
            for t in tasks:
                candidates = engine.ranked(t['question'],t['collections'],arm)[:24]
                ranked, ms, fresh = rr.run(t['question'],candidates)
                available = [f for f in t['facets'] if f['answerable']]
                required = list({sid for f in available for sid in f['required']})
                rerank_multi[arm].append({'id':t['id'],'category':t['category'],
                                          'top8':metric(ranked,required,8),'evidence':[d['id'] for d in ranked],
                                          'facet_complete':mean(metric(ranked,f['required'],8)['all_evidence'] for f in available),
                                          'rerank_ms':ms,'fresh_pairs':fresh})
        if args.split == 'dev':
            selection['winner'] = max(ARMS[2:],key=lambda a:mean(r['top8'][m] for r in rerank_rows[a]
                                                for m in ['recall_at_k','all_evidence','ndcg_at_k']))
            selection['rule'] = 'max mean recall/complete/nDCG after real24-to8 rerank on dev100; RRF60 fixed'
            selection['rerank_results'] = {a:{m:mean(r['top8'][m] for r in rs) for m in rs[0]['top8']}
                                          for a,rs in rerank_rows.items()}
            save(HERE/'selection.json',selection)
    summary_multi = {arm:{group:{k:mean(r[k]['recall_at_k'] for r in subset)
                                     for k in ['global8','global24','balanced3000']}
                          for group,subset in [('all',rs),('reports_available',[r for r in rs if r['category']!='missing_report'])] if subset}
                     for arm,rs in multi.items()}
    report = {'seal':seal,'selection':selection,'split':args.split,'questions':len(questions),'documents':len(docs),
              'embedding_cost':cost,'summary':results,'rows':rows,'multi':multi,'multi_summary':summary_multi,
              'rerank_multi':rerank_multi,
              'rerank':rerank_rows,'rerank_summary':{a:{k:{m:mean(r[k][m] for r in rs) for m in rs[0][k]}
                                                       for k in ['top4','top8']} for a,rs in rerank_rows.items()},
              'limitations':['Previously exposed FinQA test replay; known report scope; QA-bearing pages only.',
                             'Multi tasks constructed from author questions; balanced context is separate from global top8.',
                             'Cached timings are not online latency. No generation accuracy inferred.']}
    save(folder/'report.json',report)
    save(HERE/(args.split+'_latest.json'), {'path':str(folder/'report.json')})
    print('completed',folder,flush=True)


if __name__ == '__main__':
    main()
