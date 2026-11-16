"""Isolated real-report LightRAG pilot; production services/configs are untouched.

Select the smallest existing dev cluster by corpus size, before scoring. All arms
see the same report-prefixed original passages, BGE-M3 and final context budget.
No reranker: naive vs mix isolates graph retrieval; balanced vs graph-balanced
isolates graph-assisted candidates. These constructed tasks are not true hops.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sqlite3
from statistics import mean
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import requests
from backend.rag.embedder import EmbeddingService
from backend.rag.evidence_pack import balanced_evidence, pack_evidence
from experiments.docker_embedding.evaluate import DATA, Engine, context, text_tokens, sha
from experiments.financial_retrieval_v2.answer_audit import score_facet
from experiments.lightrag.adapters import ChatClient, CallBudget, LightRAGAdapter, endpoint_config
from experiments.lightrag.metrics import retrieval_metrics

HERE = Path(__file__).parent
ARMS = ['m3_global8', 'm3_balanced8', 'lightrag_naive8', 'lightrag_mix8', 'graph_balanced8']


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def choose(data):
    clusters = defaultdict(list)
    for task in data['multi']:
        if task['split'] == 'dev' and task['category'] != 'missing_report':
            clusters[tuple(sorted(task['collections']))].append(task)
    scopes = min(clusters, key=lambda cols: (sum(d['collection'] in cols for d in data['documents']), cols))
    tasks = clusters[scopes]
    assert {t['category'] for t in tasks} == {'cross_company', 'cross_year', 'three_reports'}
    docs = [{**d, 'contextual': f'Report: {d["collection"]}\n{d["contextual"]}'}
            for d in data['documents'] if d['collection'] in scopes]
    return tasks, docs


def facets_from_query(query):
    # No answer, required IDs or answerable labels influence retrieval.
    return re.findall(r'^\s*-\s*\[([^\]\n]+)\]\s*(.+)$', query, re.M)


def graph_blocks(docs, max_tokens=480):
    """Pack adjacent source rows within one report/page, retaining a full map."""
    pages = defaultdict(list)
    for d in docs:
        page = d.get('table_id') or (d.get('locations') or [{}])[0].get('page') or d['id']
        pages[(d['collection'],page)].append(d)
    blocks=[]
    for (collection,page), members in sorted(pages.items()):
        pending=[]
        def emit():
            ids=[d['id'] for d in pending]
            blocks.append(dict(id='B'+sha(collection+'\n'+page+'\n'+'\n'.join(ids))[:20],
                               text='\n'.join(f'[{d["id"]}] {d["contextual"]}' for d in pending),
                               source_ids=ids, collection=collection, page=page))
        for d in members:
            candidate=pending+[d]
            size=text_tokens('\n'.join(f'[{x["id"]}] {x["contextual"]}' for x in candidate))
            if pending and size>max_tokens:
                emit()
                pending=[]
            pending.append(d)
        if pending:
            emit()
    assert max(text_tokens(b['text']) for b in blocks)<=max_tokens
    assert sorted(s for b in blocks for s in b['source_ids'])==sorted(d['id'] for d in docs)
    return blocks


def resolve_blocks(block_ids, blocks, original_ranking, limit=24):
    """Use block order then a shared row ranking; never return block summaries."""
    lookup={b['id']:b for b in blocks}
    positions={d['id']:i for i,d in enumerate(original_ranking)}
    result=[]
    for bid in block_ids:
        if bid not in lookup:
            raise ValueError('Unmapped graph block')
        for sid in sorted(lookup[bid]['source_ids'],key=lambda sid:positions.get(sid,float('inf'))):
            if sid in positions and sid not in result:
                result.append(sid)
    return result[:limit]


def fuse_sources(base, graph_ids, docs, collection):
    """Equal-weight rank fusion, with original-source and report boundaries."""
    lookup = {d['id']: d for d in docs}
    scores = defaultdict(float)
    for ids in ([d['id'] for d in base], graph_ids):
        for rank, sid in enumerate(dict.fromkeys(ids), 1):
            if sid in lookup and lookup[sid]['collection'] == collection:
                scores[sid] += 1 / (60 + rank)
    return [lookup[sid] for sid in sorted(scores, key=lambda sid: (-scores[sid], sid))][:24]


def pack(rankings, docs, balanced=False, limit=8):
    if balanced:
        result = balanced_evidence(rankings, all_documents=docs,
                                   token_count=lambda d: text_tokens(context(d)), budget=3000)
    else:
        result = pack_evidence(rankings[0], all_documents=docs,
                               token_count=lambda d: text_tokens(context(d)), budget=3000)
    return result if limit is None else result[:limit]


class EmbeddingCache:
    model_name, dim = 'bge-m3', 1024

    def __init__(self, folder, health):
        self.profile = f'{health["model"]}:{health["dim"]}:{health["revision"]}'
        self.db = sqlite3.connect(folder/'vectors.sqlite', check_same_thread=False)
        self.db.execute('CREATE TABLE IF NOT EXISTS vectors (id TEXT PRIMARY KEY, dense BLOB, sparse TEXT)')
        self.lock = threading.Lock()
        self.encoder = EmbeddingService(force_backend='bge-m3-http')
        self.fresh_texts = 0

    def key(self, text):
        return sha(self.profile + '\n' + text)

    def prepare(self, texts):
        with self.lock:
            missing = [t for t in dict.fromkeys(texts) if not self.db.execute(
                'SELECT 1 FROM vectors WHERE id=?', (self.key(t),)).fetchone()]
            for start in range(0, len(missing), 16):
                batch = missing[start:start+16]
                encoded = self.encoder.encode(batch)
                assert encoded.profile == self.profile
                for text, dense, sparse in zip(batch, encoded.dense, encoded.sparse):
                    self.db.execute('INSERT INTO vectors VALUES (?,?,?)',
                                    (self.key(text), np.asarray(dense, dtype=np.float32).tobytes(), json.dumps(sparse.weights)))
                self.db.commit()
                self.fresh_texts += len(batch)

    def get(self, text):
        with self.lock:
            row = self.db.execute('SELECT dense,sparse FROM vectors WHERE id=?', (self.key(text),)).fetchone()
            if row is None:
                raise ValueError('Unprepared input')
            return np.frombuffer(row[0], dtype=np.float32), json.loads(row[1])

    async def dense_async(self, texts, **kwargs):
        values = list(texts)
        await asyncio.to_thread(self.prepare, values)
        return np.asarray([self.get(t)[0] for t in values], dtype=np.float32)


def summary(rows):
    result = {}
    for arm in ARMS:
        current = [r for r in rows if r['arm'] == arm and 'error' not in r]
        facts = [f for r in current for f in r.get('facets', [])]
        numeric = [f for f in facts if f['numeric_correct'] is not None]
        result[arm] = dict(cases=len(current), errors=sum('error' in r for r in rows if r['arm']==arm),
                          recall=mean(r['metrics']['recall_at_k'] for r in current) if current else None,
                          complete=mean(r['metrics']['all_evidence'] for r in current) if current else None,
                          context_tokens=mean(r['context_tokens'] for r in current) if current else None,
                          numeric_correct=sum(f['numeric_correct'] for f in numeric), numeric_scored=len(numeric),
                          numeric_and_evidence=sum(f['numeric_and_evidence_correct'] for f in numeric))
    return result


def successful(rows, task_id, arm):
    return any(r['id']==task_id and r['arm']==arm and 'error' not in r for r in rows)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    data = json.loads(DATA.read_text(encoding='utf-8'))
    tasks, docs = choose(data)
    blocks=graph_blocks(docs)
    plan = dict(selection='smallest dev report-scope cluster by passage count; no outcome selection',
                sample_ids=[t['id'] for t in tasks], documents=len(docs),
                collections=sorted({d['collection'] for d in docs}), arms=ARMS,
                field='Report: <collection> newline contextual, shared by all arms',
                dataset_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
                source_sha256=sha(json.dumps([(d['id'],d['contextual']) for d in docs])),
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                dependencies={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [HERE/'adapters.py', ROOT/'backend/rag/embedder.py', ROOT/'backend/rag/evidence_pack.py',
                     ROOT/'backend/rag/bm25.py',ROOT/'experiments/financial_retrieval_v2/answer_audit.py',
                     HERE/'metrics.py',ROOT/'experiments/docker_embedding/evaluate.py',HERE/'multidoc_label_audit.json']},
                tiktoken=importlib.metadata.version('tiktoken'),
                lightrag=importlib.metadata.version('lightrag-hku'), final_k=8, context_budget=3000,
                graph_blocks=len(blocks), graph_block_tokens=480,
                block_sha256=sha(json.dumps(blocks)),
                indexing='LightRAG groups original source-labelled rows within a report/page; naive and mix share this index. Ordinary hybrid retains original-row vectors. Block candidates resolve to original rows using the same BM25+M3 ordering within each block.',
                reranker=False, llm_call_budget=650,
                limitations=['3 constructed dev tasks, not an unseen holdout or true multi-hop benchmark.',
                             'Report scope includes distractors but not the whole KB. Missing-report tasks excluded.',
                             'All final contexts contain original sources only; graph summaries are not evidence.',
                             'No rerank in this pilot; production reranker stays enabled.',
                             'Cached embeddings make retrieval time unsuitable for online latency comparison.',
                             'Quote matching and numeric scoring do not establish full narrative entailment.'])
    if not args.live:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    folder = args.resume or HERE/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-multidoc')
    folder.mkdir(parents=True, exist_ok=True)
    health = requests.get('http://127.0.0.1:8010/health', timeout=5).json()
    assert health['ready'] and health['device']=='cpu' and health['dim']==1024
    plan['model'] = health
    config=endpoint_config()
    plan['llm']={'model':config['model'],'endpoint_url_sha256':sha(config['url'])}
    if args.resume:
        assert json.loads((folder/'plan.json').read_text(encoding='utf-8')) == plan, 'Resume identity changed'
    save(folder/'plan.json', plan)
    (folder/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    prereg=dict(answer_rows_at_sealing=0,phase='graph_build',
                plan_sha256=hashlib.sha256((folder/'plan.json').read_bytes()).hexdigest(),
                files=plan['dependencies'],tiktoken=plan['tiktoken'],
                llm_model=config['model'],endpoint_url_sha256=sha(config['url']))
    if not args.resume:
        save(folder/'scoring_preregistration.json',prereg)
    save(folder/'blocks.json', blocks)
    save(HERE/'multidoc_latest.json', {'path':str(folder/'report.json')})
    cache = EmbeddingCache(folder, health)
    previous = json.loads((folder/'report.json').read_text(encoding='utf-8')) if args.resume else {}
    old_calls = previous.get('calls', [])
    client = ChatClient(config, CallBudget(650-len(old_calls)), timeout=90)
    # Extraction may include many entities in a 512-token table; allow complete tuples.
    original_complete = client.complete
    async def complete(prompt, **kwargs):
        kwargs.setdefault('max_tokens', 6000 if client.phase=='graph_build' else 2400)
        try:
            return await original_complete(prompt, **kwargs)
        finally:
            checkpoint()
    client.complete = complete
    graph = LightRAGAdapter(folder/'index', cache, client,
                           blocks, context_tokens=3000)
    graph.rag.addon_params['language'] = 'English'
    rows = previous.get('rows', [])
    result = dict(plan=plan, rows=rows, calls=old_calls, status='running', build=previous.get('build'),
                  failed_attempts=previous.get('failed_attempts',[]))
    def checkpoint():
        result.update(calls=old_calls+client.calls, summary=summary(rows))
        save(folder/'report.json', result)
    try:
        client.phase = 'graph_build'
        await graph.initialize()
        if not result['build']:
            started = time.perf_counter()
            await asyncio.wait_for(graph.insert(blocks), timeout=3600)
            result['build'] = {'seconds':time.perf_counter()-started,
                               'calls':len(client.calls), 'status':await graph.rag.get_processing_status()}
            checkpoint()
        from lightrag.base import DocStatus
        processed=await graph.rag.doc_status.get_docs_by_status(DocStatus.PROCESSED)
        assert set(processed)=={b['id'] for b in blocks}, 'Graph sources incomplete or foreign'
        print('GRAPH_READY', flush=True)
        texts = [d['contextual'] for d in docs]+[t['question'] for t in tasks]
        texts += [q for t in tasks for _,q in facets_from_query(t['question'])]
        await asyncio.to_thread(cache.prepare, texts)
        engine = Engine(docs, cache)
        lookup = {d['id']:d for d in docs}
        async def retrieve(query, mode):
            client.phase = 'graph_query'
            value = await graph.retrieve(query, 24, mode)
            assert value['unknown_sources']==0, 'Unmapped source cannot enter evidence'
            return value
        def resolve(value, query):
            ranking=engine.ranked(query, plan['collections'], 'm3_bm25_rrf')
            return resolve_blocks(value['sources'], blocks, ranking)
        for task in tasks:
            client.phase = 'retrieval'
            base = engine.ranked(task['question'], task['collections'], 'm3_bm25_rrf')[:24]
            per_facet, assisted, traces = [], [], []
            for collection, query in facets_from_query(task['question']):
                rank = engine.ranked(query, [collection], 'm3_bm25_rrf')[:24]
                value = await retrieve(f'Report: {collection}\n{query}', 'mix')
                per_facet.append(rank)
                row_ids=resolve(value,query)
                assisted.append(fuse_sources(rank, row_ids, docs, collection))
                value['resolved_sources']=row_ids
                traces.append(dict(collection=collection, query=query, retrieval=value))
            naive = await retrieve(task['question'], 'naive')
            mix = await retrieve(task['question'], 'mix')
            naive['resolved_sources']=resolve(naive,task['question'])
            mix['resolved_sources']=resolve(mix,task['question'])
            save(folder/(task['category']+'-retrieval.json'), dict(facets=traces, naive=naive, mix=mix))
            packed = {'m3_global8':pack([base],docs), 'm3_balanced8':pack(per_facet,docs,True),
                      'lightrag_naive8':pack([[lookup[s] for s in naive['resolved_sources']]],docs),
                      'lightrag_mix8':pack([[lookup[s] for s in mix['resolved_sources']]],docs),
                      'graph_balanced8':pack(assisted,docs,True)}
            required = sorted({s for f in task['facets'] if f['answerable'] for s in f['required']})
            for arm, sources in packed.items():
                if successful(rows, task['id'], arm):
                    continue
                failed=[r for r in rows if r['id']==task['id'] and r['arm']==arm]
                result['failed_attempts'].extend(failed)
                rows[:]=[r for r in rows if not (r['id']==task['id'] and r['arm']==arm)]
                supplied = {d['id']:d for d in sources}
                row = dict(id=task['id'], arm=arm, supplied=list(supplied),
                           context_tokens=sum(text_tokens(context(d)) for d in sources),
                           metrics=retrieval_metrics(list(supplied),required,8))
                assert len(sources)<=8 and row['context_tokens']<=3000
                request = [{k:f[k] for k in ['id','question','collection']} for f in task['facets']]
                prompt = ('Use ONLY supplied evidence to answer every requested item. Return JSON with summary and facts. '
                          'Each fact must contain facet_id, report (exact requested collection), answer (number, or null if '
                          'insufficient evidence), unit, citations (exact source IDs), quotes (objects with source_id and '
                          'text copied literally from that source), calculation. Percent answers use 12 for 12%, not 0.12. '
                          'Keep years, signs, units and restatement bases separate. Summary must be supported by facts.\n'
                          'Request:\n'+task['question']+'\nRequested facets:\n'+json.dumps(request)+
                          '\nEvidence:\n'+''.join(context(d) for d in sources))
                client.phase = 'answer'
                try:
                    output = json.loads(await client.complete(prompt, response_format={'type':'json_object'}))
                    facts = {str(f['facet_id']):f for f in output['facts']}
                    row.update(output=output,facets=[score_facet(f,facts.get(f['id'],{}),supplied) for f in task['facets']])
                except Exception as exc:
                    row['error'] = type(exc).__name__
                rows.append(row)
                checkpoint()
                print('ANSWER', len(rows), len(tasks)*len(ARMS), flush=True)
        result['status'] = 'partial' if any('error' in r for r in rows) else 'complete'
    except BaseException as exc:
        result.update(status='partial',error=type(exc).__name__)
        raise
    finally:
        checkpoint()
        await client.close()
        await graph.close()
        cache.db.close()
    print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
