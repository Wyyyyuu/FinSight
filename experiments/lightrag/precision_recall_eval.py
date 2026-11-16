"""Paired retrieval evaluation of the restored real-report graph, without LLM calls.

Exact source-ID judgments, fixed K8/3000-token contexts, audited and raw labels.
Queries are executed against Neo4j and NetworkX; only embeddings/keywords are
cached. These are reused development cases, not unseen independent test samples.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.lightrag.neo4j_experiment import (
    HERE, read, save, digest, ready, runtime, close_runtime, index_lock, snapshot, source_files)
from experiments.lightrag.multidoc_pilot import (
    DATA, choose, Engine, facets_from_query, resolve_blocks, fuse_sources, pack, context, text_tokens)
from experiments.lightrag.multidoc_guarded import anchored_candidates
from experiments.lightrag.metrics import retrieval_metrics

MULTI_ARMS = ['dense_global8', 'hybrid_global8', 'hybrid_balanced8',
              'lightrag_naive8', 'lightrag_mix8', 'graph_balanced8', 'guarded_graph8']
FACET_ARMS = ['dense8', 'hybrid8', 'lightrag_mix8', 'graph_fused8', 'guarded_graph8']


def audited_tasks(tasks, overrides):
    result = deepcopy(tasks)
    for correction in overrides:
        for task in result:
            if task['id'] == correction['task']:
                facet = next(f for f in task['facets'] if f['id'] == correction['facet'])
                facet['required'] = sorted(set(facet['required'] + correction.get('additional_required', [])))
    return result


def gold_ids(task):
    return sorted({s for f in task['facets'] if f['answerable'] for s in f['required']})


def score_row(*, group, case, category, arm, sources, gold, raw_gold, allowed):
    ids = [d['id'] for d in sources]
    if len(ids) != len(set(ids)) or len(ids) > 8:
        raise ValueError('Invalid final evidence budget')
    tokens = sum(text_tokens(context(d)) for d in sources)
    if tokens > 3000 or any(d['collection'] not in allowed for d in sources):
        raise ValueError('Evidence exceeds token budget or requested report scope')
    return dict(group=group, case=case, category=category, arm=arm, sources=ids,
                required=gold, raw_required=raw_gold, hits=sorted(set(ids)&set(gold)),
                returned_precision=len(set(ids)&set(gold))/len(ids) if ids else None,
                returned=len(ids), context_tokens=tokens, collections=sorted(allowed),
                metrics=retrieval_metrics(ids, gold, 8),
                raw_metrics=retrieval_metrics(ids, raw_gold, 8),
                precision_ceiling_at_8=min(8, len(set(gold)))/8)


def aggregate(rows):
    result = {}
    for group in ['multidoc', 'facet']:
        result[group] = {}
        for arm in sorted({r['arm'] for r in rows if r['group']==group}):
            subset = [r for r in rows if r['group']==group and r['arm']==arm]
            summary = dict(cases=len(subset), mean_returned=mean(r['returned'] for r in subset),
                           precision_ceiling_at_8=mean(r['precision_ceiling_at_8'] for r in subset))
            measured = [r['returned_precision'] for r in subset if r['returned_precision'] is not None]
            summary['returned_precision'] = mean(measured) if measured else None
            summary['returned_precision_measured'] = len(measured)
            for labels, field in [('audited','metrics'), ('original','raw_metrics')]:
                summary[labels] = {name: mean(r[field][name] for r in subset)
                                  for name in ['precision_at_k','recall_at_k','ndcg_at_k','all_evidence']}
            result[group][arm] = summary
    return result


async def evaluate_backend(target, manifest, tasks, corrected, docs, networkx_backend):
    graph, cache, llm = runtime(target, manifest, networkx=networkx_backend)
    plan = read(target/'plan.json')
    blocks = read(target/'blocks.json')
    lookup = {d['id']: d for d in docs}
    engine = Engine(docs, cache)
    rows, traces = [], []
    memo = {}

    async def retrieve(query, mode):
        key = (query, mode)
        if key not in memo:
            value = await graph.retrieve(query, 24, mode)
            if value['unknown_sources']:
                raise ValueError('Unknown graph source')
            memo[key] = value
            traces.append(dict(query=query, mode=mode, sources=value['sources'],
                               entities=len(value['graph'].get('entities', [])),
                               relationships=len(value['graph'].get('relationships', []))))
        return memo[key]['sources']

    def resolve(block_ids, query):
        ranking = engine.ranked(query, plan['collections'], 'm3_bm25_rrf')
        return resolve_blocks(block_ids, blocks, ranking)

    try:
        await graph.initialize()
        for task, audited in zip(tasks, corrected):
            per_facet, fused_rankings, guarded_rankings = [], [], []
            request_facets = facets_from_query(task['question'])
            if len(request_facets) != len(task['facets']):
                raise ValueError('Question/facet alignment changed')
            for i, (collection, query) in enumerate(request_facets):
                if (collection, query) != (task['facets'][i]['collection'], task['facets'][i]['question']):
                    raise ValueError('Question/facet order changed')
                base = engine.ranked(query, [collection], 'm3_bm25_rrf')[:24]
                block_ids = await retrieve(f'Report: {collection}\n{query}', 'mix')
                resolved = resolve(block_ids, query)
                fused = fuse_sources(base, resolved, docs, collection)
                guarded = anchored_candidates(base, fused)
                per_facet.append(base)
                fused_rankings.append(fused)
                guarded_rankings.append(guarded)
                candidates = {
                    'dense8': engine.ranked(query, [collection], 'm3_dense')[:24],
                    'hybrid8': base,
                    'lightrag_mix8': [lookup[s] for s in resolved if lookup[s]['collection']==collection],
                    'graph_fused8': fused, 'guarded_graph8': guarded,
                }
                for arm in FACET_ARMS:
                    sources = pack([candidates[arm]], docs)
                    rows.append(score_row(group='facet', case=task['id']+':'+task['facets'][i]['id'],
                        category=task['category'], arm=arm, sources=sources,
                        gold=audited['facets'][i]['required'], raw_gold=task['facets'][i]['required'],
                        allowed={collection}))
            naive = resolve(await retrieve(task['question'], 'naive'), task['question'])
            mix = resolve(await retrieve(task['question'], 'mix'), task['question'])
            selected = {
                'dense_global8': pack([engine.ranked(task['question'], task['collections'], 'm3_dense')[:24]], docs),
                'hybrid_global8': pack([engine.ranked(task['question'], task['collections'], 'm3_bm25_rrf')[:24]], docs),
                'hybrid_balanced8': pack(per_facet, docs, True),
                'lightrag_naive8': pack([[lookup[s] for s in naive]], docs),
                'lightrag_mix8': pack([[lookup[s] for s in mix]], docs),
                'graph_balanced8': pack(fused_rankings, docs, True),
                'guarded_graph8': pack(guarded_rankings, docs, True),
            }
            for arm in MULTI_ARMS:
                rows.append(score_row(group='multidoc', case=task['id'], category=task['category'],
                    arm=arm, sources=selected[arm], gold=gold_ids(audited), raw_gold=gold_ids(task),
                    allowed=set(task['collections'])))
        return dict(rows=rows, traces=traces, queries=len(memo), summary=aggregate(rows))
    finally:
        await close_runtime(graph, cache, llm)


async def evaluate(target):
    manifest = ready(target)
    plan = read(target/'plan.json')
    if digest(DATA) != plan['dataset_sha256']:
        raise ValueError('Frozen corpus changed')
    tasks, docs = choose(read(DATA))
    corrected = audited_tasks(tasks, read(HERE/'multidoc_label_audit.json')['overrides'])
    if not set(s for t in corrected for s in gold_ids(t)) <= {d['id'] for d in docs}:
        raise ValueError('Gold source outside indexed corpus')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = target/('precision-recall-'+stamp)
    output.mkdir()
    prereg = dict(status='registered before retrieval', k=8, context_tokens=3000, reranker=False,
        dataset_sha256=digest(DATA), label_audit_sha256=digest(HERE/'multidoc_label_audit.json'),
        code_sha256=digest(Path(__file__)), adapter_sha256=digest(HERE/'adapters.py'),
        index_source_hashes=manifest['source_hashes'], embedding=plan['model'],
        tasks=[t['id'] for t in tasks], task_count=len(tasks), facets=sum(len(t['facets']) for t in tasks),
        unique_facets=len({(f['collection'],f['question']) for t in tasks for f in t['facets']}),
        reports=len(plan['collections']), source_passages=len(docs), graph_blocks=len(read(target/'blocks.json')),
        multi_arms=MULTI_ARMS, facet_arms=FACET_ARMS,
        metrics='Macro mean: Precision@8=distinct labeled hits/8; Recall@8=distinct labeled hits/total gold IDs. Empty slots count as nonhits. Exact-ID relevance only.',
        limitation='Reused English FinQA development cluster: 3 constructed multi-report tasks, 7 correlated subquestions. Not a fresh holdout, not Chinese, not an answer correctness evaluation. Annotations are not exhaustive; useful unlabelled/equivalent passages receive no credit.',
        network='Actual local Neo4j traversal; frozen embedding/keyword caches. New external model/embedding calls forbidden.')
    save(output/'plan.json', prereg)
    neo4j = await evaluate_backend(target, manifest, tasks, corrected, docs, False)
    save(output/'neo4j.json', neo4j)
    networkx_result = await evaluate_backend(target, manifest, tasks, corrected, docs, True)
    save(output/'networkx.json', networkx_result)
    if neo4j['rows'] != networkx_result['rows']:
        raise ValueError('Graph backend comparison differs; reports saved for investigation')
    if source_files(Path(manifest['source'])) != manifest['source_hashes']:
        raise ValueError('Original index changed')
    nodes, edges = await snapshot(manifest['workspace'])
    result = dict(status='complete', plan=prereg, output=str(output), summary=neo4j['summary'],
                  parity_rows=len(neo4j['rows']), neo4j_nodes=len(nodes), neo4j_edges=len(edges),
                  queries_per_backend=neo4j['queries'], new_llm_calls=0, new_embedding_calls=0, errors=0)
    save(output/'report.json', result)
    save(target/'precision_recall_latest.json', {'path': str(output/'report.json')})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, default=HERE/'runs'/'neo4j-multidoc')
    args = parser.parse_args()
    with index_lock(args.target.resolve()):
        asyncio.run(evaluate(args.target.resolve()))
