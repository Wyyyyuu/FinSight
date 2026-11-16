"""Restore the frozen multi-document LightRAG index into isolated Neo4j storage.

No extraction or embedding calls during migration. Original run files are never
opened for writing. `verify` replays cached queries without any external model.
`query` permits bounded keyword extraction and returns traceable raw evidence.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).parent
from experiments.lightrag.neo4j_config import configure, provision

# Frozen LightRAG 1.5.7 NetworkX + NanoVectorDB + JsonKV storage contract.
REQUIRED_STORES = {
    'graph_chunk_entity_relation.graphml', 'kv_store_doc_status.json',
    'kv_store_entity_chunks.json', 'kv_store_full_docs.json',
    'kv_store_full_entities.json', 'kv_store_full_relations.json',
    'kv_store_llm_response_cache.json', 'kv_store_relation_chunks.json',
    'kv_store_text_chunks.json', 'vdb_chunks.json', 'vdb_entities.json',
    'vdb_relationships.json',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def index_lock(target):
    """Do not allow a migration to overwrite stores while another CLI queries."""
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_name(target.name+'.lock')
    try:
        handle = lock.open('x', encoding='utf-8')
    except FileExistsError:
        raise RuntimeError('Index is locked. If a prior process crashed, confirm it has exited before removing '+str(lock)) from None
    try:
        with handle:
            handle.write(str(os.getpid()))
            handle.flush()
            yield
    finally:
        lock.unlink()


def source_files(source):
    workspace = source/'index'/'finsight_experiment'
    missing = sorted(name for name in REQUIRED_STORES if not (workspace/name).is_file())
    if missing:
        raise ValueError('Incomplete source index; missing: '+', '.join(missing))
    files = list(workspace.glob('*'))
    files += [source/name for name in ['blocks.json', 'plan.json', 'vectors.sqlite']]
    if not files or any(not p.is_file() for p in files):
        raise ValueError('Incomplete source run')
    return {str(p.relative_to(source)): digest(p) for p in sorted(files)}


def prepare(source, target):
    source, target = source.resolve(), target.resolve()
    if target == source or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError('Source and target must be separate directories')
    if read(source/'report.json')['status'] != 'complete':
        raise ValueError('Only a complete frozen run can be restored')
    files = source_files(source)
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if (target/'manifest.json').exists():
        manifest = read(target/'manifest.json')
        if manifest['source_hashes'] != files or manifest['source'] != str(source):
            raise ValueError('Source changed since migration; use a new target')
        return manifest
    if target.exists():
        raise ValueError('Target exists without a manifest; choose a new target')
    workspace = 'finsight_lightrag_' + hashlib.sha256(
        (identity + str(target).casefold()).encode()).hexdigest()[:20]
    target.mkdir(parents=True)
    manifest = dict(status='preparing', source=str(source), source_hashes=files,
                    workspace=workspace, graph_storage='Neo4JStorage',
                    llm_model=read(source/'report.json')['calls'][0]['model'])
    # The pending manifest makes an interrupted copy detectable and resumable.
    save(target/'manifest.json', manifest)
    return manifest


def copy_stores(source, target, workspace):
    dest = target/'index'/workspace
    dest.mkdir(parents=True, exist_ok=True)
    for path in (source/'index'/'finsight_experiment').iterdir():
        if path.is_file():
            shutil.copy2(path, dest/path.name)
    for name in ['blocks.json', 'plan.json', 'vectors.sqlite']:
        shutil.copy2(source/name, target/name)
    for name, expected in source_files(source).items():
        old_path = Path(name)
        copied = dest/old_path.name if old_path.parts[0] == 'index' else target/old_path.name
        if digest(copied) != expected:
            raise ValueError('Copied store checksum mismatch')
    return dest


class OfflineLLM:
    timeout = 90

    def __init__(self, model):
        self.config = {'model': model}

    async def complete(self, *args, **kwargs):
        raise RuntimeError('Cache miss: model calls are forbidden during migration/verification')


def runtime(target, manifest, *, live=False, networkx=False):
    from experiments.lightrag.adapters import ChatClient, CallBudget, LightRAGAdapter, endpoint_config
    from experiments.lightrag.multidoc_pilot import EmbeddingCache
    plan = read(target/'plan.json')
    cache = EmbeddingCache(target, plan['model'])
    if not live:
        def cached_only(texts):
            for text in texts:
                cache.get(text)  # Fail before any HTTP call on a cache miss.
        cache.prepare = cached_only
    llm = (ChatClient(endpoint_config(), CallBudget(12), timeout=60) if live
           else OfflineLLM(manifest['llm_model']))
    workspace = manifest['workspace']
    workdir = target/'index'
    if networkx:
        workspace += '_comparison'
        workdir = target/'comparison'
        # Separate labels and workdir prevent LightRAG's process-shared KV state
        # from making the two backend checks inadvertently read the same store.
        dest = workdir/workspace
        if not dest.exists():
            shutil.copytree(target/'index'/manifest['workspace'], dest)
    graph = LightRAGAdapter(workdir, cache, llm, read(target/'blocks.json'), 3000,
                            graph_storage='NetworkXStorage' if networkx else 'Neo4JStorage',
                            workspace=workspace, stable_graph_order=True)
    graph.rag.addon_params['language'] = 'English'
    return graph, cache, llm


async def close_runtime(graph, cache, llm):
    await graph.close()
    cache.db.close()
    if hasattr(llm, 'close'):
        await llm.close()


async def snapshot(workspace):
    from neo4j import AsyncGraphDatabase
    config = configure(workspace)
    async with AsyncGraphDatabase.driver(config['NEO4J_URI'],
            auth=(config['NEO4J_USERNAME'], config['NEO4J_PASSWORD'])) as driver:
        async with driver.session(database=config['NEO4J_DATABASE']) as session:
            rows = await (await session.run(
                f'MATCH (n:`{workspace}`) RETURN n.entity_id AS id, properties(n) AS props')).data()
            nodes = {r['id']: r['props'] for r in rows}
            if len(rows) != len(nodes):
                raise ValueError('Duplicate graph entity IDs')
            rows = await (await session.run(
                f'MATCH (a:`{workspace}`)-[r]->(b:`{workspace}`) '
                'RETURN a.entity_id AS a, b.entity_id AS b, properties(r) AS props')).data()
            edges = {tuple(sorted((r['a'], r['b']))): r['props'] for r in rows}
            if len(rows) != len(edges):
                raise ValueError('Duplicate graph edges')
            return nodes, edges


def validate_graph(nx_graph, nodes, edges, chunks):
    expected_nodes = {key: dict(props) for key, props in nx_graph.nodes(data=True)}
    expected_edges = {tuple(sorted((a, b))): dict(props) for a, b, props in nx_graph.edges(data=True)}
    if nodes != expected_nodes or edges != expected_edges:
        raise ValueError('Neo4j graph does not exactly match the frozen GraphML properties')
    for props in [*nodes.values(), *edges.values()]:
        ids = props.get('source_id', '').split('<SEP>')
        if not ids or any(s not in chunks for s in ids):
            raise ValueError('Graph contains an unresolved original chunk reference')
    return dict(nodes=len(nodes), edges=len(edges), all_properties_equal=True,
                all_graph_sources_resolve=True)


async def migrate(source, target):
    import networkx as nx
    manifest = prepare(source, target)
    if manifest['status'] != 'ready':
        copy_stores(source, target, manifest['workspace'])
    graph, cache, llm = runtime(target, manifest)
    started = time.perf_counter()
    try:
        await graph.initialize()
        store = graph.rag.chunk_entity_relation_graph
        original = nx.read_graphml(source/'index'/'finsight_experiment'/'graph_chunk_entity_relation.graphml')
        nodes = list(original.nodes(data=True))
        edges = list(original.edges(data=True))
        for i in range(0, len(nodes), 100):
            await store.upsert_nodes_batch(nodes[i:i+100])
        for i in range(0, len(edges), 100):
            await store.upsert_edges_batch(edges[i:i+100])
        actual = await snapshot(manifest['workspace'])
        chunks = read(target/'index'/manifest['workspace']/'kv_store_text_chunks.json')
        checks = validate_graph(original, *actual, chunks)
        if source_files(source) != manifest['source_hashes']:
            raise ValueError('Original source files changed during migration')
        manifest.update(status='ready', checks=checks, migration_seconds=time.perf_counter()-started,
                        extraction_calls=0, embedding_calls=0)
        save(target/'manifest.json', manifest)
        print(json.dumps(checks))
    finally:
        await close_runtime(graph, cache, llm)


def ready(target):
    manifest = read(target/'manifest.json')
    if manifest['status'] != 'ready':
        raise ValueError('Migration is incomplete; run migrate again')
    return manifest


async def verify(target):
    import networkx as nx
    manifest = ready(target)
    source = Path(manifest['source'])
    original = nx.read_graphml(source/'index'/'finsight_experiment'/'graph_chunk_entity_relation.graphml')
    checks = validate_graph(original, *(await snapshot(manifest['workspace'])),
                            read(target/'index'/manifest['workspace']/'kv_store_text_chunks.json'))
    # Frozen real report queries: cross-company, cross-year and three-report summary.
    queries = []
    for category in ['cross_company', 'cross_year', 'three_reports']:
        trace = read(source/(category+'-retrieval.json'))
        queries += [f'Report: {f["collection"]}\n{f["query"]}' for f in trace['facets']]
    results = []
    graph_results = []
    guarded_results = []
    for networkx_backend in [True, False]:
        graph, cache, llm = runtime(target, manifest, networkx=networkx_backend)
        current = []
        graph_current = []
        try:
            await graph.initialize()
            for query in queries:
                result = await graph.retrieve(query, 24, 'mix')
                if result['unknown_sources'] or not result['sources']:
                    raise ValueError('Missing or foreign evidence in retrieval')
                current.append(result['sources'])
                graph_current.append({
                    'entities': sorted(e['entity_name'] for e in result['graph']['entities']),
                    'relationships': sorted(tuple(sorted((e['src_id'], e['tgt_id'])))
                                            for e in result['graph']['relationships'])})
            guarded_results.append(guarded_contexts(target, cache, dict(zip(queries, current))))
        finally:
            await close_runtime(graph, cache, llm)
        results.append(current)
        graph_results.append(graph_current)
    ordered_equal = sum(a == b for a, b in zip(*results))
    sets_equal = sum(set(a) == set(b) for a, b in zip(*results))
    checks.update(queries=len(queries), same_order=ordered_equal, same_source_sets=sets_equal,
                  model_calls=0, original_unchanged=source_files(source)==manifest['source_hashes'],
                  same_graph_context=sum(a==b for a,b in zip(*graph_results)),
                  multidoc_cases=len(guarded_results[0]),
                  same_final_evidence=sum(a==b for a,b in zip(*guarded_results)))
    save(target/'verification.json', dict(checks=checks, networkx=results[0], neo4j=results[1],
                                         guarded_networkx=guarded_results[0], guarded_neo4j=guarded_results[1],
                                         graph_networkx=graph_results[0], graph_neo4j=graph_results[1]))
    if (sets_equal != len(queries) or not checks['original_unchanged']
            or checks['same_graph_context'] != len(queries)
            or checks['same_final_evidence'] != checks['multidoc_cases']):
        raise ValueError('Retrieval parity failed; see verification.json')
    print(json.dumps(checks))


def guarded_contexts(target, cache, retrieved):
    from experiments.lightrag.multidoc_pilot import (
        DATA, choose, Engine, facets_from_query, resolve_blocks, fuse_sources, pack)
    from experiments.lightrag.multidoc_guarded import anchored_candidates
    plan = read(target/'plan.json')
    if digest(DATA) != plan['dataset_sha256']:
        raise ValueError('Evaluation source corpus changed')
    tasks, docs = choose(read(DATA))
    engine = Engine(docs, cache)
    contexts = []
    for task in tasks:
        rankings = []
        for collection, query in facets_from_query(task['question']):
            base = engine.ranked(query, [collection], 'm3_bm25_rrf')[:24]
            ranking = engine.ranked(query, plan['collections'], 'm3_bm25_rrf')
            ids = retrieved[f'Report: {collection}\n{query}']
            resolved = resolve_blocks(ids, read(target/'blocks.json'), ranking)
            rankings.append(anchored_candidates(base, fuse_sources(base, resolved, docs, collection)))
        contexts.append([d['id'] for d in pack(rankings, docs, True)])
    return contexts


async def query_index(target, query, collections, *, offline=False):
    from experiments.lightrag.multidoc_pilot import (
        DATA, choose, Engine, facets_from_query, resolve_blocks, fuse_sources, pack, context, text_tokens)
    from experiments.lightrag.multidoc_guarded import anchored_candidates
    manifest = ready(target)
    plan = read(target/'plan.json')
    if digest(DATA) != plan['dataset_sha256']:
        raise ValueError('Dataset changed; frozen source mapping must not be reused')
    _, docs = choose(read(DATA))
    allowed = set(collections or plan['collections'])
    if not allowed or not allowed <= set(plan['collections']):
        raise ValueError('Requested report is not in this index')
    facets = facets_from_query(query) or [(c, query) for c in sorted(allowed)]
    if any(c not in allowed for c, _ in facets) or len(facets) > 12:
        raise ValueError('Invalid report scope or too many subquestions (maximum 12)')
    graph, cache, llm = runtime(target, manifest, live=not offline)
    try:
        await graph.initialize()
        await asyncio.to_thread(cache.prepare, [d['contextual'] for d in docs]+[q for _, q in facets])
        engine = Engine(docs, cache)
        rankings, traces = [], []
        for collection, question in facets:
            base = engine.ranked(question, [collection], 'm3_bm25_rrf')[:24]
            found = await graph.retrieve(f'Report: {collection}\n{question}', 24, 'mix')
            if found['unknown_sources']:
                raise ValueError('Graph returned an unmapped source')
            all_ranked = engine.ranked(question, plan['collections'], 'm3_bm25_rrf')
            resolved = resolve_blocks(found['sources'], read(target/'blocks.json'), all_ranked)
            fused = fuse_sources(base, resolved, docs, collection)
            rankings.append(anchored_candidates(base, fused))
            traces.append(dict(collection=collection, question=question, graph_blocks=found['sources'],
                               entities=len(found['graph'].get('entities', [])),
                               relationships=len(found['graph'].get('relationships', []))))
        sources = pack(rankings, docs, True)
        result = dict(query=query, workspace=manifest['workspace'], sources=sources, traces=traces,
                      context_tokens=sum(text_tokens(context(d)) for d in sources),
                      calls=getattr(llm, 'calls', []), offline=offline,
                      note='Original evidence retrieval; no generated answer or factuality score')
        save(target/'query_latest.json', result)
        print(json.dumps(result, ensure_ascii=False))
    finally:
        await close_runtime(graph, cache, llm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['init', 'migrate', 'verify', 'query'])
    parser.add_argument('--source', type=Path)
    parser.add_argument('--target', type=Path, default=HERE/'runs'/'neo4j-multidoc')
    parser.add_argument('--query')
    parser.add_argument('--collection', action='append', default=[])
    parser.add_argument('--offline', action='store_true', help='Query using existing model/embedding caches only')
    args = parser.parse_args()
    if args.command == 'init':
        print('Created local credentials' if provision() else 'Existing local credentials preserved')
        return
    args.target = args.target.resolve()
    if args.command == 'query' and not args.query:
        parser.error('--query is required')
    with index_lock(args.target):
        if args.command == 'migrate':
            source = args.source or Path(read(HERE/'multidoc_latest.json')['path']).parent
            asyncio.run(migrate(source.resolve(), args.target))
        elif args.command == 'verify':
            asyncio.run(verify(args.target))
        else:
            asyncio.run(query_index(args.target, args.query, args.collection, offline=args.offline))


if __name__ == '__main__':
    main()
