import json
import os
import asyncio
import networkx as nx
import pytest
from experiments.lightrag.neo4j_config import configure, provision
from experiments.lightrag.neo4j_experiment import prepare, validate_graph, index_lock, REQUIRED_STORES, source_files


def test_graph_order_independent_of_backend_traversal_without_dropping_edges():
    from experiments.lightrag.adapters import stabilize_graph_order
    class Storage:
        def __init__(self, edges):
            self.edges = edges
        async def get_nodes_edges_batch(self, node_ids):
            return {'B': self.edges, 'missing': []}
    left = Storage([('B','Z'), ('B','A'), ('B','C')])
    right = Storage(list(reversed(left.edges)))
    stabilize_graph_order(left)
    stabilize_graph_order(right)
    a = asyncio.run(left.get_nodes_edges_batch(['B','missing']))
    b = asyncio.run(right.get_nodes_edges_batch(['B','missing']))
    assert a == b and len(a['B']) == 3
    assert all(edge[0] == 'B' for edge in a['B'])


def test_lock_rejects_concurrent_index_writes_and_releases_on_failure(tmp_path):
    target = tmp_path/'index'
    with pytest.raises(ValueError, match='simulated'):
        with index_lock(target):
            with pytest.raises(RuntimeError, match='locked'):
                with index_lock(target):
                    pytest.fail('Concurrent access was allowed')
            raise ValueError('simulated operation failure')
    with index_lock(target):
        pass


def test_credentials_are_stable_and_never_overwritten(tmp_path):
    path = tmp_path/'env'
    assert provision(path)
    original = path.read_bytes()
    assert not provision(path)
    assert path.read_bytes() == original


def test_conflicting_workspace_rejected_before_environment_change(tmp_path, monkeypatch):
    path = tmp_path/'env'
    provision(path)
    monkeypatch.setenv('NEO4J_WORKSPACE', 'other_index')
    original = dict(os.environ)
    with pytest.raises(ValueError, match='conflicts'):
        configure('finsight_lightrag_abc', path)
    assert dict(os.environ) == original


def test_invalid_label_rejected(tmp_path):
    with pytest.raises(ValueError, match='isolated'):
        configure('unsafe` MATCH (n)', tmp_path/'env')


def test_restore_identity_and_no_overwrite(tmp_path):
    source = tmp_path/'source'
    store = source/'index'/'finsight_experiment'
    store.mkdir(parents=True)
    for name in REQUIRED_STORES:
        (store/name).write_text('{}')
    for name in ['blocks.json', 'plan.json', 'vectors.sqlite']:
        (source/name).write_text('{}')
    (source/'report.json').write_text(json.dumps(dict(status='complete', calls=[{'model':'test'}])))
    target = tmp_path/'target'
    manifest = prepare(source, target)
    assert prepare(source, target) == manifest
    assert prepare(source, tmp_path/'another')['workspace'] != manifest['workspace']
    (source/'blocks.json').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        prepare(source, target)
    with pytest.raises(ValueError, match='separate'):
        prepare(source, source/'nested')


def test_missing_vector_store_cannot_be_accepted_as_empty_index(tmp_path):
    store = tmp_path/'index'/'finsight_experiment'
    store.mkdir(parents=True)
    for name in REQUIRED_STORES - {'vdb_entities.json'}:
        (store/name).write_text('{}')
    with pytest.raises(ValueError, match='vdb_entities.json'):
        source_files(tmp_path)


def test_graph_check_requires_full_properties_and_resolved_sources():
    graph = nx.Graph()
    props = dict(entity_id='A', source_id='chunk')
    graph.add_node('A', **props)
    assert validate_graph(graph, {'A':props}, {}, {'chunk':{}})['all_properties_equal']
    with pytest.raises(ValueError, match='unresolved'):
        validate_graph(graph, {'A':props}, {}, {})
    with pytest.raises(ValueError, match='exactly'):
        validate_graph(graph, {'A':{**props,'source_id':'other'}}, {}, {'chunk':{}})
