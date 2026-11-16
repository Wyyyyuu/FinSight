from copy import deepcopy
from experiments.lightrag.multidoc_pilot import choose, facets_from_query, fuse_sources, pack, graph_blocks, resolve_blocks, successful
import pytest


def test_selection_does_not_depend_on_gold_answers():
    tasks = [dict(id=str(i), split='dev', category=cat, collections=['A'], answer='secret')
             for i,cat in enumerate(['cross_company','cross_year','three_reports'])]
    data = {'multi':tasks,'documents':[dict(id='a',collection='A',contextual='text')]}
    changed = deepcopy(data)
    for task in changed['multi']:
        task['answer'] = 'other'
        task['required'] = ['nonexistent']
    assert [t['id'] for t in choose(data)[0]] == [t['id'] for t in choose(changed)[0]]
    assert choose(data)[1] == choose(changed)[1]


def test_query_facets_preserve_missing_report_request():
    assert facets_from_query('Summary\n- [missing:2025] What was revenue?') == [('missing:2025','What was revenue?')]


def test_graph_fusion_rejects_unknown_and_cross_report_sources():
    docs = [dict(id='a',collection='A'),dict(id='b',collection='A'),dict(id='c',collection='B')]
    ranked = fuse_sources([docs[0]], ['b','b','c','invented'], docs, 'A')
    assert {d['id'] for d in ranked} == {'a','b'}
    assert len(ranked)==2


def test_balanced_pack_covers_both_facets_and_honors_eight_sources():
    docs = [dict(id=str(i),collection='A' if i<10 else 'B',safe='short original',table_id=None) for i in range(20)]
    result = pack([docs[:10], docs[10:]], docs, balanced=True)
    assert len(result)==8
    assert {d['collection'] for d in result} == {'A','B'}


def test_pack_never_truncates_an_over_budget_source_into_evidence():
    docs = [dict(id='huge',collection='A',safe='word '*5000),dict(id='small',collection='B',safe='fact')]
    assert [d['id'] for d in pack([docs],docs)] == ['small']


def test_blocks_preserve_report_page_and_all_original_ids():
    docs=[dict(id='a',collection='A',table_id='page1',contextual='first'),
          dict(id='b',collection='A',table_id='page1',contextual='second'),
          dict(id='c',collection='B',table_id='page1',contextual='third')]
    blocks=graph_blocks(docs)
    assert len(blocks)==2 and blocks[0]['source_ids']==['a','b']
    assert resolve_blocks([blocks[0]['id']],blocks,[docs[1],docs[0],docs[2]])==['b','a']
    with pytest.raises(ValueError):
        resolve_blocks(['invented'],blocks,docs)


def test_resume_retries_failed_answers_but_keeps_successes():
    rows=[dict(id='t',arm='mix',error='TimeoutError'),dict(id='t',arm='base',output={})]
    assert not successful(rows,'t','mix')
    assert successful(rows,'t','base')


def test_guarded_fusion_preserves_anchors_without_duplicating_sources():
    from experiments.lightrag.multidoc_guarded import anchored_candidates
    base=[dict(id=str(i)) for i in range(6)]
    fused=list(reversed(base))
    selected=anchored_candidates(base,fused,anchors=2,limit=4)
    assert [d['id'] for d in selected]==['0','1','5','4']
