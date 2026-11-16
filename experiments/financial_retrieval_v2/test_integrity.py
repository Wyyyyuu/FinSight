import json
from pathlib import Path
import pytest
import numpy as np
from experiments.financial_retrieval_v2.evaluate import Retriever, facet_plan, sha, metric, token_count


def test_request_plan_only_uses_explicit_query_references():
    plan=facet_plan('Summarize:\n- [finqa:A/2024] Revenue?\n- [finqa:B/2023] Cash flow?')
    assert plan==[{'collection':'finqa:A/2024','question':'Revenue?'},{'collection':'finqa:B/2023','question':'Cash flow?'}]
    assert facet_plan('Revenue?')==[]


def test_ranking_does_not_accept_or_read_answer_labels():
    docs=[dict(id='a',collection='c',safe='Revenue is 20',plain='Revenue is 20'),
          dict(id='b',collection='other',safe='Revenue is 99',plain='Revenue is 99')]
    vectors={sha('Revenue is 20'):np.array([1.,0.]),sha('Revenue is 99'):np.array([1.,0.]),sha('Revenue'):np.array([1.,0.])}
    engine=Retriever(docs,vectors,dict(field='safe'))
    assert [d['id'] for d in engine.ranked('Revenue',['c'])]==['a']
    assert engine.ranked('Revenue',['missing'])==[]


def test_partial_evidence_is_not_complete():
    assert metric(['a'],['a','b'])['all_evidence']==0
    assert metric(['a'],['a','b'])['recall_at_k']==.5


def test_chinese_context_locator_is_serializable():
    d=dict(id='zh',collection='zh:1',safe='营业收入：10',locations=[dict(context_uid='context1',table_row=2)])
    assert token_count(d)>0


def test_new_dataset_excludes_all_previously_used_companies():
    path=Path(__file__).parent/'dataset.json'
    if not path.exists():pytest.skip('Build public data first')
    data=json.loads(path.read_text(encoding='utf-8'))
    old=json.loads((path.parent.parent/'financial_retrieval/dataset.json').read_text(encoding='utf-8'))
    seen={q['group'].split('/')[0] for q in old['questions'] if q['dataset']=='FinQA'}
    dev={q['company'] for q in data['questions'] if q['split']=='dev'}
    test={q['company'] for q in data['questions'] if q['split']=='test'}
    assert not dev&test and not seen&(dev|test)
    docs={d['id']:d for d in data['documents']}
    for task in data['multi']:
        for facet in task['facets']:
            assert all(docs[s]['collection']==facet['collection'] for s in facet['required'])
            assert facet['answerable']==(facet['collection'] in task['collections'])
            if not facet['answerable']:
                assert not any(d['collection'] in task['collections'] for s,d in docs.items() if s in facet['required'])
