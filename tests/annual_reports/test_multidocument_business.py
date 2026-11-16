"""Source-backed business regression cases, NOT a fresh heldout dataset.

Uses the already reviewed CNINFO excerpts and physical source-page references in
test_real_financial_tables.py (Gree p7, Haier p41, Hisense p30/p116).
"""
from decimal import Decimal

import pytest
from test_real_financial_tables import (
    GREE_TABLE,
    HAIER_PAGE41,
    HISENSE_PAGE30,
    HISENSE_PAGE116,
)

from backend.annual_reports import workflow
from backend.annual_reports.documents import AnnualReportStore


@pytest.fixture
def source_store(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, '_request_model_answer', lambda state: None)
    store=AnnualReportStore(tmp_path)
    selected=[]
    for company,text in [('格力电器',GREE_TABLE),('海尔智家',HAIER_PAGE41),
                         ('海信家电',HISENSE_PAGE30+'\f'+HISENSE_PAGE116)]:
        selected.append(store.ingest_bytes(text.encode('utf-8'),company+'_2024_公开年报摘录.txt',company,2024)['id'])
    return store,selected


def test_three_public_reports_six_changes_keep_company_year_units(source_store):
    store,selected=source_store
    result=workflow.analyze_reports(store,'总结三家公司2023和2024年的营业收入与经营现金流变化',selected,mode='bm25')
    assert result['status']=='complete',result['metrics']['evidence_gaps']
    assert result['metrics']['fact_count']==12
    calculations={(c['company'],c['metric']):c for c in result['calculations']}
    assert len(calculations)==6
    expected={('格力电器','营业收入'):Decimal('189163654064.64')-Decimal('203979266387.09'),
              ('海尔智家','营业收入'):Decimal('285981225203.93')-Decimal('274204520847.97'),
              ('海信家电','经营活动产生的现金流量净额'):Decimal('5132164941.24')-Decimal('10611857591.35')}
    citations={c['label']:c for c in result['citations']}
    for key,delta in expected.items():
        calc=calculations[key]
        assert Decimal(str(calc['delta']))==delta
        assert (calc['from_year'],calc['to_year'])==(2023,2024)
        assert all(citations[label]['company']==key[0] for label in calc['source_labels'])


def test_summary_does_not_replace_missing_company_with_available_peer(source_store):
    store,selected=source_store
    result=workflow.analyze_reports(store,'比较格力电器、海尔智家、海信家电2024年营业收入',selected[:2],mode='bm25')
    assert result['status']=='insufficient_evidence'
    assert any('海信' in g for g in result['metrics']['evidence_gaps'])
    assert all(c['company']!='海信家电' for c in result['citations'])


def test_wrong_cross_company_direction_is_not_repeated_as_fact(source_store):
    store,selected=source_store
    result=workflow.analyze_reports(store,'解释2023到2024年三家公司经营现金流都增长的原因',selected,mode='bm25')
    assert result['status']=='insufficient_evidence'
    assert any('格力' in g and '并非' in g for g in result['metrics']['evidence_gaps'])
    assert any('海信' in g and '并非' in g for g in result['metrics']['evidence_gaps'])


@pytest.mark.parametrize('question',[
    '比较2023和2024年的营业收入与经营现金流',
    '总结三家公司2024年营业收入',
    '比较营业收入和归母净利润',
])
def test_topic_lists_are_not_invented_missing_companies(question):
    assert workflow._missing_named_comparands(question,{'格力电器','海尔智家'})==[]
