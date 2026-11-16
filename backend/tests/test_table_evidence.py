from backend.rag.evidence_pack import balanced_evidence, pack_evidence
from backend.rag.table_context import (
    contextual_table_row,
    header_rows,
    render_table_row,
)


def test_amounts_in_first_row_are_not_headings():
    table=[['aag series a preferred stock','$ 3329'],['single-dip equity obligations','1246'],['total','$ 5424']]
    assert header_rows(table)==[]
    assert render_table_row(table,0)=='aag series a preferred stock; column 1: $ 3329'
    assert render_table_row(table,1)=='single-dip equity obligations; column 1: 1246'
    assert '$ 3329' not in render_table_row(table,2)


def test_year_like_amount_with_metric_is_not_header():
    assert header_rows([['cash','2020'],['debt','1999']])==[]


def test_multirow_header_keeps_years_and_source_units():
    table=[['','Year ended (millions)','Year ended (millions)'],['','2019','2018'],['revenue','30','20']]
    text=render_table_row(table,2)
    assert 'Year ended (millions) 2019: 30' in text
    assert 'Year ended (millions) 2018: 20' in text


def test_employee_total_retains_table_topic():
    table=[['','American','US Airways'],['flight attendants','15000','7700'],['total','60100','32100']]
    text=contextual_table_row(table,2,['The following table lists the number of full-time employees by workgroup.'])
    assert 'employees' in text and 'American: 60100' in text


def doc(sid,collection,table=None,row=0,cost=10):
    return {'id':sid,'collection':collection,'table_id':table,'row_index':row,'cost':cost}


def test_parent_expansion_never_crosses_collection_and_budget():
    a=doc('numerator','A','table',1)
    total=doc('total','A','table',2)
    forbidden=doc('other-total','B','table',2)
    hits=pack_evidence([a],all_documents=[a,total,forbidden],token_count=lambda d:d['cost'],budget=40,expand_tables=True)
    assert [h['id'] for h in hits]==['numerator','total']


def test_balanced_summary_reserves_budget_for_every_facet():
    a=[doc(f'a{i}','A') for i in range(100)]
    b=[doc('b','B')]
    hits=balanced_evidence([a,b],all_documents=a+b,token_count=lambda d:d['cost'],budget=40)
    assert any(d['id']=='b' for d in hits)
    assert sum(d['cost'] for d in hits)<=40


def test_missing_facet_does_not_borrow_unauthorized_report():
    a=[doc('a','A')];secret=doc('secret','PRIVATE')
    hits=balanced_evidence([a,[]],all_documents=a+[secret],token_count=lambda d:d['cost'],budget=40)
    assert [d['id'] for d in hits]==['a']


def test_duplicates_do_not_consume_facet_budget_twice():
    a=doc('a','A');b=doc('b','B')
    hits=balanced_evidence([[a,a],[b]],all_documents=[a,b],token_count=lambda d:d['cost'],budget=20)
    assert [d['id'] for d in hits]==['a','b']


def test_shared_heading_keeps_all_parent_memberships_without_choosing_one():
    header=doc('header','A',None)
    header.update(table_ids=['p1','p2'],parent_rows={'p1':0,'p2':0})
    a=doc('p1-data','A','p1',1);b=doc('p2-data','A','p2',1)
    hits=pack_evidence([b],all_documents=[a,b,header],token_count=lambda d:d['cost'],budget=40,expand_tables=True)
    assert {d['id'] for d in hits}=={'header','p2-data'}
    hits=pack_evidence([header],all_documents=[a,b,header],token_count=lambda d:d['cost'],budget=40,expand_tables=True)
    assert [d['id'] for d in hits]==['header']
