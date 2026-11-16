from experiments.financial_retrieval_v2.answer_audit import number_reference, score_facet


def fixture():
    facet=dict(id='1',collection='A/2024',question='What percent?',answer='12.0%',required=['a','b'],answerable=True)
    docs={'a':dict(collection='A/2024',safe='Numerator: 12'), 'b':dict(collection='A/2024',safe='Total: 100')}
    output=dict(report='A/2024',answer=12,unit='percent',citations=['a','b'],
                quotes=[dict(source_id='a',text='Numerator: 12'),dict(source_id='b',text='Total: 100')])
    return facet,docs,output


def test_rounding_matches_author_precision_and_rejects_nonfinite():
    from decimal import Decimal
    reference,tolerance=number_reference('3.2%')
    assert abs(Decimal('3.185')-reference)<=tolerance
    assert number_reference('NaN') is None


def test_correct_number_with_wrong_unit_is_not_a_verified_answer():
    f,d,o=fixture();assert score_facet(f,o,d)['numeric_and_evidence_correct']
    o['unit']='percentage_points'
    assert not score_facet(f,o,d)['numeric_and_evidence_correct']


def test_missing_denominator_citation_fails_complete_support():
    f,d,o=fixture();o['citations']=['a'];o['quotes']=o['quotes'][:1]
    r=score_facet(f,o,d)
    assert r['citation_ids_valid'] and not r['gold_citation_complete']
    o['citations']=['a','b']
    assert not score_facet(f,o,d)['numeric_and_evidence_correct']


def test_invented_quote_and_wrong_report_are_rejected():
    f,d,o=fixture();o['quotes'][0]['text']='Numerator: 15'
    assert not score_facet(f,o,d)['quotes_valid']
    o['report']='A/2023'
    assert not score_facet(f,o,d)['report_correct']


def test_missing_report_requires_local_abstention():
    f,d,o=fixture();f['answerable']=False
    assert not score_facet(f,o,d)['missing_report_correct']
    o.update(answer=None,citations=[],quotes=[])
    assert score_facet(f,o,d)['missing_report_correct']


def test_absent_reference_is_not_counted_as_a_fully_scorable_case():
    from experiments.financial_retrieval_v2.answer_audit import summarize
    f,d,o=fixture();f['answer']=''
    row=dict(kind='multi',arm='selected',facets=[score_facet(f,o,d)])
    result=summarize([row])['multi']['selected']
    assert result['complete_cases_scored']==0 and result['complete_cases_unscored']==1
