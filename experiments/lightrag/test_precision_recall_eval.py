from copy import deepcopy
import pytest
from experiments.lightrag.precision_recall_eval import audited_tasks, score_row, aggregate


def row(sources, gold):
    return score_row(group='multidoc', case='case', category='test', arm='arm',
                     sources=sources, gold=gold, raw_gold=gold, allowed={'A'})


def test_fixed_k_precision_is_not_returned_precision_and_empty_retrieval_is_zero():
    value = row([dict(id='a', collection='A', safe='fact')], ['a','b'])
    assert value['metrics']['precision_at_k'] == 1/8
    assert value['metrics']['recall_at_k'] == 1/2
    assert value['returned_precision'] == 1
    empty = row([], ['a'])
    assert empty['returned_precision'] is None
    assert empty['metrics']['recall_at_k'] == 0
    result = aggregate([value, empty])['multidoc']['arm']
    assert result['cases'] == 2
    assert result['audited']['recall_at_k'] == 1/4


def test_duplicate_or_foreign_evidence_cannot_be_scored_as_valid():
    source = dict(id='a', collection='A', safe='fact')
    with pytest.raises(ValueError, match='budget'):
        row([source,source], ['a'])
    with pytest.raises(ValueError, match='scope'):
        row([dict(source, collection='B')], ['a'])


def test_label_audit_keeps_raw_gold_and_adds_missing_numerator_once():
    tasks = [dict(id='task', facets=[dict(id='f', required=['denominator'])])]
    original = deepcopy(tasks)
    corrected = audited_tasks(tasks, [dict(task='task', facet='f',
                                          additional_required=['numerator','denominator'])])
    assert tasks == original
    assert corrected[0]['facets'][0]['required'] == ['denominator','numerator']
