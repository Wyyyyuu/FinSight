from experiments.lightrag.multidoc_report import scale_matches


def test_monetary_scale_must_match_requested_units():
    assert scale_matches('Total issued, in millions?', 'USD millions') is True
    assert scale_matches('Total issued, in millions?', 'USD') is False
    assert scale_matches('Change ( in billions ) ?', 'million USD') is False
    assert scale_matches('Change ( in billions ) ?', 'billions') is True
    assert scale_matches('What is the percent change?', 'percent') is None
