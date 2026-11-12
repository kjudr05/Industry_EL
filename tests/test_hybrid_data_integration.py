import pandas as pd
from src.hybrid_data_integration import compare_sources, validate_join

def test_sources_are_compared_without_merging():
    real = pd.DataFrame({'temperature_c':[160,175], 'measured_oil_quality':[1.2,1.4]})
    synthetic = pd.DataFrame({'temperature_c':[160,170], 'quality_score':[80,82]})
    report = compare_sources(real, synthetic)
    assert report['real']['source_type'] == 'real'
    assert report['synthetic']['source_type'] == 'synthetic'
    assert report['same_named_columns'] == ['temperature_c']
    assert 'do not prove' in report['note']

def test_join_requires_unique_real_shared_key():
    a = pd.DataFrame({'batch_id':[1,2], 'x':[3,4]})
    b = pd.DataFrame({'batch_id':[1,2], 'y':[5,6]})
    ok, _ = validate_join(a,b,'batch_id')
    assert ok
    b_dup = pd.DataFrame({'batch_id':[1,1], 'y':[5,6]})
    ok, message = validate_join(a,b_dup,'batch_id')
    assert not ok and 'not unique' in message

def test_missing_or_unmatched_key_rejected():
    a = pd.DataFrame({'id':[1,None]})
    b = pd.DataFrame({'id':[1,2]})
    assert not validate_join(a,b,'id')[0]
    assert not validate_join(pd.DataFrame({'x':[1]}), b, 'id')[0]
