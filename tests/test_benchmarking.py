import numpy as np
import pandas as pd
from src.benchmarking import run_cross_validation

def test_regression_cv_report_excludes_synthetic_rows_and_has_fold_metrics():
    rng=np.random.default_rng(7)
    x=rng.normal(size=60)
    df=pd.DataFrame({'measured_temp':x*4+170,'measured_time':rng.uniform(1,5,60),'quality_target':x+rng.normal(0,.1,60)})
    report,path=run_cross_validation(df,'quality_target',['measured_temp','measured_time'],task='regression',dataset_name='unit_test_real',folds=3,selected_models=['Random Forest'])
    assert report['source_class']=='externally_sourced_real_data'
    assert report['usable_rows']==60
    assert report['scores'][0]['status']=='ok'
    assert len(report['scores'][0]['metrics']['mae']['folds'])==3
    assert path.exists()

def test_rejects_empty_feature_selection():
    df=pd.DataFrame({'x':range(30),'y':range(30)})
    try:
        run_cross_validation(df,'y',[],folds=3,selected_models=['Random Forest'])
    except ValueError as e:
        assert 'at least one input feature' in str(e).lower()
    else:
        raise AssertionError('Expected empty features to be rejected')
