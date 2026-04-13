from thunor.curve_fit import fit_drc, HillCurveLL4


def test_fit_drc_3_data_points():
    # 4 parameter fit with 3 data points - this should fail
    assert fit_drc([1, 2, 3], [4, 5, 6]) is None


def test_fit_drc_4_data_points():
    # 4 parameter fit with 4 data points - this should work
    # Use sigmoidal data with EC50 within dose range
    assert isinstance(
        fit_drc(
            [1, 2, 3, 4],
            [0.8759, 0.6488, 0.4689, 0.3528],
        ),
        HillCurveLL4,
    )
