from thunor.curve_fit import fit_drc, HillCurveLL4
from numpy.testing import assert_raises


def test_fit_drc_3_data_points():
    # 4 parameter fit with 3 data points - this should fail
    assert fit_drc([1, 2, 3], [4, 5, 6]) is None


def test_fit_drc_4_data_points():
    # 4 parameter fit with 4 data points - this should work
    assert isinstance(
        fit_drc([1, 2, 3, 4], [5, 6, 7, 8]),
        HillCurveLL4
    )
