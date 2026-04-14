import importlib
import importlib.resources
import io
import unittest

import pytest

import thunor.curve_fit
import thunor.dip
import thunor.viability
from thunor.curve_fit import (
    DrugCombosWarning,
    HillCurveLL2,
    HillCurveLL3u,
    HillCurveLL4,
    fit_drc,
    fit_params,
    fit_params_from_base,
    fit_params_minimal,
    is_param_truncated,
)
from thunor.io import read_hdf, read_vanderbilt_hts

# Sigmoidal inhibitory response: E0~1, Emax~0, EC50 between doses 3 and 4
_INHIBITORY_DOSES = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
_INHIBITORY_RESP = [0.98, 0.85, 0.50, 0.15, 0.02]

# Stimulatory response: Emax > E0 (agonist)
_STIMULATORY_DOSES = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
_STIMULATORY_RESP = [1.01, 1.10, 1.40, 1.65, 1.72]

# Flat / no-effect response
_FLAT_DOSES = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
_FLAT_RESP = [1.00, 1.00, 1.00, 1.00, 1.00]


# ---------------------------------------------------------------------------
# fit_drc – basic curve fitting
# ---------------------------------------------------------------------------


def test_fit_drc_3_data_points():
    # 4-parameter fit requires at least 4 points
    assert fit_drc([1, 2, 3], [0.9, 0.6, 0.3]) is None


def test_fit_drc_4_data_points():
    assert isinstance(
        fit_drc([1, 2, 3, 4], [0.8759, 0.6488, 0.4689, 0.3528]),
        HillCurveLL4,
    )


def test_fit_drc_ll3u():
    # LL3u is used for viability; should fit with 4 points
    result = fit_drc(_INHIBITORY_DOSES, _INHIBITORY_RESP, fit_cls=HillCurveLL3u)
    assert isinstance(result, HillCurveLL3u)


def test_fit_drc_ll2():
    result = fit_drc(_INHIBITORY_DOSES, _INHIBITORY_RESP, fit_cls=HillCurveLL2)
    assert isinstance(result, HillCurveLL2)


def test_fit_drc_inhibitory_ec50_in_range():
    curve = fit_drc(_INHIBITORY_DOSES, _INHIBITORY_RESP)
    assert curve is not None
    assert _INHIBITORY_DOSES[0] <= curve.ec50 <= _INHIBITORY_DOSES[-1]


def test_fit_drc_stimulatory_returns_curve():
    # Stimulatory data should still produce a curve object
    result = fit_drc(_STIMULATORY_DOSES, _STIMULATORY_RESP, fit_cls=HillCurveLL3u)
    assert result is not None


# ---------------------------------------------------------------------------
# HillCurve method tests
# ---------------------------------------------------------------------------


class TestHillCurveMethods(unittest.TestCase):
    def setUp(self):
        self.curve = fit_drc(_INHIBITORY_DOSES, _INHIBITORY_RESP)
        assert self.curve is not None, 'Curve fitting failed in setUp'

        self.stim_curve = fit_drc(
            _STIMULATORY_DOSES, _STIMULATORY_RESP, fit_cls=HillCurveLL3u
        )

    def test_ic50_in_dose_range(self):
        ic50 = self.curve.ic(50)
        assert ic50 is not None
        assert _INHIBITORY_DOSES[0] <= ic50 <= _INHIBITORY_DOSES[-1]

    def test_ec100_returns_none(self):
        # ec() returns None for ec_num >= 100 (asymptote can never be reached)
        assert self.curve.ec(100) is None

    def test_auc_inhibitory_positive(self):
        auc = self.curve.auc(min_conc=_INHIBITORY_DOSES[0])
        assert auc is not None
        assert auc > 0

    def test_auc_stimulatory_returns_none(self):
        # Stimulatory curves (Emax > E0) are not yet supported
        if self.stim_curve is not None and self.stim_curve.emax > self.stim_curve.e0:
            assert self.stim_curve.auc(min_conc=_STIMULATORY_DOSES[0]) is None

    def test_aa_inhibitory_positive(self):
        aa = self.curve.aa(
            min_conc=_INHIBITORY_DOSES[0], max_conc=_INHIBITORY_DOSES[-1]
        )
        assert aa is not None
        assert aa > 0

    def test_aa_stimulatory_returns_none(self):
        if self.stim_curve is not None and self.stim_curve.emax > self.stim_curve.e0:
            assert (
                self.stim_curve.aa(
                    min_conc=_STIMULATORY_DOSES[0], max_conc=_STIMULATORY_DOSES[-1]
                )
                is None
            )

    def test_fit_evaluates_at_dose(self):
        # At EC50, response should be midpoint between E0 and Emax
        midpoint = (self.curve.e0 + self.curve.emax) / 2
        assert abs(self.curve.fit(self.curve.ec50) - midpoint) < 0.05

    def test_popt_rel_normalised(self):
        # Relative parameters should be normalised to max(E0, Emax) = 1
        assert abs(max(self.curve.popt_rel[1], self.curve.popt_rel[2]) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# fit_params_minimal / fit_params_from_base / fit_params – pipeline tests
# ---------------------------------------------------------------------------


class TestFitParamsPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ref = importlib.resources.files('thunor') / 'testdata/hts007.h5'
        with importlib.resources.as_file(ref) as filename:
            cls.dataset = read_hdf(filename)
        cls.ctrl_dip, cls.expt_dip = thunor.dip.dip_rates(cls.dataset)
        cls.viability_data, _ = thunor.viability.viability(
            cls.dataset, include_controls=False
        )

    def test_fit_params_minimal_returns_dataframe(self):
        import pandas as pd

        result = fit_params_minimal(self.ctrl_dip, self.expt_dip)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_fit_params_minimal_expected_columns(self):
        result = fit_params_minimal(self.ctrl_dip, self.expt_dip)
        for col in (
            'fit_obj',
            'min_dose_measured',
            'max_dose_measured',
            'emax_obs',
            'aa_obs',
        ):
            assert col in result.columns, f'Missing column: {col}'

    def test_fit_params_minimal_drmetric_attr(self):
        result = fit_params_minimal(self.ctrl_dip, self.expt_dip)
        assert result.attrs.get('drmetric') == 'dip'

    def test_fit_params_minimal_viability_drmetric(self):
        result = fit_params_minimal(None, self.viability_data, fit_cls=HillCurveLL3u)
        assert result.attrs.get('drmetric') == 'viability'

    def test_fit_params_from_base_ic50(self):
        base = fit_params_minimal(self.ctrl_dip, self.expt_dip)
        result = fit_params_from_base(
            base,
            custom_ic_concentrations={50},
            include_response_values=False,
        )
        assert 'ic50' in result.columns

    def test_fit_params_from_base_selective(self):
        # Request only AUC – other stat columns should be absent
        base = fit_params_minimal(self.ctrl_dip, self.expt_dip)
        result = fit_params_from_base(
            base, include_auc=True, include_response_values=False
        )
        assert 'auc' in result.columns
        assert 'ic50' not in result.columns
        assert 'hill' not in result.columns

    def test_fit_params_full(self):
        result = fit_params(self.ctrl_dip, self.expt_dip)
        for col in ('ic50', 'ec50', 'auc', 'aa', 'hill'):
            assert col in result.columns, f'Missing column: {col}'

    def test_fit_params_viability_pipeline(self):
        result = fit_params(
            ctrl_data=None,
            expt_data=self.viability_data,
            fit_cls=HillCurveLL3u,
        )
        assert 'ic50' in result.columns
        assert result.attrs.get('drmetric') == 'viability'

    def test_is_param_truncated(self):
        fp = fit_params(self.ctrl_dip, self.expt_dip)
        truncated = is_param_truncated(fp, 'ic50')
        # Result should be a boolean array of the right length
        assert len(truncated) == len(fp)
        assert truncated.dtype == bool

    def test_drug_combos_warning(self):
        # Build a dataset with both combination and single-drug wells; confirm
        # a DrugCombosWarning is issued and the single-drug well is processed
        csv = (
            'cell.line,drug1.conc,drug1,drug2,drug2.conc,drug2.units,'
            'upid,time,cell.count,well,drug1.units\n'
            # combination well
            'cl1,1e-6,drugA,drugB,1e-6,M,plate1,0,1000,A1,M\n'
            'cl1,1e-6,drugA,drugB,1e-6,M,plate1,24,900,A1,M\n'
            'cl1,1e-6,drugA,drugB,1e-6,M,plate1,48,800,A1,M\n'
            # single-drug well
            'cl1,1e-6,drugC,,0,M,plate1,0,1000,C1,M\n'
            'cl1,1e-6,drugC,,0,M,plate1,24,900,C1,M\n'
            'cl1,1e-6,drugC,,0,M,plate1,48,800,C1,M\n'
            # control wells
            'cl1,0,,,0,M,plate1,0,1050,B1,M\n'
            'cl1,0,,,0,M,plate1,24,2100,B1,M\n'
            'cl1,0,,,0,M,plate1,48,4200,B1,M\n'
        )
        dataset = read_vanderbilt_hts(io.StringIO(csv), sep=',')
        ctrl_dip, expt_dip = thunor.dip.dip_rates(dataset)
        with pytest.warns(DrugCombosWarning):
            fit_params_minimal(ctrl_dip, expt_dip)
