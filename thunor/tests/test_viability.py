import importlib
import importlib.resources
import unittest

import numpy as np
import pandas as pd
import pytest

import thunor.viability
from thunor.io import HtsPandas, read_hdf


class TestViabilityWithDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ref = importlib.resources.files('thunor') / 'testdata/hts007.h5'
        with importlib.resources.as_file(ref) as filename:
            cls.dataset = read_hdf(filename)

    def test_returns_dataframe_and_series(self):
        df, ctrl = thunor.viability.viability(self.dataset, include_controls=True)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(ctrl, pd.Series)

    def test_include_controls_false_returns_none_ctrl(self):
        df, ctrl = thunor.viability.viability(self.dataset, include_controls=False)
        assert isinstance(df, pd.DataFrame)
        assert ctrl is None

    def test_viability_column_present(self):
        df, _ = thunor.viability.viability(self.dataset, include_controls=False)
        assert 'viability' in df.columns

    def test_viability_values_finite(self):
        df, _ = thunor.viability.viability(self.dataset, include_controls=False)
        assert np.isfinite(df['viability'].values).all()

    def test_viability_time_attr_set(self):
        df, _ = thunor.viability.viability(self.dataset, include_controls=False)
        assert 'viability_time' in df.attrs

    def test_viability_assay_attr_set(self):
        df, _ = thunor.viability.viability(self.dataset, include_controls=False)
        assert 'viability_assay' in df.attrs

    def test_time_hrs_parameter(self):
        df24, _ = thunor.viability.viability(
            self.dataset, time_hrs=24, include_controls=False
        )
        df72, _ = thunor.viability.viability(
            self.dataset, time_hrs=72, include_controls=False
        )
        # Different time points should produce different viability values
        assert not df24['viability'].equals(df72['viability'])

    def test_control_viability_near_one(self):
        _, ctrl = thunor.viability.viability(self.dataset, include_controls=True)
        # Control values are normalised by their own mean, so the mean should be ~1
        assert abs(ctrl.mean() - 1.0) < 0.1

    def test_no_controls_raises(self):
        dataset_no_ctrl = HtsPandas(
            doses=self.dataset.doses,
            assays=self.dataset.assays,
            controls=None,
        )
        with pytest.raises(ValueError, match='Control wells not found'):
            thunor.viability.viability(dataset_no_ctrl, include_controls=False)

    def test_index_contains_drug_cell_line_dose(self):
        df, _ = thunor.viability.viability(self.dataset, include_controls=False)
        for level in ('drug', 'cell_line', 'dose'):
            assert level in df.index.names, f'Missing index level: {level}'
