import unittest
import pkg_resources
import thunor.io
import thunor.dip
import thunor.viability
import thunor.curve_fit
from thunor.plots import plot_time_course, plot_drc, plot_drc_params, \
    plot_plate_map, plot_ctrl_dip_by_plate
from thunor.helpers import plotly_to_dataframe
import pandas as pd


class TestWithDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.filename = pkg_resources.resource_filename('thunor',
                                                       'testdata/hts007.h5')
        cls.dataset = thunor.io.read_hdf(cls.filename)
        ctrl_dip_data, expt_dip_data = thunor.dip.dip_rates(cls.dataset)
        cls.ctrl_dip_data = ctrl_dip_data

        cls.fit_params = thunor.curve_fit.fit_params(ctrl_dip_data,
                                                     expt_dip_data)

        viability_data, _ = thunor.viability.viability(
            cls.dataset, include_controls=False)
        cls.viability_params = thunor.curve_fit.fit_params(
            fit_cls=thunor.curve_fit.HillCurveLL3u,
            ctrl_data=None, expt_data=viability_data)

    def test_plot_viability_curves(self):
        assert isinstance(plotly_to_dataframe(plot_drc(
            self.viability_params)), pd.DataFrame)

    def test_plot_viability_params_single_param(self):
        assert isinstance(plotly_to_dataframe(plot_drc_params(
            self.viability_params, fit_param='ic50')),
                          pd.DataFrame)

    def test_plot_param_comparison(self):
        # Mock up a two-dataset set of fit params
        df1 = self.fit_params.copy()
        df1.index.set_levels(['one'], level='dataset_id', inplace=True)

        df2 = df1.copy()
        df2.index.set_levels(['two'], level='dataset_id', inplace=True)

        df = pd.concat([df1, df2])

        df._drmetric = self.fit_params._drmetric

        plot_drc_params(df, fit_param='ec50', multi_dataset=True)

    def test_plot_time_course(self):
        abe_bt20 = self.dataset.filter(drugs=['abemaciclib'],
                                       cell_lines=['BT20'])

        tc = plot_time_course(abe_bt20, show_dip_fit=True, log_yaxis=True)
        assert isinstance(plotly_to_dataframe(tc), pd.DataFrame)

    def test_plot_dip(self):
        assert isinstance(plotly_to_dataframe(plot_drc(self.fit_params)),
                          pd.DataFrame)

    def test_plot_dip_single_curve(self):
        fit_data = self.fit_params.iloc[[0], :]
        assert isinstance(plotly_to_dataframe(plot_drc(fit_data)),
                          pd.DataFrame)

    def test_plot_dip_params_single_param(self):
        assert isinstance(plotly_to_dataframe(plot_drc_params(
            self.fit_params, fit_param='ic50')),
                          pd.DataFrame)

    def test_plot_two_params(self):
        x = plot_drc_params(
            self.fit_params, fit_param='ic50', fit_param_compare='ec50',
            fit_param_sort='ec25')

        assert isinstance(plotly_to_dataframe(x),
            pd.DataFrame)

    def test_plot_dip_params_aggregation(self):
        assert isinstance(plotly_to_dataframe(plot_drc_params(
            self.fit_params, aggregate_cell_lines={'tag': ['BT20', 'HCC1143']},
            aggregate_drugs={'tag': ['abemaciclib', 'Panobinostat']},
            fit_param='ic50')),
                          pd.DataFrame)

    def test_plot_ctrl_dip_by_plate(self):
        res = plot_ctrl_dip_by_plate(self.ctrl_dip_data)
        assert isinstance(plotly_to_dataframe(res), pd.DataFrame)

    def test_plot_plate_map(self):
        plate_data = self.dataset.plate('HTS007_149-28A',
                                        include_dip_rates=True)
        res = plot_plate_map(
            plate_data
        )
        assert isinstance(plotly_to_dataframe(res), pd.DataFrame)
