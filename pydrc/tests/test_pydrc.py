import pkg_resources
from pydrc.io import read_hdf
from pydrc.dip import dip_rates, dip_fit_params
from pydrc.plots import plot_dip, plot_dip_params, plot_time_course


def test_tutorial():
    vu001_file = pkg_resources.resource_filename('pydrc', 'testdata/VU001.h5')

    vu001 = read_hdf(vu001_file)

    ctrl_dip_data, expt_dip_data = dip_rates(vu001)
    fit_params = dip_fit_params(ctrl_dip_data, expt_dip_data)

    plot_dip(fit_params)

    plot_dip_params(fit_params, 'auc')

    df_doses_filtered = vu001['doses'].xs(['abemaciclib', 'BT20'],
                                          level=['drug', 'cell_line'],
                                          drop_level=False)
    df_controls_filtered = vu001['controls'].loc['Cell count', 'BT20']
    df_assays_filtered = vu001['assays'].loc['Cell count']
    plot_time_course(df_doses_filtered,
                     df_assays_filtered,
                     df_controls_filtered)
