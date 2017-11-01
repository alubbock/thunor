import pkg_resources
from pydrc.io import read_hdf
from pydrc.dip import dip_rates, dip_fit_params
from pydrc.plots import plot_dip, plot_dip_params, plot_time_course, \
    plot_ctrl_dip_by_plate


def test_tutorial():
    vu001_file = pkg_resources.resource_filename('pydrc', 'testdata/VU001.h5')

    vu001 = read_hdf(vu001_file)

    ctrl_dip_data, expt_dip_data = dip_rates(vu001)
    fit_params = dip_fit_params(ctrl_dip_data, expt_dip_data)

    plot_dip(fit_params)

    plot_dip_params(fit_params, 'auc')

    # Test these methods
    vu001.drugs
    vu001.cell_lines

    plot_time_course(vu001.filter(drugs=['abemaciclib'], cell_lines=['BT20']))

    plot_ctrl_dip_by_plate(ctrl_dip_data)
