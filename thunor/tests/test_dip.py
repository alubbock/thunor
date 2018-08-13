from .test_io import CSV_HEADER
from thunor.dip import dip_rates
from thunor.io import read_vanderbilt_hts
import io
import numpy as np


def test_dip_two_time_points():
    csv = CSV_HEADER + \
          '\ncl1,0.00013,drug1,plate1,0,1234,A1,M' + \
          '\ncl1,0.00013,drug1,plate1,1,2468,A1,M'
    d = read_vanderbilt_hts(io.StringIO(csv), sep=',')
    ctrl_dip_data, expt_dip_data = dip_rates(d)

    assert ctrl_dip_data is None

    # Doubling time is 1hr === DIP rate is 1.0
    assert np.allclose(expt_dip_data['dip_rate'], [1.0])


# test_plots.py has functions which test DIP with >2 time points
