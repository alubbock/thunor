import importlib
import importlib.resources
import io

import numpy as np
import pytest

import thunor.dip
from .test_io import CSV_HEADER
from thunor.dip import dip_rates
from thunor.io import read_vanderbilt_hts, read_hdf


def test_dip_two_time_points():
    csv = (
        CSV_HEADER
        + '\ncl1,0.00013,drug1,plate1,0,1234,A1,M'
        + '\ncl1,0.00013,drug1,plate1,1,2468,A1,M'
    )
    d = read_vanderbilt_hts(io.StringIO(csv), sep=',')
    ctrl_dip_data, expt_dip_data = dip_rates(d)

    assert ctrl_dip_data is None

    # Doubling time is 1hr === DIP rate is 1.0
    assert np.allclose(expt_dip_data['dip_rate'], [1.0])


# test_plots.py has functions which test DIP with >2 time points


def test_dip_numpy_matches_scipy_reference():
    """Vectorised numpy DIP rates must match the scipy.stats.linregress reference."""
    import scipy.stats

    ref = importlib.resources.files('thunor') / 'testdata/hts007.h5'
    with importlib.resources.as_file(ref) as path:
        dataset = read_hdf(path)

    dip_assay = dataset.dip_assay_name
    df_assays = dataset.assays.loc[dip_assay]

    wells = []
    for well_id, grp in df_assays.groupby(level='well_id'):
        t_hours = (
            np.array(grp.index.get_level_values('timepoint').total_seconds())
            / thunor.dip.SECONDS_IN_HOUR
        )
        assay_vals = np.log2(np.array(grp['value']))
        if len(t_hours) >= 3:
            wells.append((t_hours, assay_vals))

    def dip_scipy_reference(t_hours, assay_vals):
        n_total = len(t_hours)
        best = -np.inf
        result = (np.nan,) * 4
        for i in range(n_total - 2):
            x, y = t_hours[i:], assay_vals[i:]
            n = len(x)
            slope, intercept, r, _p, se = scipy.stats.linregress(x, y)
            adj_r2 = 1.0 - (1.0 - r**2) * (n - 1) / (n - 2)
            rmse = np.linalg.norm(x * slope + intercept - y) / n**0.5
            sel = adj_r2 * (1 - rmse) ** 2 * (n - 3) ** 0.25 if n > 3 else 0.0
            if sel > best:
                best = sel
                result = (slope, se, x[0], intercept)
        return result

    slopes_ref, slopes_new = [], []
    for t_h, a_v in wells:
        r_ref = dip_scipy_reference(t_h, a_v)
        r_new = thunor.dip._expt_dip_inner(t_h, a_v)
        slopes_ref.append(r_ref[0])
        slopes_new.append(r_new[0])
        # First timepoint must match exactly
        assert r_ref[2] == pytest.approx(r_new[2], abs=1e-12), (
            f'First timepoint mismatch: ref={r_ref[2]}, new={r_new[2]}'
        )

    np.testing.assert_allclose(slopes_new, slopes_ref, rtol=1e-10)
