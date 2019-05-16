import numpy as np
import scipy.stats
import pandas as pd

SECONDS_IN_HOUR = 3600.0
DIP_ASSAYS = ('Cell count', 'lum:Lum')


def _choose_dip_assay(assay_names):
    for assay in DIP_ASSAYS:
        if assay in assay_names:
            return assay

    return None


def tyson1(adj_r_sq, rmse, n):
    """
    Tyson1 algorithm for selecting optimal DIP rate fit

    Parameters
    ----------
    adj_r_sq: float
        Adjusted r-squared value
    rmse: float
        Root mean squared error of fit
    n: int
        Number of data points used in fit

    Returns
    -------
    float
        Fit value (higher is better)
    """
    return adj_r_sq * ((1 - rmse) ** 2) * ((n - 3) ** 0.25)


def dip_rates(df_data, selector_fn=tyson1):
    """
    Calculate DIP rates on a dataset

    Parameters
    ----------
    df_data: thunor.io.HtsPandas
        Thunor HTS dataset
    selector_fn: function
        Selection function for choosing optimal DIP rate fit (default:
        :func:`tyson1`

    Returns
    -------
    list
        Two entry list, giving control DIP rates and experiment
        (non-control) DIP rates (both as Pandas DataFrames)
    """
    if df_data.controls is None or df_data.controls.empty:
        ctrl_dips = None
    else:
        if 'dataset' in df_data.controls.index.names:
            df_controls = df_data.controls.loc[(slice(None),
                                                df_data.dip_assay_name), :]
        else:
            df_controls = df_data.controls.loc[df_data.dip_assay_name]
        df_controls = df_controls.loc[df_controls.index.dropna()]
        if df_controls.empty:
            ctrl_dips = None
        else:
            ctrl_dips = ctrl_dip_rates(df_controls)

    if df_data.assays.empty:
        return ctrl_dips, None

    df_assays = df_data.assays.loc[df_data.dip_assay_name]

    return ctrl_dips, \
           expt_dip_rates(df_data.doses, df_assays, selector_fn=selector_fn)


def expt_dip_rates(df_doses, df_vals, selector_fn=tyson1):
    """
    Calculate experiment (non-control) DIP rates

    Parameters
    ----------
    df_doses: pd.DataFrame
        Pandas DataFrame of dose values from a
        :class:`thunor.io.HtsPandas` object
    df_vals: pd.DataFrame
        Pandas DataFrame of cell counts from a :class:`thunor.io.HtsPandas`
        object
    selector_fn: function
        Selection function for choosing optimal DIP rate fit (default:
        :func:`tyson1`

    Returns
    -------
    pd.DataFrame
        Fitted DIP rate values
    """
    res = df_vals.groupby(level='well_id')['value'].\
        apply(_expt_dip, selector_fn=selector_fn).apply(pd.Series).\
        rename(columns={0: 'dip_rate', 1: 'dip_fit_std_err',
                        2: 'dip_first_timepoint', 3: 'dip_y_intercept'})
    dip_df = pd.merge(df_doses, res, left_on='well_id',
                      right_index=True)
    dip_df.set_index('well_id', append=True, inplace=True)
    dip_df.sort_index(inplace=True)
    return dip_df


def _expt_dip(df_timecourses, selector_fn):
    t_hours = np.array(df_timecourses.index.get_level_values(
        level='timepoint').total_seconds()) / SECONDS_IN_HOUR

    assay_vals = np.log2(np.array(df_timecourses))
    n_total = len(t_hours)

    dip = None
    final_std_err = None
    first_timepoint = None
    final_intercept = None
    dip_selector = -np.inf
    if n_total < 2:
        return None
    if n_total == 2:
        # Only two time points, so we can't do variable delay detection
        dip, std_err, intercept = _ctrl_dip(df_timecourses)
        return dip, std_err, t_hours[0], intercept
    for i in range(n_total - 2):
        x = t_hours[i:]
        y = assay_vals[i:]
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
            x, y)

        n = len(x)
        adj_r_sq = adjusted_r_squared(r_value, n, 1)
        predictions = np.add(np.multiply(x, slope), intercept)
        rmse = np.linalg.norm(predictions - y) / np.sqrt(n)
        new_dip_selector = selector_fn(adj_r_sq, rmse, n)
        if new_dip_selector > dip_selector:
            dip_selector = new_dip_selector
            dip = slope
            final_std_err = std_err
            first_timepoint = x[0]
            final_intercept = intercept

    return dip, final_std_err, first_timepoint, final_intercept


def ctrl_dip_rates(df_controls):
    """
    Calculate control DIP rates

    Parameters
    ----------
    df_controls: pd.DataFrame
        Pandas DataFrame of control cell counts from a
        :class:`thunor.io.HtsPandas` object

    Returns
    -------
    pd.DataFrame
        Fitted control DIP rate values
    """
    res = df_controls.groupby(level=('cell_line', 'plate', 'well_id'))[
        'value'].apply(
        _ctrl_dip).apply(pd.Series).\
        rename(columns={0: 'dip_rate', 1: 'dip_fit_std_err',
                        2: 'dip_y_intercept'})

    return res


def _ctrl_dip(df_timecourse):
    t_hours = np.array(df_timecourse.index.get_level_values(
        level='timepoint').total_seconds()) / SECONDS_IN_HOUR

    ctrl_slope, ctrl_intercept, ctrl_r, ctrl_p, ctrl_std_err = \
        scipy.stats.linregress(
            t_hours, np.log2(np.array(df_timecourse)))

    return ctrl_slope, ctrl_std_err, ctrl_intercept


def adjusted_r_squared(r, n, p):
    """
    Calculate adjusted r-squared value from r value

    Parameters
    ----------
    r: float
        r value (between 0 and 1)
    n: int
        number of sample data points
    p: int
        number of free parameters used in fit

    Returns
    -------
    float
        Adjusted r-squared value
    """
    if n <= p:
        return np.nan
    return 1 - (1 - r ** 2) * ((n - 1) / (n - p - 1))
