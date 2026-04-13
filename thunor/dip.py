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
            df_controls = df_data.controls.loc[(slice(None), df_data.dip_assay_name), :]
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

    return ctrl_dips, expt_dip_rates(df_data.doses, df_assays, selector_fn=selector_fn)


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
    if selector_fn is tyson1:
        res = _expt_dip_rates_fast(df_vals)
    else:
        res = (
            df_vals.groupby(level='well_id')['value']
            .apply(_expt_dip, selector_fn=selector_fn)
            .apply(pd.Series)
            .rename(
                columns={
                    0: 'dip_rate',
                    1: 'dip_fit_std_err',
                    2: 'dip_first_timepoint',
                    3: 'dip_y_intercept',
                }
            )
        )
    dip_df = pd.merge(df_doses, res, left_on='well_id', right_index=True)
    dip_df.set_index('well_id', append=True, inplace=True)
    dip_df.sort_index(inplace=True)
    return dip_df


def _expt_dip_rates_fast(df_vals):
    """
    Bulk-vectorised DIP rate fitting for the default tyson1 selector.

    Extracts all well time series in one pass (avoiding per-group pandas
    overhead), runs _expt_dip_inner on each, and assembles the result
    DataFrame directly — bypassing the costly groupby().apply() pipeline.
    """
    well_ids = df_vals.index.get_level_values('well_id').values
    t_seconds = df_vals.index.get_level_values('timepoint').total_seconds().values
    assay_vals_all = np.log2(df_vals['value'].values)

    # Wells are contiguous and sorted; find split boundaries in one pass
    boundaries = np.flatnonzero(well_ids[1:] != well_ids[:-1]) + 1
    split_ids = well_ids[np.concatenate(([0], boundaries))]
    t_splits = np.split(t_seconds, boundaries)
    a_splits = np.split(assay_vals_all, boundaries)

    rows = []
    for well_id, t_raw, a_v in zip(split_ids, t_splits, a_splits):
        t_h = t_raw / SECONDS_IN_HOUR
        n = len(t_h)
        if n < 2:
            rows.append((well_id, np.nan, np.nan, np.nan, np.nan))
        elif n == 2:
            dip, se, intercept = _ctrl_dip_arrays(t_h, a_v)
            rows.append((well_id, dip, se, t_h[0], intercept))
        else:
            dip, se, t0, intercept = _expt_dip_inner(t_h, a_v)
            rows.append((well_id, dip, se, t0, intercept))

    return pd.DataFrame(
        rows,
        columns=[
            'well_id',
            'dip_rate',
            'dip_fit_std_err',
            'dip_first_timepoint',
            'dip_y_intercept',
        ],
    ).set_index('well_id')


def _expt_dip_inner(t_hours, assay_vals):
    """
    Vectorised sliding-window linear regression for DIP rate fitting.

    Evaluates every contiguous suffix of the time series as a candidate
    regression window and returns the best fit according to the tyson1
    selector.  Uses numpy suffix (reverse cumulative) sums so that all
    window statistics are computed in a single vectorised pass — O(n)
    work instead of the O(n²) scipy.stats.linregress-per-window approach.

    All regression quantities (slope, intercept, R², RMSE, std_err) are
    derived algebraically from the five sufficient statistics
    {sx, sy, sxx, sxy, syy}, avoiding any intermediate per-window array
    allocation.
    """
    n_total = len(t_hours)

    # Suffix sums via reverse cumulative sum:
    # suffix_X[i] = X[i] + X[i+1] + ... + X[n-1]
    t2 = t_hours * t_hours
    ta = t_hours * assay_vals
    a2 = assay_vals * assay_vals

    sx = np.cumsum(t_hours[::-1])[::-1]
    sy = np.cumsum(assay_vals[::-1])[::-1]
    sxx = np.cumsum(t2[::-1])[::-1]
    sxy = np.cumsum(ta[::-1])[::-1]
    syy = np.cumsum(a2[::-1])[::-1]

    # Window start indices 0 .. n-3; each window has at least 3 points
    idx = np.arange(n_total - 2)
    ns = n_total - idx  # window length for each start index

    sx = sx[idx]
    sy = sy[idx]
    sxx = sxx[idx]
    sxy = sxy[idx]
    syy = syy[idx]

    # Centred sums of squares
    ssxx = sxx - sx * sx / ns
    ssyy = syy - sy * sy / ns
    ssxy = sxy - sx * sy / ns

    valid = ssxx > 0.0

    # Regression parameters (np.errstate suppresses divide-by-zero warnings
    # that np.where raises when evaluating both branches before masking)
    with np.errstate(divide='ignore', invalid='ignore'):
        slope = np.where(valid, ssxy / ssxx, np.nan)
        intercept = (sy - slope * sx) / ns

        r2 = np.where(valid & (ssyy > 0.0), ssxy * ssxy / (ssxx * ssyy), 0.0)
    adj_r2 = 1.0 - (1.0 - r2) * (ns - 1) / (ns - 2)

    # RMSE via closed-form: SSres = ssyy - slope*ssxy (no per-window arrays)
    resid_ss = np.maximum(ssyy - slope * ssxy, 0.0)
    rmse = np.sqrt(resid_ss / ns)

    # Tyson1 selector
    sel = np.where(
        valid & (ns > 3),
        adj_r2 * (1.0 - rmse) ** 2 * (ns - 3) ** 0.25,
        0.0,
    )

    best_i = int(np.argmax(sel))

    se = (
        (resid_ss[best_i] / ((ns[best_i] - 2) * ssxx[best_i])) ** 0.5
        if valid[best_i]
        else np.nan
    )
    return (
        float(slope[best_i]),
        float(se),
        float(t_hours[idx[best_i]]),
        float(intercept[best_i]),
    )


def _expt_dip(df_timecourses, selector_fn):
    t_hours = (
        np.array(
            df_timecourses.index.get_level_values(level='timepoint').total_seconds()
        )
        / SECONDS_IN_HOUR
    )

    assay_vals = np.log2(np.array(df_timecourses))
    n_total = len(t_hours)

    if n_total < 2:
        return None
    if n_total == 2:
        # Only two time points — no variable delay detection possible
        dip, std_err, intercept = _ctrl_dip(df_timecourses)
        return dip, std_err, t_hours[0], intercept

    if selector_fn is tyson1:
        return _expt_dip_inner(t_hours, assay_vals)

    # Fallback: supports custom selector functions
    dip = None
    final_std_err = None
    first_timepoint = None
    final_intercept = None
    dip_selector = -np.inf
    for i in range(n_total - 2):
        x = t_hours[i:]
        y = assay_vals[i:]
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)

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
    return _ctrl_dip_rates_fast(df_controls)


def _ctrl_dip_rates_fast(df_controls):
    """
    Vectorised implementation of ctrl_dip_rates using prefix-sum regression.

    Avoids per-group Python overhead of groupby().apply() by:
    1. Extracting all time/value arrays from the MultiIndex in one pass
    2. Using prefix sums to compute OLS regression statistics for all
       wells simultaneously with pure numpy operations.
    """
    df_vals = df_controls[['value']].sort_index()

    t_hours = (
        df_vals.index.get_level_values('timepoint').total_seconds().to_numpy()
        / SECONDS_IN_HOUR
    )
    y_all = np.log2(df_vals['value'].to_numpy())
    well_ids = df_vals.index.get_level_values('well_id').to_numpy()
    cl_vals = df_vals.index.get_level_values('cell_line').to_numpy()
    plate_vals = df_vals.index.get_level_values('plate').to_numpy()

    # Find group boundaries (sorted by cell_line, plate, well_id, timepoint).
    # Compare all three key columns to correctly detect boundaries where only
    # cell_line or plate changes (same well_id can appear on different plates).
    boundaries = (
        np.where(
            (well_ids[1:] != well_ids[:-1])
            | (cl_vals[1:] != cl_vals[:-1])
            | (plate_vals[1:] != plate_vals[:-1])
        )[0]
        + 1
    )
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(y_all)]))

    # Prefix sums for vectorised OLS
    cs_t = np.concatenate(([0.0], np.cumsum(t_hours)))
    cs_y = np.concatenate(([0.0], np.cumsum(y_all)))
    cs_t2 = np.concatenate(([0.0], np.cumsum(t_hours * t_hours)))
    cs_ty = np.concatenate(([0.0], np.cumsum(t_hours * y_all)))
    cs_y2 = np.concatenate(([0.0], np.cumsum(y_all * y_all)))

    n_arr = (ends - starts).astype(float)
    sx = cs_t[ends] - cs_t[starts]
    sy = cs_y[ends] - cs_y[starts]
    sxx = cs_t2[ends] - cs_t2[starts]
    sxy = cs_ty[ends] - cs_ty[starts]
    syy = cs_y2[ends] - cs_y2[starts]

    denom = n_arr * sxx - sx * sx
    slopes = (n_arr * sxy - sx * sy) / denom
    intercepts = (sy - slopes * sx) / n_arr
    resid_var = (syy - slopes * sxy - intercepts * sy) / (n_arr - 2.0)
    std_errs = np.sqrt(np.maximum(resid_var, 0.0) / (sxx - sx * sx / n_arr))

    index_tuples = [(cl_vals[s], plate_vals[s], well_ids[s]) for s in starts]
    idx = pd.MultiIndex.from_tuples(
        index_tuples, names=['cell_line', 'plate', 'well_id']
    )
    return pd.DataFrame(
        {
            'dip_rate': slopes,
            'dip_fit_std_err': std_errs,
            'dip_y_intercept': intercepts,
        },
        index=idx,
    )


def _ctrl_dip_arrays(t_hours, assay_vals):
    """Fit a single linear regression on pre-extracted numpy arrays."""
    slope, intercept, _r, _p, std_err = scipy.stats.linregress(t_hours, assay_vals)
    return slope, std_err, intercept


def _ctrl_dip(df_timecourse):
    t_hours = (
        np.array(
            df_timecourse.index.get_level_values(level='timepoint').total_seconds()
        )
        / SECONDS_IN_HOUR
    )

    ctrl_slope, ctrl_intercept, ctrl_r, ctrl_p, ctrl_std_err = scipy.stats.linregress(
        t_hours, np.log2(np.array(df_timecourse))
    )

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
    return 1 - (1 - r**2) * ((n - 1) / (n - p - 1))
