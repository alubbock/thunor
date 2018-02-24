import numpy as np
import scipy.stats
import pandas as pd
from .curve_fit import fit_drc, ll4, ll4_initials

SECONDS_IN_HOUR = 3600.0
PARAM_EQUAL_ATOL = 1e-16
PARAM_EQUAL_RTOL = 1e-12
DIP_ASSAYS = ('Cell count', 'lum:Lum')


class ValueWarning(UserWarning):
    pass


class AUCFitWarning(ValueWarning):
    pass


class AAFitWarning(ValueWarning):
    pass


class DrugCombosNotImplementedError(NotImplementedError):
    """ This function does not support drug combinations yet """
    pass


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
    if df_data.controls is None:
        ctrl_dips = None
    else:
        df_controls = df_data.controls.loc[df_data.dip_assay_name]
        ctrl_dips = ctrl_dip_rates(df_controls)
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
    if n_total < 3:
        return None
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


def find_icN(fit_params, ic_num=50):
    """
    Find the inhibitory concentration value (e.g. IC50)

    Parameters
    ----------
    fit_params: Iterable
        List of fit parameters: Hill slope, E0, Emax, EC50
    ic_num: int
        IC number between 0 and 100 (response level)

    Returns
    -------
    float
        Inhibitory concentration value for requested response value
    """
    hill_slope, e0, emax, ec50 = fit_params

    if emax > e0:
        emax, e0 = e0, emax

    ic_frac = ic_num / 100.0

    icN = ec50 * (ic_frac / (1 - ic_frac - (emax / e0))) ** (1 / hill_slope)

    if np.isnan(icN):
        icN = None

    return icN


def find_ecN(fit_params, ec_num=50):
    """
    Find the effective concentration value (e.g. IC50)

    Parameters
    ----------
    fit_params: Iterable
        List of fit parameters: Hill slope, E0, Emax, EC50
    ec_num: int
        EC number between 0 and 100 (response level)

    Returns
    -------
    float
        Effective concentration value for requested response value
    """
    hill_slope, e0, emax, ec50 = fit_params

    if ec_num >= 100:
        return None

    ec_frac = ec_num / 100.0

    return ec50 * (ec_frac / (1 - ec_frac)) ** (1 / hill_slope)


def find_auc(fit_params, min_conc):
    """
    Find the area under the curve

    Parameters
    ----------
    fit_params: Iterable
        List of fit parameters: Hill slope, E0, Emax, EC50
    min_conc: float
        Minimum concentration to consider for fitting the curve

    Returns
    -------
    float
        Area under the curve (AUC) value
    """
    hill_slope, e0, emax, ec50 = fit_params

    if emax > e0:
        emax, e0 = e0, emax
    min_conc_hill = min_conc ** hill_slope
    return (np.log10((ec50 ** hill_slope + min_conc_hill) / min_conc_hill) /
            hill_slope) * ((e0 - emax) / e0)


def find_aa(fit_params, max_conc, emax_obs=None):
    """
    Find the activity area (area over the curve)

    Parameters
    ----------
    fit_params: Iterable
        List of fit parameters: Hill slope, E0, Emax, EC50
    max_conc: float
        Maximum concentration to consider for fitting the curve
    emax_obs: float, optional
        Observed Emax value

    Returns
    -------
    float
        Activity area value
    """
    hill_slope, e0, emax, ec50 = fit_params

    if emax > e0:
        emax, e0 = e0, emax

    if emax_obs is not None:
        emax = emax_obs

    ec50_hill = ec50 ** hill_slope

    return np.log10((ec50_hill + max_conc ** hill_slope) / ec50_hill) \
        * ((e0 - emax) / e0) / hill_slope


def dip_fit_params(ctrl_dip_data, expt_dip_data,
                   hill_fn=ll4,
                   curve_initial_guess_fn=ll4_initials,
                   ctrl_dose_fn=lambda doses: np.min(doses) / 10.0,
                   custom_ic_concentrations=set(),
                   custom_ec_concentrations=set(),
                   custom_e_values=set(),
                   custom_e_rel_values=set(),
                   include_dip_rates=True, include_stats=True):
    """
    Fit dose response curves to DIP rates and calculate statistics

    Parameters
    ----------
    ctrl_dip_data: pd.DataFrame
        Control DIP rates from :func:`dip_rates` or :func:`ctrl_dip_rates`
    expt_dip_data: pd.DataFrame
        Experiment (non-control) DIP rates from :func:`dip_rates` or
        :func:`expt_dip_rates`
    hill_fn: function
        Function to use for curve fitting (default: :func:`ll4`)
    curve_initial_guess_fn: function
        Function to use for initial guesses before curve fitting (default:
        :func:`ll4_initials`)
    ctrl_dose_fn: function
        Function to use to set an effective "dose" (non-zero) for controls.
        Takes the list of experiment doses as an argument.
    custom_ic_concentrations: set
        Set of additional inhibitory concentrations to calculate. Integer 
        values 0-100.
    custom_ec_concentrations: set
        Set of additional effective concentrations to calculate. Integer
        values 0-100.
    custom_e_values: set
        Set of additional effect values to calculate. Integer
        values 0-100.
    custom_e_rel_values: set
        Set of additional relative effect values to calculate. Integer
        values 0-100.
    include_dip_rates: bool
        Include the supplied DIP rates in the return value if True
    include_stats: bool
        Include extra statistics such as IC50 and AUC if True (increases
        processing time)

    Returns
    -------
    pd.DataFrame
        DataFrame containing DIP rate curve fits and parameters
    """

    if 'dataset' in expt_dip_data.index.names:
        datasets = expt_dip_data.index.get_level_values('dataset').unique()
    else:
        datasets = None
    cell_lines = expt_dip_data.index.get_level_values('cell_line').unique()
    drugs = expt_dip_data.index.get_level_values('drug').unique()

    has_drug_combos = drugs.map(len).max() > 1
    if has_drug_combos:
        raise DrugCombosNotImplementedError()
    else:
        # TODO: Support drug combos
        expt_dip_data.reset_index(['drug', 'dose'], inplace=True)
        expt_dip_data['drug'] = expt_dip_data['drug'].apply(pd.Series)
        expt_dip_data['dose'] = expt_dip_data['dose'].apply(pd.Series)
        expt_dip_data.set_index(['drug', 'dose'], append=True,
                                inplace=True)
        drugs = expt_dip_data.index.get_level_values('drug').unique()

    if len(drugs) > 1 and len(cell_lines) == 1:
        group_by = ['drug']
    elif len(cell_lines) > 1 and len(drugs) == 1:
        group_by = ['cell_line']
    else:
        group_by = ['cell_line', 'drug']

    if datasets is not None and len(datasets) > 1:
        if len(cell_lines) == 1 and len(drugs) == 1:
            group_by = ['dataset']
        else:
            group_by = ['dataset'] + group_by

    fit_params = []

    for group_name, dip_grp in expt_dip_data.groupby(level=group_by):
        group_name_components = []

        if 'dataset' in group_by:
            if len(group_by) > 1:
                dataset = group_name[group_by.index('dataset')]
            else:
                dataset = group_name
            group_name_components.append(str(dataset))
        elif datasets is not None:
            dataset = datasets[0]
        else:
            dataset = None

        if 'cell_line' in group_by:
            if len(group_by) > 1:
                cl_name = group_name[group_by.index('cell_line')]
            else:
                cl_name = group_name
            group_name_components.append(str(cl_name))
        else:
            cl_name = cell_lines[0]

        if 'drug' in group_by:
            if len(group_by) > 1:
                dr_name = group_name[group_by.index('drug')]
            else:
                dr_name = group_name
            group_name_components.append(str(dr_name))
        else:
            dr_name = drugs[0]

        group_name_disp = "\n".join(group_name_components)

        try:
            if dataset is None and 'dataset' in ctrl_dip_data.index.names:
                raise ValueError('Experimental data does not have "dataset" '
                                 'in index, but control data does. Please '
                                 'make sure "dataset" is in both dataframes, '
                                 'or neither.')
            if dataset is None:
                ctrl_dip_data_cl = ctrl_dip_data.loc[cl_name]
            else:
                ctrl_dip_data_cl = ctrl_dip_data.loc[dataset, cl_name]
            dip_ctrl = ctrl_dip_data_cl['dip_rate'].values
            dip_ctrl_std_err = ctrl_dip_data_cl['dip_fit_std_err'].values
        except (KeyError, AttributeError):
            ctrl_dip_data_cl = None
            dip_ctrl = []
            dip_ctrl_std_err = []

        n_controls = len(dip_ctrl)

        doses_expt = dip_grp.index.get_level_values('dose').values

        doses_ctrl = np.repeat(ctrl_dose_fn(doses_expt), n_controls)
        doses = np.concatenate((doses_ctrl, doses_expt))
        dip_expt = dip_grp['dip_rate'].values
        dip_all = np.concatenate((dip_ctrl, dip_expt))
        dip_std_errs = np.concatenate((
            dip_ctrl_std_err,
            dip_grp['dip_fit_std_err'].values))

        popt, popt_rel, divisor = fit_drc(
            doses, dip_all, dip_std_errs,
            hill_fn=hill_fn, curve_initial_guess_fn=curve_initial_guess_fn
        )

        max_dose_measured = np.max(doses)
        min_dose_measured = np.min(doses)
        emax = None
        if popt is not None:
            emax = hill_fn(max_dose_measured, *popt)

        emax_obs = np.min(dip_expt)
        emax_obs_rel = None
        if emax_obs and divisor is not None:
            emax_obs_rel = emax_obs / divisor

        emax_rel = None
        if emax is not None and divisor is not None:
            emax_rel = emax / divisor

        ec50 = None if popt is None else np.min((popt[3], max_dose_measured))

        fit_data = dict(
            label=group_name_disp,
            dataset_id=dataset if dataset is not None else '',
            cell_line=cl_name,
            drug=dr_name,
            divisor=divisor,
            popt=popt,
            popt_rel=popt_rel,
            einf=None if popt is None else popt[1],
            emax=emax,
            emax_rel=emax_rel,
            emax_obs=emax_obs,
            emax_obs_rel=emax_obs_rel,
            ec50=ec50,
            max_dose_measured=max_dose_measured,
            hill=None if popt is None else popt[0]
        )

        if include_dip_rates:
            if ctrl_dip_data_cl is not None:
                ctrl_dip_data_cl['dose'] = doses_ctrl
                ctrl_dip_data_cl.reset_index('well_id', inplace=True)
                ctrl_dip_data_cl.set_index(['dose', 'well_id'], inplace=True)
                fit_data['dip_ctrl'] = ctrl_dip_data_cl['dip_rate']

            fit_data['dip_expt'] = pd.Series(
                data=dip_grp['dip_rate'].values,
                index=[doses_expt, dip_grp.index.get_level_values('well_id')]
            )
            fit_data['dip_expt'].index.rename(['dose', 'well_id'],
                                              inplace=True)

        # Only calculate AUC and IC50 if needed
        if include_stats:
            custom_ic_concentrations.add(50)
            for ic_num in custom_ic_concentrations:
                if popt is None:
                    ic_n = None
                else:
                    ic_n = find_icN(popt, ic_num=ic_num)

                if ic_n is not None:
                    fit_data['ic{:d}'.format(ic_num)] = \
                        np.min((ic_n, max_dose_measured))
                else:
                    fit_data['ic{:d}'.format(ic_num)] = None

            custom_ec_concentrations.discard(50)
            custom_ec_concentrations = custom_ec_concentrations.union(
                custom_e_values)
            custom_ec_concentrations = custom_ec_concentrations.union(
                custom_e_rel_values)

            for ec_num in custom_ec_concentrations:
                if popt is None:
                    ec_n = None
                else:
                    ec_n = find_ecN(popt, ec_num=ec_num)

                if ec_n is not None:
                    fit_data['ec{:d}'.format(ec_num)] = \
                        np.min((ec_n, max_dose_measured))
                else:
                    fit_data['ec{:d}'.format(ec_num)] = None

            for e_num in custom_e_values:
                ec_val = fit_data['ec{:d}'.format(e_num)]
                if popt is None or ec_val is None:
                    fit_data['e{:d}'.format(e_num)] = None
                else:
                    fit_data['e{:d}'.format(e_num)] = hill_fn(ec_val, *popt)

            for e_num in custom_e_rel_values:
                ec_val = fit_data['ec{:d}'.format(e_num)]
                if popt_rel is None or ec_val is None:
                    fit_data['e{:d}_rel'.format(e_num)] = None
                else:
                    fit_data['e{:d}_rel'.format(e_num)] = \
                        hill_fn(ec_val, *popt_rel)

            if popt is None or fit_data['ec50'] is None:
                fit_data['aa'] = None
                fit_data['auc'] = None
            else:
                fit_data['aa'] = find_aa(fit_params=popt,
                                         max_conc=max_dose_measured,
                                         emax_obs=emax)
                fit_data['auc'] = find_auc(fit_params=popt,
                                           min_conc=min_dose_measured)
        fit_params.append(fit_data)

    df_params = pd.DataFrame(fit_params)
    df_params.set_index(['dataset_id', 'cell_line', 'drug'], inplace=True)

    return df_params


def is_param_truncated(df_params, param_name):
    """
    Checks if parameter values are truncated at boundaries of measured range

    Parameters
    ----------
    df_params: pd.DataFrame
        DataFrame of DIP curve fits with parameters from :func:`dip_fit_params`
    param_name: str
        Name of a parameter, e.g. 'ic50'

    Returns
    -------
    np.ndarray
        Array of booleans showing whether each entry in the DataFrame is
        truncated
    """
    return np.isclose(df_params[param_name].fillna(value=np.nan),
                      df_params['max_dose_measured'],
                      atol=PARAM_EQUAL_ATOL,
                      rtol=PARAM_EQUAL_RTOL)
