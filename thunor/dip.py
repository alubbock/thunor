import numpy as np
import scipy.stats
import pandas as pd
from .curve_fit import fit_drc, HillCurveLL4, HillCurveNull

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
        if 'dataset' in df_data.controls.index.names:
            df_controls = df_data.controls.loc[(slice(None),
                                                df_data.dip_assay_name), :]
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


def _get_control_responses(ctrl_dip_data, dataset, cl_name, dip_grp):
    if ctrl_dip_data is None:
        return None

    if dataset is not None and 'dataset' in ctrl_dip_data.index.names:
        ctrl_dip_data_cl = ctrl_dip_data.loc[dataset]
    else:
        ctrl_dip_data_cl = ctrl_dip_data

    ctrl_dip_data_cl = ctrl_dip_data_cl.loc[cl_name]

    # Only use controls from the same plates as the expt
    if 'plate' in dip_grp.index.names:
        plates = dip_grp.index.get_level_values('plate').unique().values
    elif 'plate' in dip_grp.columns:
        plates = dip_grp['plate'].unique()
    else:
        plates = []
    ctrl_dip_data_cl = ctrl_dip_data_cl.loc[plates]

    return ctrl_dip_data_cl


def fit_params_minimal(ctrl_data, expt_data,
                       fit_cls=HillCurveLL4,
                       ctrl_dose_fn=lambda doses: np.min(doses) / 10.0):
    """
    Fit dose response curves to DIP or viability, and calculate statistics

    This function only fits curves and stores basic fit parameters. Use
    :func:`dip_fit_params` for more statistics and parameters.

    Parameters
    ----------
    ctrl_data: pd.DataFrame or None
        Control DIP rates from :func:`dip_rates` or :func:`ctrl_dip_rates`.
        Set to None to not use control data.
    expt_data: pd.DataFrame
        Experiment (non-control) DIP rates from :func:`dip_rates` or
        :func:`expt_dip_rates`
    fit_cls: Class
        Class to use for curve fitting (default: :func:`HillCurveLL4`)
    ctrl_dose_fn: function
        Function to use to set an effective "dose" (non-zero) for controls.
        Takes the list of experiment doses as an argument.

    Returns
    -------
    pd.DataFrame
        DataFrame containing DIP rate curve fits and parameters
    """
    expt_data_orig = expt_data

    if 'dataset' in expt_data.index.names:
        datasets = expt_data.index.get_level_values('dataset').unique()
    else:
        datasets = None
    cell_lines = expt_data.index.get_level_values('cell_line').unique()
    drugs = expt_data.index.get_level_values('drug').unique()

    has_drug_combos = drugs.map(len).max() > 1
    if has_drug_combos:
        raise DrugCombosNotImplementedError()
    else:
        # TODO: Support drug combos
        expt_data = expt_data.reset_index(['drug', 'dose'])
        expt_data['drug'] = expt_data['drug'].apply(lambda x: x[0])
        expt_data['dose'] = expt_data['dose'].apply(lambda x: x[0])
        expt_data.set_index(['drug', 'dose'], append=True,
                            inplace=True)
        drugs = expt_data.index.get_level_values('drug').unique()

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

    is_viability = 'viability' in expt_data.columns

    for group_name, dip_grp in expt_data.groupby(level=group_by):
        if 'dataset' in group_by:
            if len(group_by) > 1:
                dataset = group_name[group_by.index('dataset')]
            else:
                dataset = group_name
        elif datasets is not None:
            dataset = datasets[0]
        else:
            dataset = None

        if 'cell_line' in group_by:
            if len(group_by) > 1:
                cl_name = group_name[group_by.index('cell_line')]
            else:
                cl_name = group_name
        else:
            cl_name = cell_lines[0]

        if 'drug' in group_by:
            if len(group_by) > 1:
                dr_name = group_name[group_by.index('drug')]
            else:
                dr_name = group_name
        else:
            dr_name = drugs[0]

        doses_expt = dip_grp.index.get_level_values('dose').values

        if is_viability:
            resp_expt = dip_grp['viability']
            doses = doses_expt
            fit_obj = fit_drc(
                doses_expt, resp_expt, response_std_errs=None,
                fit_cls=fit_cls
            )
        else:
            if dataset is None and ctrl_data is not None and \
                    'dataset' in ctrl_data.index.names:
                raise ValueError('Experimental data does not have "dataset" '
                                 'in index, but control data does. Please '
                                 'make sure "dataset" is in both dataframes, '
                                 'or neither.')

            ctrl_dip_data_cl = \
                _get_control_responses(ctrl_data, dataset, cl_name,
                                       dip_grp)
            dip_ctrl = []
            dip_ctrl_std_err = []
            if ctrl_dip_data_cl is not None:
                if is_viability:
                    dip_ctrl = ctrl_dip_data_cl.values
                else:
                    dip_ctrl = ctrl_dip_data_cl['dip_rate'].values
                    dip_ctrl_std_err = ctrl_dip_data_cl[
                        'dip_fit_std_err'].values

            n_controls = len(dip_ctrl)
            ctrl_dose_val = ctrl_dose_fn(doses_expt)
            doses_ctrl = np.repeat(ctrl_dose_val, n_controls)
            doses = np.concatenate((doses_ctrl, doses_expt))
            resp_expt = dip_grp['dip_rate'].values
            dip_all = np.concatenate((dip_ctrl, resp_expt))
            dip_std_errs = np.concatenate((
                dip_ctrl_std_err,
                dip_grp['dip_fit_std_err'].values))

            fit_obj = fit_drc(
                doses, dip_all, dip_std_errs,
                fit_cls=fit_cls,
                # ctrl_dose=ctrl_dose_val
            )

        max_dose_measured = np.max(doses)
        min_dose_measured = np.min(doses)

        fit_data = dict(
            dataset_id=dataset if dataset is not None else '',
            cell_line=cl_name,
            drug=dr_name,
            fit_obj=fit_obj,
            min_dose_measured=min_dose_measured,
            max_dose_measured=max_dose_measured,
            emax_obs=np.min(resp_expt)
        )

        fit_params.append(fit_data)

    df_params = pd.DataFrame(fit_params)
    df_params.set_index(['dataset_id', 'cell_line', 'drug'], inplace=True)

    df_params._drmetric = 'viability' if is_viability else 'dip'
    if is_viability:
        df_params._viability_time = expt_data_orig._viability_time
        df_params._viability_assay = expt_data_orig._viability_assay

    return df_params


def _attach_extra_params(base_params,
                         custom_ic_concentrations=None,
                         custom_ec_concentrations=None,
                         custom_e_values=None,
                         custom_e_rel_values=None,
                         extra_stats=True):
    datasets = base_params.index.get_level_values('dataset_id').unique()
    if len(datasets) == 1 and datasets[0] == '':
        datasets = None

    cell_lines = base_params.index.get_level_values('cell_line').unique()
    drugs = base_params.index.get_level_values('drug').unique()

    if len(drugs) > 1 and len(cell_lines) == 1:
        group_by = ['drug']
    elif len(cell_lines) > 1 and len(drugs) == 1:
        group_by = ['cell_line']
    else:
        group_by = ['cell_line', 'drug']

    if datasets is not None and len(datasets) > 1:
        if len(cell_lines) == 1 and len(drugs) == 1:
            group_by = ['dataset_id']
        else:
            group_by = ['dataset_id'] + group_by

    is_viability = base_params._drmetric == 'viability'

    extra_params = []

    for group_name, dip_grp in base_params.groupby(level=group_by, sort=False):
        group_name_components = []

        if 'dataset_id' in group_by:
            if len(group_by) > 1:
                dataset = group_name[group_by.index('dataset_id')]
            else:
                dataset = group_name
            group_name_components.append(str(dataset))

        if 'cell_line' in group_by:
            if len(group_by) > 1:
                cl_name = group_name[group_by.index('cell_line')]
            else:
                cl_name = group_name
            group_name_components.append(str(cl_name))

        if 'drug' in group_by:
            if len(group_by) > 1:
                dr_name = group_name[group_by.index('drug')]
            else:
                dr_name = group_name
            group_name_components.append(str(dr_name))

        group_name_disp = "\n".join(group_name_components)

        fit_obj = dip_grp.fit_obj.item()
        max_dose_measured = dip_grp.max_dose_measured.item()
        min_dose_measured = dip_grp.min_dose_measured.item()
        emax_obs = dip_grp.emax_obs.item()

        emax = None
        divisor = None
        if fit_obj is not None and not isinstance(fit_obj, HillCurveNull):
            emax = fit_obj.fit(max_dose_measured)
            divisor = fit_obj.divisor

        emax_obs_rel = None
        if emax_obs and divisor is not None:
            emax_obs_rel = emax_obs / divisor

        emax_rel = None
        if emax is not None and divisor is not None:
            emax_rel = emax / divisor

        ec50 = None
        hill = None
        if fit_obj is not None and not isinstance(fit_obj, HillCurveNull):
            ec50 = np.min((fit_obj.ec50, max_dose_measured))
            ec50 = np.max((ec50, min_dose_measured))
            hill = fit_obj.hill_slope

        fit_data = dict(
            label=group_name_disp,
            divisor=divisor,
            hill=hill,
            emax=emax,
            emax_rel=emax_rel,
            emax_obs_rel=emax_obs_rel,
            ec50=ec50
        )

        # Only calculate AUC and IC50 if needed
        if extra_stats:
            if custom_ic_concentrations is None:
                custom_ic_concentrations = set()
            if custom_ec_concentrations is None:
                custom_ec_concentrations = set()
            if custom_e_values is None:
                custom_e_values = set()
            if custom_e_rel_values is None:
                custom_e_rel_values = set()

            custom_ic_concentrations.add(50)
            for ic_num in custom_ic_concentrations:
                if fit_obj is None:
                    ic_n = None
                else:
                    ic_n = fit_obj.ic(ic_num=ic_num)

                if ic_n is not None:
                    ic_n = np.min((ic_n, max_dose_measured))
                    ic_n = np.max((ic_n, min_dose_measured))
                    fit_data['ic{:d}'.format(ic_num)] = ic_n
                else:
                    fit_data['ic{:d}'.format(ic_num)] = None

            custom_ec_concentrations.discard(50)
            custom_ec_concentrations = custom_ec_concentrations.union(
                custom_e_values)
            custom_ec_concentrations = custom_ec_concentrations.union(
                custom_e_rel_values)

            for ec_num in custom_ec_concentrations:
                if fit_obj is None:
                    ec_n = None
                else:
                    ec_n = fit_obj.ec(ec_num=ec_num)

                if ec_n is not None:
                    ec_n = np.min((ec_n, max_dose_measured))
                    ec_n = np.max((ec_n, min_dose_measured))
                    fit_data['ec{:d}'.format(ec_num)] = ec_n
                else:
                    fit_data['ec{:d}'.format(ec_num)] = None

            for e_num in custom_e_values:
                ec_val = fit_data['ec{:d}'.format(e_num)]
                if fit_obj is None or ec_val is None:
                    fit_data['e{:d}'.format(e_num)] = None
                else:
                    fit_data['e{:d}'.format(e_num)] = fit_obj.fit(ec_val)

            for e_num in custom_e_rel_values:
                ec_val = fit_data['ec{:d}'.format(e_num)]
                if fit_obj is None or ec_val is None:
                    fit_data['e{:d}_rel'.format(e_num)] = None
                else:
                    fit_data['e{:d}_rel'.format(e_num)] = \
                        fit_obj.fit_rel(ec_val)

            if fit_obj is None or fit_data['ec50'] is None:
                fit_data['aa'] = None
                fit_data['auc'] = None
            else:
                fit_data['aa'] = fit_obj.aa(max_conc=max_dose_measured,
                                            emax_obs=emax)
                fit_data['auc'] = fit_obj.auc(min_conc=min_dose_measured)
        extra_params.append(fit_data)

    extra_params = pd.DataFrame(extra_params)

    extra_params.index = base_params.index
    df_params = pd.concat([base_params, extra_params], axis=1)

    df_params._drmetric = base_params._drmetric
    if is_viability:
        df_params._viability_time = base_params._viability_time
        df_params._viability_assay = base_params._viability_assay

    return df_params


def _attach_response_values(df_params, ctrl_dip_data, expt_dip_data,
                            ctrl_dose_fn):
    is_viability = df_params._drmetric == 'viability'
    data_list = []
    for grp, dip_grp in expt_dip_data.groupby(
            ['dataset', 'cell_line', 'drug'], sort=False):
        # Assumes drug combinations have been ruled out by fit_params_minimal
        doses_expt = [d[0] for d in dip_grp.index.get_level_values(
            'dose').values]
        fit_data = {}
        ctrl_dip_data_cl = \
            _get_control_responses(ctrl_dip_data, grp[0], grp[1],
                                   dip_grp)
        if ctrl_dip_data_cl is not None:
            if is_viability:
                ctrl_dip_data_cl = ctrl_dip_data_cl.to_frame()
            n_controls = len(ctrl_dip_data_cl.index)
            ctrl_dose_val = ctrl_dose_fn(doses_expt)
            doses_ctrl = np.repeat(ctrl_dose_val, n_controls)
            ctrl_dip_data_cl['dose'] = doses_ctrl
            ctrl_dip_data_cl.reset_index('well_id', inplace=True)
            ctrl_dip_data_cl.set_index(['dose', 'well_id'], inplace=True)
            if is_viability:
                fit_data['viability_ctrl'] = ctrl_dip_data_cl['value']
            else:
                fit_data['dip_ctrl'] = ctrl_dip_data_cl['dip_rate']

        if is_viability:
            fit_data['viability_time'] = dip_grp['timepoint'].values
            fit_data['viability'] = pd.Series(
                data=dip_grp['viability'].values,
                index=[doses_expt, dip_grp.index.get_level_values(
                    'well_id')]
            )
            fit_data['viability'].index.rename(['dose', 'well_id'],
                                               inplace=True)
        else:
            fit_data['dip_expt'] = pd.Series(
                data=dip_grp['dip_rate'].values,
                index=[doses_expt,
                       dip_grp.index.get_level_values('well_id')]
            )
            fit_data['dip_expt'].index.rename(['dose', 'well_id'],
                                              inplace=True)

        data_list.append(fit_data)

    df = pd.DataFrame(data_list)
    df.index = df_params.index
    df_params_old = df_params
    df_params = pd.concat([df, df_params], axis=1)

    df_params._drmetric = df_params_old._drmetric
    if is_viability:
        df_params._viability_time = df_params_old._viability_time
        df_params._viability_assay = df_params_old._viability_assay

    return df_params


def dip_fit_params(ctrl_dip_data, expt_dip_data,
                   fit_cls=HillCurveLL4,
                   ctrl_dose_fn=lambda doses: np.min(doses) / 10.0,
                   custom_ic_concentrations=None,
                   custom_ec_concentrations=None,
                   custom_e_values=None,
                   custom_e_rel_values=None,
                   include_response_values=True,
                   extra_stats=True):
    """
    Fit dose response curves to DIP rates and calculate statistics

    Parameters
    ----------
    ctrl_dip_data: pd.DataFrame or None
        Control DIP rates from :func:`dip_rates` or :func:`ctrl_dip_rates`.
        Set to None to not use control data.
    expt_dip_data: pd.DataFrame
        Experiment (non-control) DIP rates from :func:`dip_rates` or
        :func:`expt_dip_rates`
    fit_cls: Class
        Class to use for curve fitting (default: :func:`HillCurveLL4`)
    ctrl_dose_fn: function
        Function to use to set an effective "dose" (non-zero) for controls.
        Takes the list of experiment doses as an argument.
    custom_ic_concentrations: set, optional
        Set of additional inhibitory concentrations to calculate. Integer
        values 0-100. Requires extra_stats=True.
    custom_ec_concentrations: set, optional
        Set of additional effective concentrations to calculate. Integer
        values 0-100. Requires extra_stats=True.
    custom_e_values: set, optional
        Set of additional effect values to calculate. Integer
        values 0-100. Requires extra_stats=True.
    custom_e_rel_values: set, optional
        Set of additional relative effect values to calculate. Integer
        values 0-100. Requires extra_stats=True.
    include_response_values: bool
        Include the supplied DIP rates in the return value if True
    extra_stats: bool
        Include extra statistics such as IC50 and AUC if True (increases
        processing time)

    Returns
    -------
    pd.DataFrame
        DataFrame containing DIP rate curve fits and parameters
    """
    base_params = fit_params_minimal(ctrl_dip_data, expt_dip_data, fit_cls,
                                     ctrl_dose_fn)

    return fit_params_from_base(
        base_params, ctrl_dip_data, expt_dip_data,
        ctrl_dose_fn=ctrl_dose_fn,
        custom_ic_concentrations=custom_ic_concentrations,
        custom_ec_concentrations=custom_ec_concentrations,
        custom_e_values=custom_e_values,
        custom_e_rel_values=custom_e_rel_values,
        include_response_values=include_response_values,
        extra_stats=extra_stats
    )


def fit_params_from_base(
        base_params,
        ctrl_resp_data=None, expt_resp_data=None,
        ctrl_dose_fn=lambda doses: np.min(doses) / 10.0,
        custom_ic_concentrations=None,
        custom_ec_concentrations=None,
        custom_e_values=None,
        custom_e_rel_values=None,
        include_response_values=True,
        extra_stats=True):
    """
    Attach additional parameters to basic set of fit parameters
    """
    df_params = _attach_extra_params(base_params, custom_ic_concentrations,
                                     custom_ec_concentrations,
                                     custom_e_values, custom_e_rel_values,
                                     extra_stats)

    if include_response_values:
        df_params = _attach_response_values(df_params, ctrl_resp_data,
                                            expt_resp_data, ctrl_dose_fn)

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
    values = df_params[param_name].fillna(value=np.nan)
    return np.isclose(
        values, df_params['max_dose_measured'],
        atol=PARAM_EQUAL_ATOL, rtol=PARAM_EQUAL_RTOL) | np.isclose(
        values, df_params['min_dose_measured'],
        atol=PARAM_EQUAL_ATOL, rtol=PARAM_EQUAL_RTOL
    )
