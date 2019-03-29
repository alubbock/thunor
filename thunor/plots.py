import plotly.graph_objs as go
import numpy as np
import seaborn as sns
from .helpers import format_dose
from .dip import ctrl_dip_rates, expt_dip_rates
from thunor.curve_fit import HillCurveNull, is_param_truncated
import scipy.stats
import re
import pandas as pd
import collections


class CannotPlotError(ValueError):
    pass


def _activity_area_units(**kwargs):
    if 'aa_max_conc' in kwargs:
        return '[max. dose={}]'.format(format_dose(kwargs['aa_max_conc']))
    else:
        return ''


def _auc_units(**kwargs):
    if 'auc_min_conc' in kwargs:
        return '[min. dose={}]'.format(format_dose(kwargs['auc_min_conc']))
    else:
        return ''


SECONDS_IN_HOUR = 3600.0
NS_IN_SEC = 1e9
PLATE_MAP_WELL_DIAM = 0.95
ASCII_CAP_A = 65
PARAM_UNITS = {'auc': _activity_area_units,
               'aa': _activity_area_units,
               'einf': 'h<sup>-1</sup>',
               'emax': 'h<sup>-1</sup>',
               'emax_obs': 'h<sup>-1</sup>'}
PARAM_NAMES = {'aa': 'Activity area',
               'aa_obs': 'Activity area (observed)',
               'aa_num': 'Activity area (numerical integration)',
               'auc': 'Area under curve',
               'einf': 'E<sub>inf</sub>',
               'emax': 'E<sub>max</sub>',
               'emax_rel': 'E<sub>max</sub> (relative)',
               'emax_obs': 'E<sub>max</sub> observed',
               'emax_obs_rel': 'E<sub>Max</sub> observed (relative)',
               'hill': 'Hill coefficient'}
IC_REGEX = re.compile('^ic([0-9]+)$')
EC_REGEX = re.compile('^ec([0-9]+)$')
E_REGEX = re.compile('^e([0-9]+)$')
E_REL_REGEX = re.compile('^e([0-9]+)_rel$')


def _secs_to_str(seconds):
    """
    Convert seconds to HH:MM:SS string

    Parameters
    ----------
    seconds: int
        Number of seconds

    Returns
    -------
    str
        Formatted string like "Time: 2:04:00"
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "Time: %d:%02d:%02d" % (h, m, s)


def _remove_drmetric_prefix(param_id):
    """ Remove DR metric prefix on parameter name, if present """
    for prefix in ('viability__', 'dip__'):
        if param_id.startswith(prefix):
            param_id = param_id[len(prefix):]

    return param_id


def _param_is_log(param_id):
    param_id = _remove_drmetric_prefix(param_id)
    return (IC_REGEX.match(param_id) or EC_REGEX.match(param_id) or
            param_id == 'hill')


def _param_na_first(param_id):
    """ NAs go in the position corresponding to no effect """
    param_id = _remove_drmetric_prefix(param_id)
    # Which is first for E, Emax, Erel, AA and Hill
    return param_id in ('hill', 'aa', 'emax', 'emax_rel', 'einf') \
           or E_REGEX.match(param_id) \
           or E_REL_REGEX.match(param_id)


def _get_param_name(param_id):
    param_id = _remove_drmetric_prefix(param_id)
    try:
        return PARAM_NAMES[param_id]
    except KeyError:
        if IC_REGEX.match(param_id):
            return 'IC<sub>{:d}</sub>'.format(int(param_id[2:]))
        elif EC_REGEX.match(param_id):
            return 'EC<sub>{:d}</sub>'.format(int(param_id[2:]))
        elif E_REGEX.match(param_id):
            return 'E<sub>{:d}</sub>'.format(int(param_id[1:]))
        elif E_REL_REGEX.match(param_id):
            return 'E<sub>{:d}</sub> (relative)'.format(
                int(param_id[1:param_id.index('_')]))
        else:
            return param_id


def _get_param_units(param_id):
    param_id = _remove_drmetric_prefix(param_id)
    try:
        return PARAM_UNITS[param_id]
    except KeyError:
        if IC_REGEX.match(param_id) or EC_REGEX.match(param_id):
            return 'M'
        elif E_REGEX.match(param_id):
            return 'h<sup>-1</sup>'
        else:
            return ''


def _out_of_range_msg(param_id):
    return '{} truncated to limit of measured concentrations'.format(
        _get_param_name(param_id)
    )


def _sns_to_rgb(palette):
    return ['rgb(%d, %d, %d)' % (c[0] * 255, c[1] * 255, c[2] * 255)
            if not isinstance(c, str) else c
            for c in palette]


def _make_title(title, df):
    drug_list = df.index.get_level_values('drug').unique()
    if len(drug_list) == 1:
        drug_name = drug_list[0]
        if not isinstance(drug_name, str):
            if isinstance(drug_name, collections.Iterable):
                if len(drug_name) == 1:
                    drug_name = drug_name[0]
                else:
                    drug_name = " &amp; ".join(drug_name)
            else:
                raise ValueError('Unknown drug_name type: {}'.format(type(
                    drug_name)))
        title += ' for {}'.format(drug_name)

    cell_line_list = df.index.get_level_values('cell_line').unique()
    if len(cell_line_list) == 1:
        title += ' on {}'.format(cell_line_list[0])

    return title


def _combine_title_subtitle(title, subtitle):
    if subtitle:
        title += '<br> <span style="color:#999;font-size:0.9em">' \
                 '{}</span>'.format(subtitle)

    return title


def plot_drc(fit_params, is_absolute=False, color_by=None, color_groups=None,
             title=None, subtitle=None):
    """
    Plot dose response curve fits

    Parameters
    ----------
    fit_params: pd.DataFrame
        Fit parameters from :func:`thunor.curve_fit.fit_params`
    is_absolute: bool
        For DIP rate plots, use absolute (True) or relative (False)
        y-axis scale. **Ignored for viability plots.**
    color_by: str or None
        Color the traces by cell lines if 'cl', drugs if 'dr',
        or arbitrarily if None (default)
    color_groups: dict or None
        If using color_by, provide a dictionary containing the color groups,
        where the values are cell line or drug names
    title: str, optional
        Title (or None to auto-generate)
    subtitle: str, optional
        Subtitle (or None to auto-generate)

    Returns
    -------
    plotly.graph_objs.Figure
        A plotly figure object containing the graph
    """
    datasets = fit_params.index.get_level_values('dataset_id').unique()
    if color_by:
        num_colors = len(color_groups)
    elif len(datasets) == 2:
        num_colors = 2
    else:
        num_colors = len(fit_params)

    colours = _sns_to_rgb(sns.color_palette("husl", num_colors))
    color_index_col = None
    if color_by == 'cl':
        color_index_col = fit_params.index.names.index('cell_line')
    elif color_by == 'dr':
        color_index_col = fit_params.index.names.index('drug')
    elif color_by is None:
        if len(datasets) == 2:
            color_by = 'dataset'
            color_index_col = fit_params.index.names.index('dataset_id')
            color_groups = {dataset: [dataset] for dataset in datasets}
    else:
        raise ValueError('color_by must be "cl", "dr" or None')
    # Shapes used for replicate markers
    shapes = ['circle', 'circle-open']

    try:
        is_viability = 'viability' in fit_params.columns or \
                       fit_params._drmetric == 'viability'
    except AttributeError:
        is_viability = False

    if is_viability:
        # Only "absolute" (non-transformed y-axis) makes sense for viability
        is_absolute = True

    if is_viability:
        yaxis_title = '{:g} hr viability'.format(fit_params._viability_time)
    else:
        yaxis_title = 'DIP rate'
    if is_absolute:
        if not is_viability:
            yaxis_title += ' (h<sup>-1</sup>)'
    else:
        yaxis_title = 'Relative ' + yaxis_title

    multi_dataset = len(datasets) > 1
    show_annotations = len(fit_params.index) == 1

    show_replicates = len(fit_params.index) == 1 or \
        (multi_dataset and
         len(fit_params.index.get_level_values('cell_line').unique()) == 1 and
         len(fit_params.index.get_level_values('drug').unique()) == 1)

    if title is None:
        title = _make_title('Dose response', fit_params)
    if subtitle is None:
        subtitle = " &amp; ".join(str(d) for d in datasets)
    title = _combine_title_subtitle(title, subtitle)

    if show_annotations:
        fit_params = fit_params.copy()
        fit_params['ec50_truncated'] = is_param_truncated(fit_params, 'ec50')
        fit_params['ic50_truncated'] = is_param_truncated(fit_params, 'ic50')

    annotations = []
    traces = []
    if color_by:
        for idx, name in enumerate(color_groups):
            traces.append(go.Scatter(mode='none', x=[0], y=[0],
                                     legendgroup=name,
                                     showlegend=True,
                                     name='<b>{}</b>'.format(name))
                          )
    xaxis_min = np.Inf
    xaxis_max = np.NINF
    for fp in fit_params.itertuples():
        if color_by:
            grp = fp.Index[color_index_col]
            this_colour = None
            legend_grp = None
            for idx, color_group in enumerate(color_groups.items()):
                grp_label, grp_entries = color_group
                # Not an efficient lookup, but for reasonable number of
                # traces it's fine
                if grp in grp_entries:
                    legend_grp = grp_label
                    this_colour = colours[idx]
                    break
            if this_colour is None:
                raise ValueError('"{}" is not in the color_groups'.format(grp))
        else:
            this_colour = colours.pop()
            legend_grp = fp.label

        group_name_disp = fp.label

        log_dose_min = int(np.floor(np.log10(fp.min_dose_measured)))
        log_dose_max = int(np.ceil(np.log10(fp.max_dose_measured)))

        try:
            if is_viability:
                ctrl_doses = fp.viability_ctrl.index.get_level_values('dose')
            else:
                ctrl_doses = fp.dip_ctrl.index.get_level_values('dose')

            log_dose_min = min(log_dose_min, int(np.floor(np.log10(np.min(
                ctrl_doses.values)))))
        except AttributeError:
            ctrl_doses = []

        xaxis_max = max(xaxis_max, log_dose_max)
        xaxis_min = min(xaxis_min, log_dose_min)

        try:
            if is_viability:
                expt_doses = fp.viability.index.get_level_values('dose')
            else:
                expt_doses = fp.dip_expt.index.get_level_values('dose')
        except AttributeError:
            expt_doses = []

        dose_x_range = np.concatenate(
            # [np.arange(2, 11) * 10 ** dose_mag
            [0.5 * np.arange(3, 21) * 10 ** dose_mag
             for dose_mag in range(log_dose_min, log_dose_max + 1)],
            axis=0)

        dose_x_range = np.append([10 ** log_dose_min], dose_x_range,
                                 axis=0)

        line_dash = 'solid'
        line_mode = 'lines'
        hoverinfo = 'all'
        visible = True
        if fp.fit_obj is None:
            # Curve fit numerical error or QC failure
            dose_x_range = None
            dip_rate_fit = None
            line_mode = 'none'
            group_name_disp = '<i>{}</i>'.format(
                group_name_disp)
            hoverinfo = 'none'
            visible = 'legendonly'
        elif isinstance(fp.fit_obj, HillCurveNull):
            # No effect null hypothesis
            dip_rate_fit = [1 if not is_absolute else fp.fit_obj.divisor] * \
                            len(dose_x_range)
            line_dash = 'longdash'
        else:
            # Fit succeeded
            if is_absolute:
                dip_rate_fit = fp.fit_obj.fit(dose_x_range)
            else:
                dip_rate_fit = fp.fit_obj.fit_rel(dose_x_range)

        traces.append(go.Scatter(x=dose_x_range,
                                 y=dip_rate_fit,
                                 mode=line_mode,
                                 line={'shape': 'spline',
                                       'color': this_colour,
                                       'dash': line_dash,
                                       'width': 3},
                                 hoverinfo=hoverinfo,
                                 legendgroup=legend_grp,
                                 showlegend=not show_replicates or
                                            multi_dataset,
                                 visible=visible,
                                 name=group_name_disp)
                      )

        if show_replicates:
            y_trace = fp.viability if is_viability else fp.dip_expt
            try:
                ctrl_resp = fp.viability_ctrl if is_viability else fp.dip_ctrl
            except AttributeError:
                ctrl_resp = None
            if not is_absolute:
                if fp.fit_obj is not None:
                    divisor = fp.fit_obj.divisor
                elif fp.fit_obj is None and ctrl_resp is not None:
                    divisor = np.mean(ctrl_resp)
                else:
                    divisor = 1
                y_trace /= divisor
                if ctrl_resp is not None:
                    ctrl_resp /= divisor

            repl_name = 'Replicate'
            ctrl_name = 'Control'
            if multi_dataset:
                repl_name = '{} {}'.format(fp.Index[0], repl_name)
                ctrl_name = '{} {}'.format(fp.Index[0], ctrl_name)

            if is_viability:
                # viability times are in nanoseconds - convert
                hoverlabels = [_secs_to_str(int(x) / 1e9) for x in
                               fp.viability_time]
            else:
                hoverlabels = repl_name

            shape = shapes.pop(0)

            traces.append(go.Scatter(x=expt_doses,
                                     y=y_trace,
                                     mode='markers',
                                     marker={'symbol': shape,
                                             'color': this_colour,
                                             'size': 5},
                                     legendgroup=group_name_disp,
                                     hoverinfo='x+y+text',
                                     text=hoverlabels,
                                     showlegend=False,
                                     name=repl_name)
                          )
            if ctrl_resp is not None:
                traces.append(go.Scatter(x=ctrl_doses,
                                         y=ctrl_resp,
                                         mode='markers',
                                         marker={'symbol': shape,
                                                 'color': 'black',
                                                 'size': 5},
                                         hoverinfo='y+text',
                                         text=ctrl_name,
                                         name=ctrl_name,
                                         legendgroup=group_name_disp,
                                         showlegend=False)
                              )

        if show_annotations:
            annotation_label = ''
            if fp.ec50 is not None:
                annotation_label += 'EC<sub>50</sub>{}: {} '.format(
                    '*' if fp.ec50_truncated else '',
                    format_dose(fp.ec50, sig_digits=5)
                )
            if fp.ic50 is not None:
                annotation_label += 'IC<sub>50</sub>{}: {} '.format(
                    '*' if fp.ic50_truncated else '',
                    format_dose(fp.ic50, sig_digits=5)
                )
            if fp.emax is not None:
                annotation_label += 'E<sub>max{}</sub>{}: {:.5g}'.format(
                    ' rel' if not is_absolute else '',
                    '*' if fp.fit_obj.emax < fp.emax else '',
                    fp.emax if is_absolute else fp.emax_rel)
            if annotation_label:
                hovermsgs = []
                hovertext = None
                if fp.ec50_truncated:
                    hovermsgs.append(_out_of_range_msg('ec50'))
                if fp.ic50_truncated:
                    hovermsgs.append(_out_of_range_msg('ic50'))
                if hovermsgs:
                    hovertext = '*' + '<br>'.join(hovermsgs)
                annotations.append({
                    'x': 0.5,
                    'y': 1.0,
                    'xref': 'paper',
                    'yanchor': 'bottom',
                    'yref': 'paper',
                    'showarrow': False,
                    'hovertext': hovertext,
                    'text': annotation_label
                })

    yaxis_range = None
    yaxis_rangemode = 'tozero' if is_viability else 'normal'
    if not is_absolute:
        yaxis_range = (-0.2, 1.2)
    elif not is_viability:
        yaxis_range = (-0.02, 0.07)

    layout = go.Layout(title=title,
                       hovermode='closest' if show_replicates
                                 or len(traces) > 50 else 'x',
                       xaxis={'title': 'Dose (M)',
                              'range': (xaxis_min, xaxis_max),
                              'type': 'log'},
                       yaxis={'title': yaxis_title,
                              'range': yaxis_range,
                              'rangemode': yaxis_rangemode
                              },
                       annotations=annotations,
                       )

    return go.Figure(data=traces, layout=layout)


def plot_drug_combination_heatmap(ctrl_resp_data, expt_resp_data,
                                  title=None, subtitle=None):
    """
    Plot heatmap of drug combination response by DIP rate

    Two dimensional plot (each dimension is a drug concentration) where
    squares are coloured by DIP rate value.

    Parameters
    ----------
    ctrl_resp_data: pd.DataFrame
        Control DIP rates from :func:`thunor.dip.dip_rates`
    expt_resp_data: pd.DataFrame
        Experiment (non-control) DIP rates from :func:`thunor.dip.dip_rates`
    title: str, optional
        Title (or None to auto-generate)
    subtitle: str, optional
        Subtitle (or None to auto-generate)

    Returns
    -------
    plotly.graph_objs.Figure
        A plotly figure object containing the graph
    """
    heat_label = 'DIP<br>rate'

    if title is None:
        title = _make_title('Dose response', expt_resp_data)

    if subtitle is None:
        datasets = expt_resp_data.index.get_level_values('dataset').unique()
        subtitle = datasets[0]

    title = _combine_title_subtitle(title, subtitle)

    expt_resp_data = expt_resp_data['dip_rate']
    if ctrl_resp_data is None:
        raise CannotPlotError('There are no matching control wells for this '
                              'selection, so relative DIP rate cannot be '
                              'calculated')

    expt_resp_data = expt_resp_data / ctrl_resp_data['dip_rate'].mean()
    heat_label = 'Relative<br>' + heat_label

    expt_resp_data = expt_resp_data.reset_index(['dose', 'drug'])
    doses = expt_resp_data['dose'].apply(pd.Series)
    doses.columns = ['dose1', 'dose2']

    drugs = expt_resp_data['drug'].apply(pd.Series)
    drugs.columns = ['drug1', 'drug2']
    drug1 = drugs['drug1'].unique()[0]
    drug2 = drugs['drug2'].unique()[0]

    expt_resp_data = pd.concat([doses, drugs,
                                expt_resp_data['dip_rate']], axis=1)

    expt_resp_data = expt_resp_data.set_index(['dose1', 'dose2'])

    dat = []
    dose1 = sorted(expt_resp_data.index.get_level_values('dose1').unique())
    dose2 = sorted(expt_resp_data.index.get_level_values('dose2').unique())
    for d1, grp in expt_resp_data['dip_rate'].groupby('dose1'):
        dat2 = []
        grp.index = grp.index.droplevel()
        for d2 in dose2:
            try:
                dat2.append(grp.loc[d2].mean())
            except KeyError:
                if d1 == 0 and d2 == 0:
                    # Control well relative DIP rate is 1.0 by definition
                    dat2.append(1.0)
                else:
                    dat2.append(None)
        dat.append(dat2)

    trace = go.Heatmap(x=format_dose(dose2), y=format_dose(dose1), z=dat,
                       colorbar={'title': heat_label}, zmin=-1.5, zmax=1.5,
                       colorscale=[
                           (0, 'rgb(255,0,0)'),
                           (0.5, 'rgb(255,255,0)'),
                           (1, 'rgb(0,0,255)')
                       ])

    layout = go.Layout(
        title=title,
        xaxis={
            'title': '{} concentration'.format(drug2)
        },
        yaxis={
            'title': '{} concentration'.format(drug1)
        }
    )

    return go.Figure(data=[trace], layout=layout)


def _symbols_hovertext_two_dataset_scatter(df_params, range_bounded_params,
                                           fit_param, dataset_names):
    symbols = ['circle'] * len(df_params.index)
    hovertext = [" ".join(l) for l in df_params.index.values]
    for param in range_bounded_params:
        msg = _out_of_range_msg(param)
        for i in (0, 1):
            tmp_df = pd.concat([
                df_params.loc[:, fit_param].iloc[:, i],
                df_params.loc[:, 'max_dose_measured'].iloc[:, i],
                df_params.loc[:, 'min_dose_measured'].iloc[:, i]
            ], axis=1)
            tmp_df.columns = [fit_param,
                              'max_dose_measured',
                              'min_dose_measured']
            param_truncated = is_param_truncated(tmp_df, fit_param)
            addtxt = ['<br> {} {}'.format(dataset_names[i], msg) if x else
                      '' for x in param_truncated]
            hovertext = [ht + at for ht, at in zip(hovertext, addtxt)]
            symbols = ['cross' if x else old for x, old in
                       zip(param_truncated, symbols)]

    return symbols, hovertext


def plot_two_dataset_param_scatter(df_params, fit_param, title, subtitle,
                                   color_by, color_groups, **kwargs):
    """
    Plot a parameter comparison across two datasets

    Parameters
    ----------
    df_params: pd.DataFrame
        DIP fit parameters from :func:`thunor.dip.dip_params`
    fit_param: str
        The name of the parameter to compare across datasets, e.g. ic50
    title: str, optional
        Title (or None to auto-generate)
    subtitle: str, optional
        Subtitle (or None to auto-generate)
    kwargs: dict, optional
        Additional keyword arguments

    Returns
    -------
    plotly.graph_objs.Figure
        A plotly figure object containing the graph
    """
    if title is None:
        title = _make_title('Dose response parameters', df_params)
    if subtitle is None:
        datasets = df_params.index.get_level_values('dataset_id').unique()
        subtitle = " &amp; ".join(str(d) for d in datasets)
    title = _combine_title_subtitle(title, subtitle)

    if df_params._drmetric == 'dip':
        dr_metric = 'DIP'
    else:
        dr_metric = '{:g} hr viability'.format(df_params._viability_time)

    df_params = df_params.loc[:, [fit_param,
                                  'max_dose_measured',
                                  'min_dose_measured']]
    df_params.dropna(subset=[fit_param], inplace=True)
    df_params.reset_index(inplace=True)
    df_params = df_params.pivot_table(index=['cell_line', 'drug'], columns=[
        'dataset_id'])
    df_params.dropna(inplace=True)

    if len(df_params.index) == 0:
        raise CannotPlotError(
            'Dataset vs dataset scatter plot is empty. Check the cell lines '
            'and drugs in the two datasets overlap. If you want a bar plot '
            'instead, choose an ordering parameter.')

    if len(df_params.columns) < 4:
        raise CannotPlotError(
            'The cell lines and/or drugs selected only have data in one of '
            'the two datasets, so a scatter plot cannot be created. If you '
            'want a bar plot instead, choose an ordering parameter.')

    if color_by:
        colours = _sns_to_rgb(sns.color_palette("husl", len(color_groups)))
    else:
        colours = _sns_to_rgb(sns.color_palette("Paired"))[0:2]

    param_name = _get_param_name(fit_param)
    param_units = _get_param_units(fit_param)
    try:
        param_units = param_units(**kwargs)
    except TypeError:
        pass
    if param_units:
        axis_title = '{} ({})'.format(param_name, param_units)
    else:
        axis_title = param_name

    axis_title = '{} {}'.format(dr_metric, axis_title)

    fit_param_data = df_params.loc[:, fit_param]

    range_bounded_params = set()

    match = IC_REGEX.match(fit_param)
    if match:
        range_bounded_params.add(match.group())
    match = EC_REGEX.match(fit_param)
    if match:
        range_bounded_params.add(match.group())
    match = E_REGEX.match(fit_param)
    if match:
        range_bounded_params.add('ec' + match.groups(0)[0])
    match = E_REL_REGEX.match(fit_param)
    if match:
        range_bounded_params.add('ec' + match.groups(0)[0])

    dataset_names = fit_param_data.columns

    symbols, hovertext = _symbols_hovertext_two_dataset_scatter(
        df_params, range_bounded_params, fit_param, dataset_names
    )

    xdat = fit_param_data.iloc[:, 0]
    xdat_fit = xdat
    ydat = fit_param_data.iloc[:, 1]
    ydat_fit = ydat
    if _param_is_log(fit_param):
        xdat_fit = np.log10(xdat)
        ydat_fit = np.log10(ydat)

    fitdat_mask = (~np.isnan(xdat_fit) & np.isfinite(xdat_fit) &
                   ~np.isnan(ydat_fit) & np.isfinite(ydat_fit) &
                   [s != 'cross' for s in symbols])
    xdat_fit = xdat_fit[fitdat_mask].values
    ydat_fit = ydat_fit[fitdat_mask].values

    data = []
    layout = go.Layout(title=title)

    slope, intercept, r_value, p_value, std_err = \
        scipy.stats.linregress(xdat_fit, ydat_fit)
    if not np.isnan(slope):
        xfit = (min(xdat_fit), max(xdat_fit))
        yfit = [x * slope + intercept for x in xfit]
        if _param_is_log(fit_param):
            xfit = np.power(10, xfit)
            yfit = np.power(10, yfit)
        data.append(go.Scatter(
            x=xfit,
            y=yfit,
            mode='lines',
            hoverinfo="none",
            line=dict(
                color="darkorange"
            ),
            name='{} vs {} {} Linear Fit'.format(dataset_names[0],
                                      dataset_names[1],
                                      param_name),
            showlegend=False
        ))
        layout['annotations'] = [{
            'x': 0.5, 'y': 1.0, 'xref': 'paper', 'yanchor': 'bottom',
            'yref': 'paper', 'showarrow': False,
            'text': 'R<sup>2</sup>: {:0.4g} '
                    'p-value: {:0.4g} '.format(r_value ** 2, p_value)
        }]

    custom_data = [
        {'c': cl, 'd': dr} for cl, dr in zip(
            df_params.index.get_level_values('cell_line'),
            df_params.index.get_level_values('drug')
        )
    ]

    if color_by:
        for idx, tag_name in enumerate(color_groups):
            dat = df_params[df_params.index.get_level_values(
                'cell_line' if color_by == 'cl' else 'drug').isin(
                color_groups[tag_name])]
            symbols, hovertext = _symbols_hovertext_two_dataset_scatter(dat, range_bounded_params,
                                                                        fit_param, dataset_names)

            fit_param_data = dat.loc[:, fit_param]
            xdat = fit_param_data.iloc[:, 0]
            ydat = fit_param_data.iloc[:, 1]

            data.append(go.Scatter(
                x=xdat,
                y=ydat,
                hovertext=hovertext,
                hoverinfo="text+x+y",
                mode='markers',
                customdata=custom_data,
                marker={'symbol': symbols,
                        'color': colours[idx]},
                name=tag_name
            ))
    else:
        colour_list = [colours[1] if s == 'circle' else 'crimson' for s in
                       symbols]

        data.append(go.Scatter(
            x=xdat,
            y=ydat,
            hovertext=hovertext,
            hoverinfo="text+x+y",
            mode='markers',
            customdata=custom_data,
            marker={'symbol': symbols,
                    'color': colour_list},
            name='{} vs {} {}'.format(dataset_names[0],
                                      dataset_names[1],
                                      param_name)
        ))

    layout['xaxis'] = {'title': '{} {}'.format(dataset_names[0], axis_title),
                       'type': 'log' if _param_is_log(fit_param) else None}
    layout['yaxis'] = {'title': '{} {}'.format(dataset_names[1], axis_title),
                       'type': 'log' if _param_is_log(fit_param) else None}
    layout['hovermode'] = 'closest'
    layout['showlegend'] = color_by is not None

    return go.Figure(layout=layout, data=data)


def _symbols_hovertext_two_param_scatter(df_params,
                                         range_bounded_params):
    hovertext = df_params['label']
    symbols = ['circle'] * len(df_params)
    for param in range_bounded_params:
        msg = _out_of_range_msg(param)
        param_truncated = is_param_truncated(df_params, param)
        addtxt = ['<br> ' + msg if x else '' for x in
                  param_truncated]
        hovertext = [ht + at for ht, at in zip(hovertext, addtxt)]
        symbols = ['cross' if x else old for x, old in
                   zip(param_truncated, symbols)]
    return symbols, hovertext


def plot_drc_params(df_params, fit_param,
                    fit_param_compare=None,
                    fit_param_sort=None,
                    title=None,
                    subtitle=None,
                    aggregate_cell_lines=False,
                    aggregate_drugs=False,
                    multi_dataset=False,
                    color_by=None,
                    color_groups=None,
                    **kwargs):
    """
    Box, bar, or scatter plots of DIP rate fit parameters

    Parameters
    ----------
    df_params: pd.DataFrame
        DIP fit parameters from :func:`thunor.dip.dip_params`
    fit_param: str
        Fit parameter name, e.g. 'ic50'
    fit_param_compare: str, optional
        Second fit parameter name for comparative plots, e.g. 'ec50'
    fit_param_sort: str, optional
        Fit parameter name to use for sorting the x-axis, if different from
        fit_param
    title: str, optional
        Title (or None to auto-generate)
    subtitle: str, optional
        Subtitle (or None to auto-generate)
    aggregate_cell_lines: bool or dict, optional
        Aggregate all cell lines (if True), or aggregate by the specified
        groups (dict of cell line names as values, with group labels as keys)
    aggregate_drugs: bool or dict, optional
        Aggregate all drugs (if True), or aggregate by the specified
        groups (dict of drug names as values, with group labels as keys)
    multi_dataset: bool
        Set to true to compare two datasets contained in fit_params
    color_by: str or None
        Color by cell lines if "cl", drugs if "dr", or arbitrarily if None
        (default)
    color_groups: dict or None
        Groups of cell lines of drugs to color by
    kwargs: dict, optional
        Additional keyword arguments

    Returns
    -------
    plotly.graph_objs.Figure
        A plotly figure object containing the graph
    """
    if fit_param_compare and (aggregate_cell_lines or aggregate_drugs):
        raise CannotPlotError(
            'Aggregation is not available when comparing two dose response '
            'parameters')

    if multi_dataset and fit_param_compare is None and \
            not aggregate_cell_lines and \
            not aggregate_drugs and fit_param_sort is None:
        return plot_two_dataset_param_scatter(
            df_params,
            fit_param,
            title,
            subtitle,
            color_by,
            color_groups,
            **kwargs
        )

    color_by_col = None
    if multi_dataset and not color_by:
        color_by_col = 'dataset_id'
        color_by = 'dataset'
        color_groups = {dataset: [dataset] for dataset in df_params.index.get_level_values('dataset_id').unique()}
        colours = _sns_to_rgb(sns.color_palette("husl", 2))
    elif color_by:
        color_by_col = 'cell_line' if color_by == 'cl' else 'drug'
        colours = _sns_to_rgb(sns.color_palette("husl", len(color_groups)))
    else:
        colours = _sns_to_rgb(sns.color_palette("Paired"))[0:2]

    if title is None:
        title = _make_title('Dose response parameters', df_params)
    if subtitle is None:
        datasets = df_params.index.get_level_values('dataset_id').unique()
        subtitle = " &amp; ".join(str(d) for d in datasets)
    title = _combine_title_subtitle(title, subtitle)

    yaxis_param_name = _get_param_name(fit_param)
    yaxis_units = _get_param_units(fit_param)
    try:
        yaxis_units = yaxis_units(**kwargs)
    except TypeError:
        pass
    if yaxis_units:
        yaxis_title = '{} ({})'.format(yaxis_param_name, yaxis_units)
    else:
        yaxis_title = yaxis_param_name
    if df_params._drmetric in ('dip', 'compare'):
        yaxis_title = 'DIP {}'.format(yaxis_title)
    else:
        yaxis_title = '{:g} hr viability {}'.format(df_params._viability_time,
                                                     yaxis_title)

    layout = dict(title=title,
                  yaxis={'title': yaxis_title,
                         'type': 'log' if _param_is_log(fit_param) else None})

    if fit_param_compare:
        df_params.dropna(subset=[fit_param, fit_param_compare],
                         inplace=True)
        if df_params.empty:
            raise CannotPlotError(
                'No data exists for this selection. This may be due to '
                'missing drug/cell line combinations, or undefined parameters '
                'for the selection.')
        if fit_param == fit_param_compare:
            xdat = df_params.loc[:, fit_param]
            ydat = xdat
        else:
            dat = df_params.loc[:, [fit_param_compare, fit_param]]
            xdat = dat[fit_param_compare]
            ydat = dat[fit_param]

        xaxis_param_name = _get_param_name(fit_param_compare)
        xaxis_units = _get_param_units(fit_param_compare)
        try:
            xaxis_units = xaxis_units(**kwargs)
        except TypeError:
            pass
        if xaxis_units:
            xaxis_title = '{} ({})'.format(xaxis_param_name, xaxis_units)
        else:
            xaxis_title = xaxis_param_name
        if df_params._drmetric == 'dip':
            xaxis_title = 'DIP {}'.format(xaxis_title)
        else:
            xaxis_title = '{:g} hr viability {}'.format(
                df_params._viability_time,
                xaxis_title)

        range_bounded_params = set()

        for param in (fit_param, fit_param_compare):
            match = IC_REGEX.match(param)
            if match:
                range_bounded_params.add(match.group())
            match = EC_REGEX.match(param)
            if match:
                range_bounded_params.add(match.group())
            match = E_REGEX.match(param)
            if match:
                range_bounded_params.add('ec' + match.groups(0)[0])
            match = E_REL_REGEX.match(param)
            if match:
                range_bounded_params.add('ec' + match.groups(0)[0])

        symbols, hovertext = _symbols_hovertext_two_param_scatter(
            df_params, range_bounded_params)

        xdat_fit = np.log10(xdat) if _param_is_log(fit_param_compare) else xdat
        ydat_fit = np.log10(ydat) if _param_is_log(fit_param) else ydat

        data = []

        if len(xdat_fit) > 0:
            # Remove any infinite or NaN values from best fit line data
            fitdat_mask = (~np.isnan(xdat_fit) & np.isfinite(xdat_fit) &
                           ~np.isnan(ydat_fit) & np.isfinite(ydat_fit) &
                           [s != 'cross' for s in symbols])
            xdat_fit = xdat_fit[fitdat_mask].values
            ydat_fit = ydat_fit[fitdat_mask].values

        if len(xdat_fit) > 0:
            slope, intercept, r_value, p_value, std_err = \
                scipy.stats.linregress(xdat_fit, ydat_fit)

            xfit = (min(xdat_fit), max(xdat_fit))
            yfit = [x * slope + intercept for x in xfit]
            if _param_is_log(fit_param_compare):
                xfit = np.power(10, xfit)
            if _param_is_log(fit_param):
                yfit = np.power(10, yfit)
            data.append(go.Scatter(
                x=xfit,
                y=yfit,
                mode='lines',
                hoverinfo="none",
                line=dict(
                    color="darkorange"
                ),
                name='{} vs {} Linear Fit'.format(xaxis_param_name,
                                                  yaxis_param_name),
                showlegend=False
            ))
            layout['annotations'] = [{
                'x': 0.5, 'y': 1.0, 'xref': 'paper', 'yanchor': 'bottom',
                'yref': 'paper', 'showarrow': False,
                'text': 'R<sup>2</sup>: {:0.4g} '
                        'p-value: {:0.4g} '.format(r_value ** 2, p_value)
            }]

        custom_data = [
            {'c': cl, 'd': dr} for cl, dr in zip(
                df_params.index.get_level_values('cell_line'),
                df_params.index.get_level_values('drug')
            )
        ]

        if color_by:
            for idx, tag_name in enumerate(color_groups):
                location = df_params.index.get_level_values(color_by_col).isin(color_groups[tag_name])
                dat = df_params[location]
                symbols, hovertext = _symbols_hovertext_two_param_scatter(
                    dat, range_bounded_params)

                xdat = dat.loc[:, fit_param_compare]
                ydat = dat.loc[:, fit_param]

                data.append(go.Scatter(
                    x=xdat,
                    y=ydat,
                    hovertext=hovertext,
                    hoverinfo="text+x+y",
                    mode='markers',
                    customdata=custom_data,
                    marker={'symbol': symbols,
                            'color': colours[idx]},
                    name=tag_name
                ))
        else:
            colour_list = [colours[1] if s == 'circle' else 'crimson' for s in
                           symbols]

            data.append(go.Scatter(
                x=xdat,
                y=ydat,
                hovertext=hovertext,
                hoverinfo="text+x+y",
                mode='markers',
                customdata=custom_data,
                marker={'symbol': symbols,
                        'color': colour_list},
                name='{} vs {}'.format(xaxis_param_name,
                                       yaxis_param_name)
            ))

        layout['xaxis'] = {'title': xaxis_title,
                           'type': 'log' if _param_is_log(fit_param_compare)
                           else None}
        layout['hovermode'] = 'closest'
        layout['showlegend'] = color_by is not None
    elif not aggregate_cell_lines and not aggregate_drugs:
        if fit_param_sort is None:
            sort_by = [fit_param, 'label']
        elif fit_param_sort == 'label':
            sort_by = ['label', fit_param]
            fit_param_sort = None
        else:
            sort_by = [fit_param_sort, 'label']
        df_params = df_params.sort_values(
            by=sort_by, na_position='first' if _param_na_first(sort_by[0])
            else 'last')
        groups = df_params['label']
        yvals = df_params[fit_param]

        text = None

        ec_match = None
        for regex in (EC_REGEX, E_REGEX, E_REL_REGEX):
            match = regex.match(fit_param)
            if match:
                ec_match = 'ec' + match.groups(0)[0]
                break

        marker_cols = None
        if not color_by:
            marker_cols = colours[1]

        if ec_match:
            msg = _out_of_range_msg(ec_match)
            if not fit_param.startswith('ec'):
                msg = 'Based on ' + msg
            ec_truncated = is_param_truncated(df_params, ec_match)
            text = [msg if x else '' for x in ec_truncated]
            if not color_by:
                marker_cols = [colours[0] if est else colours[1] for
                               est in ec_truncated]
        elif IC_REGEX.match(fit_param):
            msg = _out_of_range_msg(fit_param)
            ic_truncated = is_param_truncated(df_params, fit_param)
            text = [msg if x else '' for x in ic_truncated]
            if not color_by:
                marker_cols = [colours[0] if est else colours[1] for
                               est in ic_truncated]

        if color_by:
            color_ent = df_params.index.get_level_values(color_by_col)

            marker_cols = []
            for c in color_ent:
                for idx, tag_name in enumerate(color_groups):
                    if c in color_groups[tag_name]:
                        marker_cols.append(colours[idx])
                        break
                else:
                    raise ValueError('Entity not found: {}'.format(c))

        if fit_param_sort is not None:
            na_list = df_params[fit_param_sort].isnull()
            if text is None:
                text = [''] * len(df_params)
            for idx in range(len(df_params)):
                if na_list.iloc[idx]:
                    if text[idx]:
                        text[idx] += '<br>'
                    text[idx] += '{} undefined, sorted by {}'.format(
                        _get_param_name(fit_param_sort),
                        _get_param_name(fit_param)
                    )

        custom_data = [
            {'c': cl, 'd': dr} for cl, dr in zip(
                df_params.index.get_level_values('cell_line'),
                df_params.index.get_level_values('drug')
            )
        ]

        data = [go.Bar(x=groups,
                       y=yvals,
                       text=text,
                       name='',
                       customdata=custom_data,
                       showlegend=False,
                       marker={'color': marker_cols}
                       )]

        layout['annotations'] = []

        if color_by:
            # Nasty cludge to get legend to show, by stacking dummy traces
            # with zero height (Plotly doesn't support legend by colour at
            # this time)
            for idx, tag in enumerate(color_groups.items()):
                tag_name, tag_targets = tag
                tag_targets = set(tag_targets)
                data.append(go.Bar(
                    x=groups,
                    y=[c in tag_targets for c in color_ent],
                    text=tag_name,
                    name=tag_name,
                    hoverinfo='none',
                    showlegend=True,
                    marker={'color': colours[idx]}
                ))

            # Mann Whitney U test w/ cont. correction, two-sided
            if len(color_groups) == 2 and fit_param_sort is None:
                group1 = yvals[(y == colours[0] for y in marker_cols)]
                group2 = yvals[(y != colours[0] for y in marker_cols)]
                # Need more than 20 entries in each group, as per scipy docs
                if len(group1) > 20 and len(group2) > 20:
                    mw_u, mw_p = scipy.stats.mannwhitneyu(
                        group1,
                        group2,
                        use_continuity=True,
                        alternative='two-sided'
                    )
                    if not np.isnan(mw_u):
                        layout['annotations'].append({
                            'x': 0.5, 'y': 0.95, 'xref': 'paper',
                            'yanchor': 'bottom',
                            'yref': 'paper', 'showarrow': False,
                            'text': 'Two-sided Mann-Whitney U: {:.4g} '
                                    'p-value: {:.4g}'.format(
                                mw_u, mw_p)
                        })

        layout['annotations'].extend([
            {'x': x, 'y': 0, 'text': '<em>N/A</em>',
             'textangle': 90,
             'xanchor': 'center', 'yanchor': 'bottom',
             'yref': 'paper',
             'showarrow': False,
             'font': {'color': 'rgba(150, 150, 150, 1)'}}
            for x in groups[yvals.isnull().values]])
        layout.setdefault('xaxis', {})['type'] = 'category'
        layout['barmode'] = 'stack'
        layout['showlegend'] = color_by is not None
    else:
        layout['boxmode'] = 'group'

        if color_by and color_by != 'dataset':
            raise CannotPlotError(
                'Custom color schemes are not currently supported with box '
                'plots. Either disable the coloring, or turn off aggregation.'
            )

        if fit_param_sort == fit_param:
            fit_param_sort = None

        if fit_param_sort is None or fit_param_sort == 'label':
            yvals = df_params.loc[:, [fit_param]].dropna()
        else:
            yvals = df_params.loc[:, [fit_param,
                                      fit_param_sort]]
            yvals.dropna(subset=[fit_param], inplace=True)

        if yvals.empty:
            raise CannotPlotError(
                'The selected cell line/drug combinations have no data '
                'available for the selected parameter(s)')

        if aggregate_cell_lines:
            yvals = _aggregate_by_cell_line(yvals, aggregate_cell_lines,
                                            replace_index=True)

        if aggregate_drugs:
            yvals = _aggregate_by_drug(yvals, aggregate_drugs,
                                       replace_index=True)

        drug_groups = yvals.index.get_level_values('drug').unique()
        cell_line_groups = yvals.index.get_level_values('cell_line').unique()

        if len(cell_line_groups) > 1:
            aggregate_by = ['dataset_id', 'cell_line']
            groups = drug_groups
        else:
            groups = cell_line_groups
            aggregate_by = ['dataset_id', 'drug']

        # Add sort character '' to 'Everything else' tag
        SORT_AT_END_CHAR = ''
        vals = yvals.index.levels[yvals.index.names.index(
            aggregate_by[1])].values
        vals = [SORT_AT_END_CHAR + y if y.startswith('Everything else (')
                else y for y in vals]
        yvals.index = yvals.index.set_levels(vals, level=aggregate_by[1])

        # Sort by median effect per drug set, or cell line set if there's
        # only one drug/drug group
        if fit_param_sort is None or fit_param_sort == 'label':
            if fit_param_sort == 'label':
                sort_cols = aggregate_by + ['median']
            else:
                sort_cols = ['median'] + aggregate_by
            yvals['median'] = yvals[fit_param].groupby(
                level=aggregate_by).transform(np.nanmedian)
            yvals.set_index('median', append=True, inplace=True)
            yvals.sort_index(level=sort_cols, ascending=True,
                             inplace=True)
            yvals.reset_index('median', drop=True, inplace=True)
        else:
            # Sort by fit_column_sort, with tie breakers determined by
            # fit_param
            median_cols = yvals.loc[:, [fit_param_sort, 'label']].groupby(
                level=aggregate_by).transform(np.nanmedian)
            median_cols.rename(columns={fit_param_sort: 'median',
                                        'label': 'median2'},
                               inplace=True)
            yvals = pd.concat([yvals, median_cols], axis=1)
            yvals.set_index(['median', 'median2'], append=True, inplace=True)
            yvals.sort_index(level=['median', 'median2'] + aggregate_by,
                             ascending=True, inplace=True)
            yvals.reset_index(['median', 'median2'], drop=True, inplace=True)

        # Remove sort character '' from 'Everything else' tag
        vals = yvals.index.levels[yvals.index.names.index(
            aggregate_by[1])].values
        vals = [y.replace(SORT_AT_END_CHAR, '') for y in vals]
        yvals.index = yvals.index.set_levels(vals, level=aggregate_by[1])

        # Convert yvals to a series
        yvals = yvals.iloc[:, 0]

        data = []
        aggregate_by.remove('dataset_id')
        aggregate_by = aggregate_by[0]

        datasets = yvals.index.get_level_values('dataset_id').unique()

        group_by = ['dataset_id'] if len(datasets) > 1 else []
        group_by += ['drug'] if aggregate_by == 'cell_line' else ['cell_line']

        for grp_name, grp in yvals.groupby(level=group_by):
            if len(groups) > 1:
                group_name = grp_name if isinstance(grp_name, str) else \
                    "<br>".join(str(g) for g in grp_name)
            else:
                # If there's only one drug/cell line group, just need dataset
                group_name = grp_name[0] if grp_name else None
            data.append(go.Box(x=grp.index.get_level_values(aggregate_by),
                               y=grp,
                               name=group_name
                               ))

        layout['annotations'] = []
        if len(groups) == 1:
            annotation_label = str(groups[0])
            layout['annotations'].append({
                'x': 0.5, 'y': 1.0, 'xref': 'paper', 'yanchor': 'bottom',
                'yref': 'paper', 'showarrow': False, 'text': annotation_label
            })

        # One way anova test
        anova_f, anova_p = scipy.stats.f_oneway(
            *[x[1].values for x in yvals.groupby(level=aggregate_by)]
        )
        if not np.isnan(anova_f):
            layout['annotations'].append({
                'x': 0.5, 'y': 0.95, 'xref': 'paper',
                'yanchor': 'bottom',
                'yref': 'paper', 'showarrow': False,
                'text': 'One-way ANOVA F: {:.4g} p-value: {:.4g}'.format(
                    anova_f, anova_p)
            })

    layout = go.Layout(layout)

    return go.Figure(data=data, layout=layout)


def _aggregate_by_drug(yvals, aggregate_drugs, replace_index=True):
    return _aggregate_by_tag(yvals, aggregate_drugs, 'drug',
                             replace_index=replace_index)


def _aggregate_by_cell_line(yvals, aggregate_cell_lines,
                            replace_index=True):
    return _aggregate_by_tag(yvals, aggregate_cell_lines,
                             'cell_line', replace_index=replace_index)


def _aggregate_by_tag(yvals, aggregate_items, label_type,
                      replace_index=True, add_counts=True):
    if aggregate_items in (None, False):
        return yvals

    if aggregate_items is True:
        items = yvals.index.get_level_values(
            level=label_type).unique().tolist()
        aggregate_items = {
            _create_label_max_items(items, 5): items
        }

    new = pd.DataFrame()

    label_type_tag = label_type + '_tag'

    for tag_name, names in aggregate_items.items():
        yvals_tmp = yvals.loc[yvals.index.isin(names, level=label_type), :]
        # Avoid warning about setting on copy, since we're using the copy to
        # build a new dataframe
        yvals_tmp.is_copy = None

        # Add counts to the tag names
        if add_counts:
            tag_name = '{} ({})'.format(tag_name, len(
                yvals_tmp.index.get_level_values(label_type).unique()))

        yvals_tmp[label_type_tag] = np.repeat(tag_name, len(yvals_tmp))
        new = new.append(yvals_tmp)

    labels = list(new.index.names)
    new.reset_index([l for l in labels if l != label_type], inplace=True)
    labels[labels.index(label_type)] = label_type_tag
    new.set_index(labels, inplace=True, drop=replace_index)
    if replace_index:
        new.index.rename(label_type, level=label_type_tag, inplace=True)

    return new


def _create_label_max_items(items, max_items=5):
    n = len(items)
    if n > max_items:
        items = items[0:max_items]
    annotation_label = ", ".join(items)
    if n > max_items:
        annotation_label += " and {} more".format(n - len(items))
    return annotation_label


def plot_time_course(hts_pandas,
                     log_yaxis=False, assay_name='Assay', title=None,
                     subtitle=None, show_dip_fit=False):
    """
    Plot a dose response time course

    Parameters
    ----------
    hts_pandas: HtsPandas
        Dataset containing a single cell line/drug combination
    log_yaxis: bool
        Use log scale on y-axis
    assay_name: str
        The name of the assay to use for the time course (only used for
        multi-assay datasets)
    title: str, optional
        Title (or None to auto-generate)
    subtitle: str, optional
        Subtitle (or None to auto-generate)
    show_dip_fit: bool
        Overlay the DIP rate fit on the time course

    Returns
    -------
    plotly.graph_objs.Figure
        A plotly figure object containing the graph
    """
    if show_dip_fit and not log_yaxis:
        raise ValueError('log_yaxis must be True when show_dip_fit is True')

    df_doses = hts_pandas.doses
    if hts_pandas.controls is not None:
        df_controls = hts_pandas.controls
    else:
        df_controls = None
    df_vals = hts_pandas.assays

    df_assays_avail = hts_pandas.assay_names
    if len(df_assays_avail) == 1:
        assay = df_assays_avail[0]
    elif assay_name in df_assays_avail:
        assay = assay_name
    else:
        raise ValueError('{} is not a valid assay. Options are {}'.format(
            assay_name, df_assays_avail))

    if df_controls is not None:
        if 'dataset' in df_controls.index.names:
            dsets = df_controls.index.get_level_values('dataset').unique()
            if len(dsets) > 1:
                raise ValueError('Multiple control datasets present. '
                                 'Plotting a time course requires a single '
                                 'dataset.')
            df_controls = df_controls.loc[dsets[0]]
        df_controls = df_controls.loc[assay]

        # Only use controls from the plates in the expt data
        plates = df_doses['plate'].unique()
        df_controls = df_controls.loc[(slice(None), plates), :]
    df_vals = df_vals.loc[assay]

    if len(hts_pandas.drugs) > 1 or len(hts_pandas.cell_lines) > 1:
        raise ValueError('Dataset has multiple drugs and/or cell lines. Time '
                         'courses are currently only available for single '
                         'drug/cell line combinations.')

    traces = []
    traces_fits = []

    if title is None:
        title = _make_title('Time course', df_doses)
    title = _combine_title_subtitle(title, subtitle)

    colours = _sns_to_rgb(sns.color_palette(
        "husl", len(df_doses.index.get_level_values(level='dose').unique())))

    if show_dip_fit:
        if df_controls is not None:
            dip_rate_ctrl = ctrl_dip_rates(df_controls)
            dip_rate_ctrl.index = dip_rate_ctrl.index.droplevel(level='cell_line')
        dip_rates = expt_dip_rates(df_doses, df_vals)
        dip_rates.reset_index(inplace=True)
        dip_rates.set_index('well_id', inplace=True)

    # Controls
    if df_controls is not None:
        is_first_control = True
        for well_id, timecourse in df_controls.groupby(level='well_id'):
            timecourse = timecourse['value']
            t0_offset = 0
            if log_yaxis:
                timecourse = np.log2(timecourse)
                t0_offset = timecourse[0]
                timecourse -= t0_offset
            x_range = [t.total_seconds() / SECONDS_IN_HOUR for t in
                       timecourse.index.get_level_values('timepoint')]
            traces.append(go.Scatter(
                x=x_range,
                y=timecourse,
                mode='lines+markers',
                line={'color': 'black',
                      'shape': 'spline',
                      'dash': 'dot' if show_dip_fit else None},
                marker={'size': 5},
                name='Control',
                legendgroup='__Control',
                showlegend=is_first_control
            ))
            is_first_control = False

            if show_dip_fit and df_controls is not None:
                dip_well = dip_rate_ctrl.loc[
                    dip_rate_ctrl.index.get_level_values('well_id') == well_id]
                minmax = [np.min(x_range), np.max(x_range)]

                dip_points = [x * dip_well['dip_rate'] +
                              dip_well['dip_y_intercept'] - t0_offset
                              for x in minmax]

                traces_fits.append(go.Scatter(
                    x=minmax,
                    y=dip_points,
                    mode='lines',
                    line={'color': 'black'},
                    marker={'size': 5},
                    name='DIP fit Control',
                    legendgroup='__Control',
                    showlegend=False
                ))

    # Experiment (non-control)
    for dose, wells in df_doses.groupby(level='dose'):
        this_colour = colours.pop()

        for well_idx, well_id in enumerate(wells['well_id']):
            try:
                timecourse = df_vals.loc[well_id]['value']
            except KeyError:
                continue
            t0_offset = 0
            if log_yaxis:
                timecourse = np.log2(timecourse)
                t0_offset = timecourse[0]
                timecourse -= t0_offset
            x_range = [t.total_seconds() / SECONDS_IN_HOUR for t in
                       timecourse.index.get_level_values('timepoint')]
            dose_str = format_dose(dose, array_as_string=" &amp; ")
            traces.append(go.Scatter(
                x=x_range,
                y=timecourse,
                mode='lines+markers',
                line={'color': this_colour,
                      'shape': 'spline',
                      'dash': 'dot' if show_dip_fit else None},
                marker={'size': 5},
                name=dose_str,
                customdata=({'csvname': dose}, ),
                legendgroup=dose_str,
                showlegend=well_idx == 0
            ))

            if show_dip_fit:
                dip_well = dip_rates.loc[well_id]
                minmax = [dip_well['dip_first_timepoint'], np.max(x_range)]
                dip_points = [x*dip_well['dip_rate'] +
                              dip_well['dip_y_intercept'] - t0_offset
                              for x in minmax]

                traces_fits.append(go.Scatter(
                    x=minmax,
                    y=dip_points,
                    mode='lines',
                    line={'color': this_colour},
                    marker={'size': 5},
                    name=dose_str,
                    customdata=({'csvname': 'DIP fit ' + dose_str}, ),
                    legendgroup=dose_str,
                    showlegend=False
                ))

    data = (traces + traces_fits)
    if log_yaxis:
        assay_name = "Change in log<sub>2</sub> {}".format(assay_name)
    max_time = df_vals.index.get_level_values('timepoint').max()
    layout = go.Layout(title=title,
                       xaxis={'title': 'Time (hours)',
                              'range': (0, 120) if max_time <=
                                       np.timedelta64(120, 'h') else None,
                              'dtick': 12},
                       yaxis={'title': assay_name,
                              'range': (-2, 7) if log_yaxis else None},
                       )
    return go.Figure(data=data, layout=layout)


def plot_ctrl_dip_by_plate(df_controls, title=None, subtitle=None):
    """

    Parameters
    ----------
    df_controls: pd.DataFrame
        Control well DIP values
    title: str, optional
        Title (or None to auto-generate)
    subtitle: str, optional
        Subtitle (or None to auto-generate)

    Returns
    -------
    plotly.graph_objs.Figure
        A plotly figure object containing the graph
    """
    # Sort by median DIP rate
    df_controls = df_controls.copy()
    df_controls['cl_median'] = df_controls['dip_rate'].groupby(
        level=['cell_line']).transform(np.nanmedian)
    df_controls['plate_median'] = df_controls['dip_rate'].groupby(
        level=['cell_line', 'plate']).transform(np.nanmedian)
    df_controls.sort_values(by=['cl_median', 'plate_median'], inplace=True)

    if title is None:
        title = 'Control DIP rates by plate'

    if 'dataset' in df_controls.index.names:
        dataset_names = df_controls.index.get_level_values('dataset').unique()

        if len(dataset_names) != 1:
            raise ValueError('This function can only plot controls from a '
                             'single dataset')

        if subtitle is None:
            subtitle = dataset_names[0]

    title = _combine_title_subtitle(title, subtitle)

    traces = []
    for grp, ctrl_dat in df_controls.groupby(level=['cell_line']):
        traces.append(go.Box(
            x=ctrl_dat.index.get_level_values('plate').values,
            y=ctrl_dat['dip_rate'].values,
            name=grp
        ))

    layout = go.Layout(title=title,
                       yaxis={'title': 'DIP Rate (h<sup>-1</sup>)'})
    return go.Figure(data=traces, layout=layout)


def plot_plate_map(plate_data, color_by='dip_rates',
                   missing_color='lightgray', subtitle=None):
    """

    Parameters
    ----------
    plate_data: thunor.io.PlateData
        Plate map layout data
    color_by: str
        Attribute to color wells by, must be numerical (default: dip_rates)
    missing_color: str
        Color to use for missing values (default: lightgray)
    subtitle: str or None
        Subtitle, or None to auto-generate

    Returns
    -------
    plotly.graph_objs.Figure
        A plotly figure object containing the graph
    """
    maintitle = 'DIP Rate Plate Map'
    if subtitle is None:
        subtitle = 'Plate {}'.format(plate_data.plate_name)
        if plate_data.dataset_name:
            subtitle += ' ({})'.format(plate_data.dataset_name)
    title = _combine_title_subtitle(maintitle, subtitle)

    well_color_basis = np.array(getattr(plate_data, color_by),
                                dtype=np.double)

    well_max = np.nanmax(well_color_basis)
    well_min = np.nanmin(well_color_basis)
    pos_pal = sns.light_palette("#f48000", as_cmap=True)
    neg_pal = sns.light_palette("#3f83a3", as_cmap=True)

    well_color = _sns_to_rgb([
        missing_color
        if np.isnan(val)
        else pos_pal(val / well_max)
        if val > 0
        else neg_pal(val / well_min)
        for val in well_color_basis
    ])

    cols = plate_data.width
    rows = plate_data.height
    num_wells = cols * rows

    well_rad = PLATE_MAP_WELL_DIAM / 2

    col_labels = [str(i + 1) for i in range(cols)]
    row_labels = [chr(ASCII_CAP_A + i) for i in range(rows)]

    well_shapes = [{
        'opacity': 1.0,
        'xref': 'x',
        'yref': 'y',
        'fillcolor': well_color[well_num],
        'x0': (well_num % cols),
        'y0': rows - (well_num // cols),
        'x1': (well_num % cols) + PLATE_MAP_WELL_DIAM,
        'y1': rows - (well_num // cols) + PLATE_MAP_WELL_DIAM,
        'type': 'circle',
        'line': {
            'color': 'darkgray',
            'width': 0.5
        }
    } for well_num in range(num_wells)]

    well_objs = go.Scatter(
        x=[(well_num % cols) + well_rad for well_num in range(num_wells)],
        y=[rows - (well_num // cols) + well_rad for well_num in
           range(num_wells)],
        text='',
        hovertext=['Well {}{}<br>'
                   'DIP: {}<br>'
                   'Cell Line: {}<br>'
                   'Drug: {}<br>'
                   'Dose: {}'.format(
                        row_labels[well_num // cols],
                        col_labels[well_num % cols],
                        plate_data.dip_rates[well_num],
                        plate_data.cell_lines[well_num],
                        " &amp; ".join(["(None)" if d is None else d for d in
                                        plate_data.drugs[well_num]]) if
                                        plate_data.drugs[well_num] else 'None',
                        " &amp; ".join([format_dose(d) for d in
                                        plate_data.doses[well_num]]) if
                                        plate_data.doses[well_num] else 'N/A'
            )
            for well_num in range(num_wells)],
        hoverinfo='text',
        mode='text'
    )

    col_labels = go.Scatter(
        x=[i + well_rad for i in range(cols)],
        y=[rows + 1.2] * cols,
        mode='text',
        text=col_labels,
        hoverinfo='none',
        textfont=dict(
            color='black',
            # size=18,
        )
    )

    row_labels = go.Scatter(
        x=[-well_rad] * rows,
        y=[i + 1 + well_rad for i in reversed(range(rows))],
        mode='text',
        text=row_labels,
        hoverinfo='none',
        textfont=dict(
            color='black',
            # size=18,
        )
    )

    data = (well_objs, col_labels, row_labels)

    layout = go.Layout({
        'xaxis': {
            'showticklabels': False,
            'showgrid': False,
            'zeroline': False,
        },
        'yaxis': {
            'showticklabels': False,
            'showgrid': False,
            'zeroline': False,
            'scaleanchor': 'x'
        },
        'shapes': well_shapes,
        'margin': {
            'l': 0,
            'r': 0,
            'b': 0,
            't': 50
        },
        'hovermode': 'closest',
        'showlegend': False,
        'title': title
    })

    return go.Figure(data=data, layout=layout)
