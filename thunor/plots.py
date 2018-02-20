import plotly.graph_objs as go
import numpy as np
import seaborn as sns
from .helpers import format_dose
from .curve_fit import ll4
from .dip import ctrl_dip_rates, expt_dip_rates, is_param_truncated, \
    PARAM_EQUAL_ATOL, PARAM_EQUAL_RTOL, DrugCombosNotImplementedError
import scipy.stats
import re
import pandas as pd
import collections


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
PARAM_UNITS = {'auc': _activity_area_units,
               'aa': _activity_area_units,
               'einf': 'h<sup>-1</sup>',
               'emax': 'h<sup>-1</sup>',
               'emax_obs': 'h<sup>-1</sup>'}
PARAM_NAMES = {'aa': 'Activity area',
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


def _param_is_log(param_id):
    return IC_REGEX.match(param_id) or EC_REGEX.match(param_id)


def _get_param_name(param_id):
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
    return '{} truncated to maximum measured concentration'.format(
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


def plot_dip(fit_params, is_absolute=False,
             title=None, subtitle=None, hill_fn=ll4):
    """
    Plot DIP rate curve fits

    Parameters
    ----------
    fit_params: pd.DataFrame
        DIP fit parameters from :func:`thunor.dip.dip_params`
    is_absolute: bool
        Plot using absolute (True) or relative (False) scale
    title: str, optional
        Title (or None to auto-generate)
    subtitle: str, optional
        Subtitle (or None to auto-generate)
    hill_fn: function
        Curve fitting function to use (default: :func:`thunor.curve_fit.ll4`)

    Returns
    -------
    plotly.graph_objs.Figure
        A plotly figure object containing the graph
    """

    colours = _sns_to_rgb(sns.color_palette("husl", len(fit_params)))
    # Shapes used for replicate markers
    shapes = ['circle', 'circle-open']

    datasets = fit_params.index.get_level_values('dataset_id').unique()

    yaxis_title = 'DIP rate'
    if is_absolute:
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
        subtitle = " &amp; ".join(datasets)
    title = _combine_title_subtitle(title, subtitle)

    annotations = []
    traces = []
    for fp in fit_params.itertuples():
        this_colour = colours.pop()
        group_name_disp = fp.label

        popt_plot = fp.popt if is_absolute else fp.popt_rel

        try:
            ctrl_doses = fp.dip_ctrl.index.get_level_values('dose')
        except AttributeError:
            ctrl_doses = []

        expt_doses = fp.dip_expt.index.get_level_values('dose')

        doses = np.concatenate((ctrl_doses, expt_doses))

        # Calculate the dip rate fit
        log_dose_min = int(np.floor(np.log10(min(doses))))
        log_dose_max = int(np.ceil(np.log10(max(doses))))

        dose_x_range = np.concatenate(
            # [np.arange(2, 11) * 10 ** dose_mag
            [0.5 * np.arange(3, 21) * 10 ** dose_mag
             for dose_mag in range(log_dose_min, log_dose_max + 1)],
            axis=0)

        dose_x_range = np.append([10 ** log_dose_min], dose_x_range,
                                 axis=0)

        if popt_plot is None:
            dip_rate_fit = [1 if not is_absolute else fp.divisor] * \
                           len(dose_x_range)
        else:
            dip_rate_fit = hill_fn(dose_x_range, *popt_plot)

        traces.append(go.Scatter(x=dose_x_range,
                                 y=dip_rate_fit,
                                 mode='lines',
                                 line={'shape': 'spline',
                                       'color': this_colour,
                                       'dash': 5 if popt_plot is None else
                                       'solid',
                                       'width': 3},
                                 legendgroup=group_name_disp,
                                 showlegend=not show_replicates or
                                            multi_dataset,
                                 name=group_name_disp)
                      )

        if show_replicates:
            y_trace = fp.dip_expt
            try:
                dip_ctrl = fp.dip_ctrl
            except AttributeError:
                dip_ctrl = None
            if not is_absolute:
                y_trace /= fp.divisor
                if dip_ctrl is not None:
                    dip_ctrl /= fp.divisor

            repl_name = 'Replicate'
            ctrl_name = 'Control'
            if multi_dataset:
                repl_name = '{} {}'.format(fp.Index[0], repl_name)
                ctrl_name = '{} {}'.format(fp.Index[0], ctrl_name)

            shape = shapes.pop(0)

            traces.append(go.Scatter(x=expt_doses,
                                     y=y_trace,
                                     mode='markers',
                                     marker={'symbol': shape,
                                             'color': this_colour,
                                             'size': 5},
                                     legendgroup=group_name_disp,
                                     hoverinfo='x+y+text',
                                     text=repl_name,
                                     showlegend=False,
                                     name=repl_name)
                          )
            if dip_ctrl is not None:
                traces.append(go.Scatter(x=ctrl_doses,
                                         y=dip_ctrl,
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
            ec50_truncated = False
            ic50_truncated = False
            if fp.ec50 is not None:
                ec50_truncated = np.allclose(fp.ec50, fp.max_dose_measured,
                                             atol=PARAM_EQUAL_ATOL,
                                             rtol=PARAM_EQUAL_RTOL)
                annotation_label += 'EC<sub>50</sub>{}: {} '.format(
                    '*' if ec50_truncated else '',
                    format_dose(fp.ec50, sig_digits=5)
                )
            if fp.ic50 is not None:
                ic50_truncated = np.allclose(fp.ic50, fp.max_dose_measured,
                                             atol=PARAM_EQUAL_ATOL,
                                             rtol=PARAM_EQUAL_RTOL)
                annotation_label += 'IC<sub>50</sub>{}: {} '.format(
                    '*' if ic50_truncated else '',
                    format_dose(fp.ic50, sig_digits=5)
                )
            if fp.emax is not None:
                annotation_label += 'E<sub>max</sub>{}: {:.5g}'.format(
                    '*' if fp.einf < fp.emax else '',
                    fp.emax)
            if annotation_label:
                hovermsgs = []
                hovertext = None
                if ec50_truncated:
                    hovermsgs.append(_out_of_range_msg('ec50'))
                if ic50_truncated:
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
    data = go.Data(traces)
    layout = go.Layout(title=title,
                       hovermode='closest' if show_replicates
                                 or len(traces) > 50 else 'x',
                       xaxis={'title': 'Dose (M)',
                              'range': np.log10((1e-12, 1e-5)),
                              'type': 'log'},
                       yaxis={'title': yaxis_title,
                              'range': (-0.02, 0.07) if is_absolute else
                              (-0.2, 1.2)
                              },
                       annotations=annotations,
                       )

    return go.Figure(data=data, layout=layout)


def plot_two_dataset_param_scatter(df_params, fit_param, title, subtitle,
                                   **kwargs):
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
        subtitle = " &amp; ".join(datasets)
    title = _combine_title_subtitle(title, subtitle)

    df_params = df_params.loc[:, [fit_param, 'max_dose_measured']]
    df_params.dropna(subset=[fit_param], inplace=True)
    df_params.reset_index(inplace=True)
    df_params = df_params.pivot_table(index=['cell_line', 'drug'], columns=[
        'dataset_id'])
    df_params.dropna(inplace=True)

    if len(df_params.index) == 0:
        raise ValueError('Dataset vs dataset scatter plot is empty. Check '
                         'the cell lines and drugs in the two datasets '
                         'overlap. If you want a bar plot instead, choose an '
                         'ordering parameter.')

    if len(df_params.columns) < 4:
        raise ValueError('The cell lines and/or drugs selected only have '
                         'data in one of the two datasets, so a scatter plot '
                         'cannot be created. If you want a bar plot instead, '
                         'choose an ordering parameter.')

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

    fit_param_data = df_params.loc[:, fit_param]
    xdat = fit_param_data.iloc[:, 0]
    xdat_fit = xdat
    ydat = fit_param_data.iloc[:, 1]
    ydat_fit = ydat
    if _param_is_log(fit_param):
        xdat_fit = np.log10(xdat)
        ydat_fit = np.log10(ydat)

    dataset_names = fit_param_data.columns

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
                                      param_name)
        ))
        layout['annotations'] = [{
            'x': 0.5, 'y': 1.0, 'xref': 'paper', 'yanchor': 'bottom',
            'yref': 'paper', 'showarrow': False,
            'text': 'R<sup>2</sup>: {:0.4g} '
                    'p-value: {:0.4g} '.format(r_value ** 2, p_value)
        }]

    hovertext = [" ".join(l) for l in fit_param_data.index.values]
    symbols = ['circle'] * len(fit_param_data.index)
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

    for param in range_bounded_params:
        msg = _out_of_range_msg(param)
        for i in (0, 1):
            tmp_df = pd.concat([
                fit_param_data.iloc[:, i],
                df_params.loc[:, 'max_dose_measured'].iloc[:, i]
            ],
            axis=1
            )
            tmp_df.columns = [fit_param, 'max_dose_measured']
            param_truncated = is_param_truncated(tmp_df, fit_param)
            addtxt = ['<br> {} {}'.format(dataset_names[i], msg) if x else
                      '' for x in param_truncated]
            hovertext = [ht + at for ht, at in zip(hovertext, addtxt)]
            symbols = ['cross' if x else old for x, old in
                       zip(param_truncated, symbols)]

    data.append(go.Scatter(
        x=xdat,
        y=ydat,
        hovertext=hovertext,
        hoverinfo="text+x+y",
        mode='markers',
        marker={'symbol': symbols,
                'color': [colours[1] if s == 'circle' else
                          'crimson' for s in symbols]},
        name='{} vs {} {}'.format(dataset_names[0],
                                  dataset_names[1],
                                  param_name)
    ))

    layout['xaxis'] = {'title': '{} {}'.format(dataset_names[0], axis_title),
                       'type': 'log' if _param_is_log(fit_param) else None}
    layout['yaxis'] = {'title': '{} {}'.format(dataset_names[1], axis_title),
                       'type': 'log' if _param_is_log(fit_param) else None}
    layout['hovermode'] = 'closest'
    layout['showlegend'] = False

    return go.Figure(layout=layout, data=data)


def plot_dip_params(df_params, fit_param,
                    fit_param_compare=None,
                    fit_param_sort=None,
                    title=None,
                    subtitle=None,
                    aggregate_cell_lines=False,
                    aggregate_drugs=False,
                    multi_dataset=False,
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
    aggregate_cell_lines: bool or list, optional
        Aggregate all cell lines (if True), or aggregate by the specified
        groups (dict of cell line names as values, with group labels as keys)
    aggregate_drugs, bool or list, optional
        Aggregate all drugs (if True), or aggregate by the specified
        groups (dict of drug names as values, with group labels as keys)
    multi_dataset: bool
        Set to true to compare two datasets contained in fit_params
    kwargs: dict, optional
        Additional keyword arguments

    Returns
    -------
    plotly.graph_objs.Figure
        A plotly figure object containing the graph
    """
    if fit_param_compare and (aggregate_cell_lines or aggregate_drugs):
        raise ValueError('Aggregation is not available when comparing two '
                         'dose response parameters')

    if multi_dataset and fit_param_compare is None and \
            not aggregate_cell_lines and \
            not aggregate_drugs and fit_param_sort is None:
        return plot_two_dataset_param_scatter(
            df_params,
            fit_param,
            title,
            subtitle,
            **kwargs
        )

    colours = _sns_to_rgb(sns.color_palette("Paired"))[0:2]

    if title is None:
        title = _make_title('Dose response parameters', df_params)
    if subtitle is None:
        datasets = df_params.index.get_level_values('dataset_id').unique()
        subtitle = " &amp; ".join(datasets)
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

    layout = dict(title=title,
                  yaxis={'title': yaxis_title,
                         'type': 'log' if _param_is_log(fit_param) else None})

    if fit_param_compare:
        df_params.dropna(subset=[fit_param, fit_param_compare],
                         inplace=True)
        if df_params.empty:
            raise ValueError('No data exists for this selection. This may be '
                             'due to missing drug/cell line combinations, or '
                             'undefined parameters for the selection.')
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

        xdat_fit = np.log10(xdat) if _param_is_log(fit_param_compare) else xdat
        ydat_fit = np.log10(ydat) if _param_is_log(fit_param) else ydat

        data = []

        if len(xdat_fit) > 0 and len(ydat_fit) > 0:
            slope, intercept, r_value, p_value, std_err = \
                scipy.stats.linregress(xdat_fit, ydat_fit)
            if not np.isnan(slope):
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
                                                      yaxis_param_name)
                ))
                layout['annotations'] = [{
                    'x': 0.5, 'y': 1.0, 'xref': 'paper', 'yanchor': 'bottom',
                    'yref': 'paper', 'showarrow': False,
                    'text': 'R<sup>2</sup>: {:0.4g} '
                            'p-value: {:0.4g} '.format(r_value ** 2, p_value)
                }]

        hovertext = df_params['label']
        symbols = ['circle'] * len(df_params)
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

        for param in range_bounded_params:
            msg = _out_of_range_msg(param)
            param_truncated = is_param_truncated(df_params, param)
            addtxt = ['<br> ' + msg if x else '' for x in
                      param_truncated]
            hovertext = [ht + at for ht, at in zip(hovertext, addtxt)]
            symbols = ['cross' if x else old for x, old in
                       zip(param_truncated, symbols)]

        data.append(go.Scatter(
            x=xdat,
            y=ydat,
            hovertext=hovertext,
            hoverinfo="text+x+y",
            mode='markers',
            marker={'symbol': symbols,
                    'color': [colours[1] if s == 'circle' else
                              'crimson' for s in symbols]},
            name='{} vs {}'.format(xaxis_param_name,
                                   yaxis_param_name)
        ))

        layout['xaxis'] = {'title': xaxis_title,
                           'type': 'log' if _param_is_log(fit_param_compare)
                           else None}
        layout['hovermode'] = 'closest'
        layout['showlegend'] = False
    elif not aggregate_cell_lines and not aggregate_drugs:
        sort_by = [fit_param_sort, fit_param] if fit_param_sort is not None \
                   else fit_param
        df_params = df_params.sort_values(by=sort_by)
        groups = df_params['label']
        yvals = df_params[fit_param]

        text = None
        marker_cols = colours[1]

        ec_match = None
        for regex in (EC_REGEX, E_REGEX, E_REL_REGEX):
            match = regex.match(fit_param)
            if match:
                ec_match = 'ec' + match.groups(0)[0]
                break

        if ec_match:
            msg = _out_of_range_msg(ec_match)
            if not fit_param.startswith('ec'):
                msg = 'Based on ' + msg
            ec_truncated = is_param_truncated(df_params, ec_match)
            text = [msg if x else '' for x in ec_truncated]
            marker_cols = [colours[0] if est else colours[1] for
                           est in ec_truncated]
        elif IC_REGEX.match(fit_param):
            msg = _out_of_range_msg(fit_param)
            ic_truncated = is_param_truncated(df_params, fit_param)
            text = [msg if x else '' for x in ic_truncated]
            marker_cols = [colours[0] if est else colours[1] for
                           est in ic_truncated]

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

        data = [go.Bar(x=groups,
                       y=yvals,
                       name=fit_param,
                       text=text,
                       marker={'color': marker_cols}
                       )]
        layout['annotations'] = [
            {'x': x, 'y': 0, 'text': '<em>N/A</em>',
             'textangle': 90,
             'xanchor': 'center', 'yanchor': 'bottom',
             'yref': 'paper',
             'showarrow': False,
             'font': {'color': 'rgba(150, 150, 150, 1)'}}
            for x in groups[yvals.isnull().values]]
        layout.setdefault('xaxis', {})['type'] = 'category'
        layout['barmode'] = 'group'
    else:
        layout['boxmode'] = 'group'

        if fit_param_sort == fit_param:
            fit_param_sort = None

        if fit_param_sort is None:
            yvals = df_params.loc[:, [fit_param]].dropna()
        else:
            yvals = df_params.loc[:, [fit_param,
                                      fit_param_sort]]
            yvals.dropna(subset=[fit_param], inplace=True)

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

        # Sort by median effect per drug set, or cell line set if there's
        # only one drug/drug group
        if fit_param_sort is None:
            yvals['median'] = yvals[fit_param].groupby(
                level=aggregate_by).transform(np.nanmedian)
            yvals.set_index('median', append=True, inplace=True)
            yvals.sort_index(level=['median'] + aggregate_by, ascending=False,
                             inplace=True)
            yvals.reset_index('median', drop=True, inplace=True)
        else:
            # Sort by fit_column_sort, with tie breakers determined by
            # fit_param
            median_cols = yvals.loc[:, [fit_param_sort, fit_param]].groupby(
                level=aggregate_by).transform(np.nanmedian)
            median_cols.rename(columns={fit_param_sort: 'median',
                                        fit_param: 'median2'},
                               inplace=True)
            yvals = pd.concat([yvals, median_cols], axis=1)
            yvals.set_index(['median', 'median2'], append=True, inplace=True)
            yvals.sort_index(level=['median', 'median2'] + aggregate_by,
                             ascending=False, inplace=True)
            yvals.reset_index(['median', 'median2'], drop=True, inplace=True)

        # Convert yvals to a series
        yvals = yvals.iloc[:, 0]

        data = []
        aggregate_by.remove('dataset_id')
        aggregate_by = aggregate_by[0]
        group_by = ['dataset_id', 'drug'] \
                    if aggregate_by == 'cell_line' \
                    else ['dataset_id', 'cell_line']
        for grp_name, grp in yvals.groupby(level=group_by):
            if len(groups) > 1:
                group_name = "<br>".join(str(g) for g in grp_name)
            else:
                # If there's only one drug/cell line group, just need dataset
                group_name = grp_name[0]
            data.append(go.Box(x=grp.index.get_level_values(aggregate_by),
                               y=grp,
                               name=group_name
                               ))

        if len(groups) == 1:
            annotation_label = str(groups[0])
            layout['annotations'] = [{'x': 0.5, 'y': 1.0, 'xref': 'paper',
                                      'yanchor': 'bottom', 'yref': 'paper',
                                      'showarrow': False,
                                      'text': annotation_label
                                      }]

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
                      replace_index=True):
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
        yvals_tmp = yvals.iloc[yvals.index.isin(names,
                                                level=label_type), :]
        yvals_tmp.is_copy = False  # suppress warning about assigning to copy

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
        df_controls = df_controls.loc[assay]
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
            timecourse = df_vals.loc[well_id]['value']
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
                customdata={'csvname': dose},
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
                    customdata={'csvname': 'DIP fit ' + dose_str},
                    legendgroup=dose_str,
                    showlegend=False
                ))

    data = go.Data(traces + traces_fits)
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

    data = go.Data(traces)
    layout = go.Layout(title=title,
                       yaxis={'title': 'DIP Rate (h<sup>-1</sup>)'})
    return go.Figure(data=data, layout=layout)


def plot_plate_map(plate_map, color_by='dipRate', missing_color='lightgray'):
    """

    Parameters
    ----------
    plate_map: dict
        Plate map layout data
    color_by: str
        Attribute to color wells by (default: dipRate)
    missing_color: str
        Color to use for missing values (default: lightgray)

    Returns
    -------
    plotly.graph_objs.Figure
        A plotly figure object containing the graph
    """
    maintitle = 'DIP Rate Plate Map'
    subtitle = 'Plate {} ({})'.format(plate_map['plateName'], plate_map[
        'datasetName'])
    title = _combine_title_subtitle(maintitle, subtitle)

    wells = plate_map['wells']
    well_color_basis = np.array([well[color_by] for well in wells],
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

    cols = plate_map['numCols']
    rows = len(wells) // cols
    WELL_DIAM = 0.95
    ASCII_CAP_A = 65

    well_rad = WELL_DIAM / 2

    col_labels = [str(i + 1) for i in range(cols)]
    row_labels = [chr(ASCII_CAP_A + i) for i in range(rows)]

    well_shapes = [{
        'opacity': 1.0,
        'xref': 'x',
        'yref': 'y',
        'fillcolor': well_color[well_num],
        'x0': (well_num % cols),
        'y0': rows - (well_num // cols),
        'x1': (well_num % cols) + WELL_DIAM,
        'y1': rows - (well_num // cols) + WELL_DIAM,
        'type': 'circle',
        'line': {
            'color': 'darkgray',
            'width': 0.5
        }
    } for well_num, well in enumerate(wells)]

    well_objs = go.Scatter(
        x=[(well_num % cols) + well_rad for well_num in range(len(
            wells))],
        y=[rows - (well_num // cols) + well_rad for well_num in range(
            len(wells))],
        text='',
        hovertext=['Well {}{}<br>'
                   'DIP: {}<br>'
                   'Cell Line: {}<br>'
                   'Drug: {}<br>'
                   'Dose: {}'.format(
                        row_labels[well_num // cols],
                        col_labels[well_num % cols],
                        well['dipRate'],
                        well['cellLine'],
                        " &amp; ".join(["(None)" if d is None else d for d in
                                        well['drugs']]) if well['drugs']
                        else 'None',
                        " &amp; ".join([format_dose(d) for d in well[
                            'doses']]) if well['doses'] else 'N/A'
            )
            for well_num, well in enumerate(wells)],
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

    data = go.Data([well_objs, col_labels, row_labels])

    layout = go.Layout({
        'xaxis': {
            'showticklabels': False,
            'autotick': False,
            'showgrid': False,
            'zeroline': False,
        },
        'yaxis': {
            'showticklabels': False,
            'autotick': False,
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
