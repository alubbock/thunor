import plotly.graph_objs as go
import numpy as np
import seaborn as sns
from .helpers import format_dose
from .curve_fit import ll4
from .dip import ctrl_dip_rates, expt_dip_rates, is_param_truncated, \
    PARAM_EQUAL_ATOL, PARAM_EQUAL_RTOL
import scipy.stats
import re
import pandas as pd


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
               'ic10': 'M',
               'ic50': 'M',
               'ic100': 'M',
               'ec50': 'M',
               'emax': 'h<sup>-1</sup>',
               'emax_obs': 'h<sup>-1</sup>',
               'e50': 'h<sup>-1</sup>'}
PARAM_NAMES = {'aa': 'Activity area',
               'auc': 'Area under curve',
               'ic10': 'IC<sub>10</sub>',
               'ic50': 'IC<sub>50</sub>',
               'ic100': 'IC<sub>100</sub>',
               'ec50': 'EC<sub>50</sub>',
               'emax': 'E<sub>max</sub>',
               'emax_rel': 'E<sub>max</sub> (relative)',
               'emax_obs': 'E<sub>max</sub> observed',
               'emax_obs_rel': 'E<sub>Max</sub> observed (relative)',
               'e50': 'E<sub>50</sub>',
               'hill': 'Hill coefficient'}
EMAX_TRUNCATED_MSG = 'E<sub>max</sub> truncated at effect of maximum dose'
PARAMETERS_LOG_SCALE = ('ec50', 'ic50', 'ic10', 'ic100')
IC_REGEX = re.compile('ic[0-9]+$')


def _out_of_range_msg(param_name):
    return '{} truncated to maximum measured concentration'.format(
        PARAM_NAMES.get(param_name, param_name)
    )


def _sns_to_rgb(palette):
    return ['rgb(%d, %d, %d)' % (c[0] * 255, c[1] * 255, c[2] * 255) for c
            in palette]


def _make_title(title, df):
    drug_list = df.index.get_level_values('drug').unique()
    if len(drug_list) == 1:
        title += ' for {}'.format(drug_list[0])

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

    colours = _sns_to_rgb(sns.color_palette("husl", len(fit_params)))

    yaxis_title = 'DIP rate'
    if is_absolute:
        yaxis_title += ' (h<sup>-1</sup>)'
    else:
        yaxis_title = 'Relative ' + yaxis_title

    show_replicates = len(fit_params) == 1

    if title is None:
        title = _make_title('Dose response', fit_params)
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
                                 showlegend=not show_replicates,
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

            traces.append(go.Scatter(x=expt_doses,
                                     y=y_trace,
                                     mode='markers',
                                     line={'shape': 'spline',
                                           'color': this_colour,
                                           'width': 3},
                                     legendgroup=group_name_disp,
                                     showlegend=False,
                                     name='Replicate',
                                     marker={'size': 5})
                          )
            if dip_ctrl is not None:
                traces.append(go.Scatter(x=ctrl_doses,
                                         y=dip_ctrl,
                                         mode='markers',
                                         line={'shape': 'spline',
                                               'color': 'black',
                                               'width': 3},
                                         hoverinfo='y+name',
                                         name='Control',
                                         legendgroup=group_name_disp,
                                         showlegend=False,
                                         marker={'size': 5})
                              )

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
                if fp.einf < fp.emax:
                    hovermsgs.append(EMAX_TRUNCATED_MSG)
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


def plot_dip_params(df_params, fit_param,
                    fit_param_compare=None,
                    fit_param_sort=None,
                    title=None,
                    subtitle=None, aggregate_cell_lines=False,
                    aggregate_drugs=False, **kwargs):
    if fit_param_compare and (aggregate_cell_lines or aggregate_drugs):
        raise ValueError('Aggregation is not available when comparing two '
                         'dose response parameters')

    if fit_param_sort and aggregate_cell_lines:
        raise ValueError('Cannot use another parameter for sort order when '
                         'aggregating cell lines')

    colours = _sns_to_rgb(sns.color_palette("Paired"))[0:2]

    if title is None:
        title = _make_title('Dose response parameters', df_params)
    title = _combine_title_subtitle(title, subtitle)

    yaxis_param_name = PARAM_NAMES.get(fit_param, fit_param)
    yaxis_units = PARAM_UNITS.get(fit_param, '')
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
                         'type': 'log' if fit_param in
                                          PARAMETERS_LOG_SCALE else None})

    if fit_param_compare:
        df_params.dropna(subset=[fit_param, fit_param_compare],
                         inplace=True)
        if fit_param == fit_param_compare:
            xdat = df_params.loc[:, fit_param]
            ydat = xdat
        else:
            dat = df_params.loc[:, [fit_param_compare, fit_param]]
            xdat = dat[fit_param_compare]
            ydat = dat[fit_param]

        xdat_fit = np.log10(xdat) if fit_param_compare in \
                                     PARAMETERS_LOG_SCALE else xdat
        ydat_fit = np.log10(ydat) if fit_param in PARAMETERS_LOG_SCALE \
            else ydat

        if len(xdat_fit) > 0 and len(ydat_fit) > 0:
            slope, intercept, r_value, p_value, std_err = \
                scipy.stats.linregress(xdat_fit, ydat_fit)
        else:
            slope = np.nan

        hovertext = df_params['label']
        symbols = ['circle'] * len(df_params)
        ic_params = set()

        for param in (fit_param, fit_param_compare):
            match = IC_REGEX.match(param)
            if match:
                ic_params.add(match.group())

        for ic_param in ic_params:
            msg = _out_of_range_msg(ic_param)
            ic_truncated = is_param_truncated(df_params, ic_param)
            addtxt = ['<br> ' + msg if x else '' for x in
                      ic_truncated]
            hovertext = [ht + at for ht, at in zip(hovertext, addtxt)]
            symbols = ['cross' if x else old for x, old in
                       zip(ic_truncated, symbols)]

        if fit_param_compare in ('ec50', 'e50') or \
                fit_param in ('ec50', 'e50'):
            msg = _out_of_range_msg('ec50')
            ec50_truncated = is_param_truncated(df_params, 'ec50')
            addtxt = ['<br> ' + msg if x else '' for x in
                      ec50_truncated]
            hovertext = [ht + at for ht, at in zip(hovertext, addtxt)]
            symbols = ['cross' if x else old for x, old in
                       zip(ec50_truncated, symbols)]

        if fit_param_compare in ('emax', 'emax_rel', 'aa') or \
                        fit_param in ('emax', 'emax_rel', 'aa'):
            emax_truncated = df_params['einf'] < df_params['emax']
            msg = EMAX_TRUNCATED_MSG
            if fit_param_compare == 'aa' or fit_param == 'aa':
                msg = 'Based on ' + msg
            addtxt = ['<br> ' + msg if x else '' for x in
                      emax_truncated]
            hovertext = [ht + at for ht, at in zip(hovertext, addtxt)]
            symbols = ['cross' if x else old for x, old in
                       zip(emax_truncated, symbols)]

        data = [go.Scatter(
            x=xdat,
            y=ydat,
            hovertext=hovertext,
            hoverinfo="text+x+y",
            mode='markers',
            marker={'symbol': symbols,
                    'color': [colours[1] if s == 'circle' else
                              'crimson' for s in symbols]}
        )]
        if not np.isnan(slope):
            xfit = (min(xdat_fit), max(xdat_fit))
            yfit = [x * slope + intercept for x in xfit]
            if fit_param_compare in PARAMETERS_LOG_SCALE:
                xfit = np.power(10, xfit)
            if fit_param in PARAMETERS_LOG_SCALE:
                yfit = np.power(10, yfit)
            data.append(go.Scatter(
                x=xfit,
                y=yfit,
                mode='lines',
                hoverinfo="none"
            ))
            layout['annotations'] = [{
                    'x': 0.5, 'y': 1.0, 'xref': 'paper', 'yanchor': 'bottom',
                    'yref': 'paper', 'showarrow': False,
                    'text': 'R<sup>2</sup>: {:0.4g} '
                            'p-value: {:0.4g} '.format(r_value**2, p_value)
                }]
        xaxis_param_name = PARAM_NAMES.get(fit_param_compare,
                                           fit_param_compare)
        xaxis_units = PARAM_UNITS.get(fit_param_compare, '')
        try:
            xaxis_units = xaxis_units(**kwargs)
        except TypeError:
            pass
        if xaxis_units:
            xaxis_title = '{} ({})'.format(xaxis_param_name, xaxis_units)
        else:
            xaxis_title = xaxis_param_name
        layout['xaxis'] = {'title': xaxis_title,
                           'type': 'log' if fit_param_compare in
                                            PARAMETERS_LOG_SCALE else None}
        layout['hovermode'] = 'closest'
        layout['showlegend'] = False
    elif not aggregate_cell_lines and not aggregate_drugs:
        sort_by = fit_param_sort if fit_param_sort is not None else fit_param
        df_params = df_params.sort_values(by=sort_by,
                                          na_position='first')
        groups = df_params['label']
        yvals = df_params[fit_param]

        text = None
        marker_cols = colours[1]

        if fit_param in ('ec50', 'e50'):
            msg = _out_of_range_msg('ec50')
            if fit_param != 'ec50':
                msg = 'Based on ' + msg
            ec50_truncated = is_param_truncated(df_params, 'ec50')
            text = [msg if x else None for x in ec50_truncated]
            marker_cols = [colours[0] if est else colours[1] for
                           est in ec50_truncated]
        elif IC_REGEX.match(fit_param):
            msg = _out_of_range_msg(fit_param)
            ic_truncated = is_param_truncated(df_params, fit_param)
            text = [msg if x else None for x in ic_truncated]
            marker_cols = [colours[0] if est else colours[1] for
                           est in ic_truncated]
        elif fit_param in ('emax', 'emax_rel', 'aa'):
            emax_truncated = df_params['einf'] < df_params['emax']
            msg = EMAX_TRUNCATED_MSG
            if fit_param == 'aa':
                msg = 'Based on ' + msg
            text = [msg if x else None for x in emax_truncated]
            marker_cols = [colours[0] if x else colours[1] for x in
                           emax_truncated]

        data = [go.Bar(x=groups, y=yvals,
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
            for x in groups[yvals.isnull()]]
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

        # Sort by median effect per drug set
        sortcol = fit_param_sort if fit_param_sort is \
                                    not None else fit_param
        yvals['median'] = yvals[sortcol].groupby(
            level='cell_line').transform(np.median)
        if fit_param_sort:
            yvals.drop(labels=[fit_param_sort], axis=1, inplace=True)
        yvals.set_index('median', append=True, inplace=True)
        print(yvals)
        yvals.sort_index(level=['median', 'cell_line'], ascending=False,
                         inplace=True)
        print(yvals)
        yvals.reset_index('median', drop=True, inplace=True)

        # Convert yvals to a series
        yvals = yvals.iloc[:, 0]

        data = []
        for drug, grp in yvals.groupby(level='drug'):
            # print(grp)
            data.append(go.Box(x=grp.index.get_level_values('cell_line'),
                               y=grp,
                               name=drug
                               ))

        drug_groups = yvals.index.get_level_values('drug').unique()
        if len(drug_groups) == 1:
            annotation_label = str(drug_groups[0])
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

    other_label = 'drug' if label_type == 'cell_line' else 'cell_line'
    new.reset_index(other_label, inplace=True)
    if label_type == 'cell_line':
        index_labels = [label_type_tag, other_label]
    else:
        index_labels = [other_label, label_type_tag]
    new.set_index(index_labels, inplace=True, drop=replace_index)
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


def plot_time_course(df_doses, df_vals, df_controls,
                     log_yaxis=False, assay_name='Assay', title=None,
                     subtitle=None, show_dip_fit=False):
    if show_dip_fit and not log_yaxis:
        raise ValueError('log_yaxis must be True when show_dip_fit is True')
    traces = []
    traces_fits = []

    if title is None:
        title = _make_title('Time course', df_doses)
    title = _combine_title_subtitle(title, subtitle)

    colours = _sns_to_rgb(sns.color_palette(
        "husl", len(df_doses.index.get_level_values(level='dose').unique())))

    if show_dip_fit:
        dip_rate_ctrl = ctrl_dip_rates(df_controls)
        dip_rate_ctrl.index = dip_rate_ctrl.index.droplevel(level='cell_line')
        dip_rates = expt_dip_rates(df_doses, df_vals)
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

            if show_dip_fit:
                dip_well = dip_rate_ctrl.loc[well_id]
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
            timecourse = df_vals.loc[well_id, 'value']
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
