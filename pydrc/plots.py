import plotly.graph_objs as go
import numpy as np
import seaborn as sns
from .helpers import format_dose
from .curve_fit import ll4
from .dip import ctrl_dip_rates, expt_dip_rates
import scipy.stats
import pandas as pd


def _activity_area_title(**kwargs):
    title = 'Activity area'
    if 'aa_max_conc' in kwargs:
        title += ' [max. dose={}]'.format(format_dose(kwargs['aa_max_conc']))
    return title


def _auc_title(**kwargs):
    title = 'Area under curve (AUC)'
    if 'auc_min_conc' in kwargs:
        title += ' [min. dose={}]'.format(format_dose(kwargs['auc_min_conc']))
    return title

SECONDS_IN_HOUR = 3600.0
PLOT_AXIS_LABELS = {'auc': _activity_area_title,
                    'aa': _activity_area_title,
                    'ic50': 'IC<sub>50</sub> (M)',
                    'ec50': 'EC<sub>50</sub> (M)',
                    'emax': 'E<sub>max</sub> (h<sup>-1</sup>)',
                    'hill': 'Hill coefficient'}
EC50_OUT_OF_RANGE_MSG = 'EC<sub>50</sub> &gt; measured concentrations<br> '
IC50_OUT_OF_RANGE_MSG = 'IC<sub>50</sub> &gt; measured concentrations<br> '
EMAX_TRUNCATED_MSG = 'E<sub>max</sub> truncated at effect of maximum dose<br> '
PARAMETERS_LOG_SCALE = ('ec50', 'ic50')


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
            if fp.ec50 is not None:
                annotation_label += 'EC<sub>50</sub>{}: {} '.format(
                    '*' if fp.ec50_out_of_range else '',
                    format_dose(fp.ec50, sig_digits=5)
                )
            if fp.ic50 is not None:
                annotation_label += 'IC<sub>50</sub>{}: {} '.format(
                    '*' if fp.ic50_out_of_range else '',
                    format_dose(fp.ic50, sig_digits=5)
                )
            if fp.emax is not None:
                annotation_label += 'E<sub>max</sub>{}: {:.5g}'.format(
                    '*' if fp.einf < fp.emax else '',
                    fp.emax)
            if annotation_label:
                hovertext = None
                if fp.ec50_out_of_range or fp.ic50_out_of_range \
                        or fp.einf < fp.emax:
                    hovertext = '*'
                    if fp.ec50_out_of_range:
                        hovertext += EC50_OUT_OF_RANGE_MSG
                    if fp.ic50_out_of_range:
                        hovertext += IC50_OUT_OF_RANGE_MSG
                    if fp.einf < fp.emax:
                        hovertext += EMAX_TRUNCATED_MSG
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
                       hovermode='closest' if show_replicates else 'x',
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


def plot_dip_params(fit_params, fit_params_sort,
                    fit_params_compare=None,
                    title=None,
                    subtitle=None, aggregate_cell_lines=False,
                    aggregate_drugs=False, **kwargs):
    if fit_params_compare and (aggregate_cell_lines or aggregate_drugs):
        raise ValueError('Aggregation is not available when comparing two '
                         'dose response parameters')

    colours = _sns_to_rgb(sns.color_palette("Paired"))[0:2]

    if title is None:
        title = _make_title('Dose response parameters', fit_params)
    title = _combine_title_subtitle(title, subtitle)

    yaxis_title = PLOT_AXIS_LABELS.get(fit_params_sort, fit_params_sort)
    try:
        yaxis_title = yaxis_title(**kwargs)
    except TypeError:
        pass

    layout = dict(title=title,
                  yaxis={'title': yaxis_title,
                         'type': 'log' if fit_params_sort in
                                 PARAMETERS_LOG_SCALE else None})

    if fit_params_compare:
        fit_params.dropna(subset=[fit_params_sort, fit_params_compare],
                          inplace=True)
        if fit_params_sort == fit_params_compare:
            xdat = fit_params.loc[:, fit_params_sort]
            ydat = xdat
        else:
            dat = fit_params.loc[:, [fit_params_compare, fit_params_sort]]
            xdat = dat[fit_params_compare]
            ydat = dat[fit_params_sort]

        xdat_fit = np.log10(xdat) if fit_params_compare in \
            PARAMETERS_LOG_SCALE else xdat
        ydat_fit = np.log10(ydat) if fit_params_sort in PARAMETERS_LOG_SCALE \
            else ydat

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
            xdat_fit, ydat_fit)

        hovertext = fit_params['label']
        symbols = ['circle'] * len(fit_params)
        if fit_params_compare == 'ic50' or fit_params_sort == 'ic50':
            addtxt = ['<br> ' + IC50_OUT_OF_RANGE_MSG if x else '' for x in
                      fit_params['ic50_out_of_range']]
            hovertext = [ht + at for ht, at in zip(hovertext, addtxt)]
            symbols = ['cross' if x else 'circle' for x in
                       fit_params['ic50_out_of_range']]

        if fit_params_compare in ('ec50', 'auc', 'aa') or \
                fit_params_sort in ('ec50', 'auc', 'aa'):
            addtxt = ['<br> ' + EC50_OUT_OF_RANGE_MSG if x else '' for x in
                      fit_params['ec50_out_of_range']]
            hovertext = [ht + at for ht, at in zip(hovertext, addtxt)]
            symbols = ['cross' if x else old for x, old in
                       zip(fit_params['ec50_out_of_range'], symbols)]

        if fit_params_compare == 'emax' or fit_params_sort == 'emax':
            emax_truncated = fit_params['einf'] < fit_params['emax']
            addtxt = ['<br> ' + EMAX_TRUNCATED_MSG if x else '' for x in
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
            marker={'symbol': symbols}
        )]
        if not np.isnan(slope):
            xfit = (min(xdat_fit), max(xdat_fit))
            yfit = [x * slope + intercept for x in xfit]
            if fit_params_compare in PARAMETERS_LOG_SCALE:
                xfit = np.power(10, xfit)
            if fit_params_sort in PARAMETERS_LOG_SCALE:
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
        xaxis_title = PLOT_AXIS_LABELS.get(fit_params_compare,
                                           fit_params_compare)
        try:
            xaxis_title = xaxis_title(**kwargs)
        except TypeError:
            pass
        layout['xaxis'] = {'title': xaxis_title,
                           'type': 'log' if fit_params_compare in
                                   ('ec50', 'ic50') else None}
        layout['hovermode'] = 'closest'
        layout['showlegend'] = False
    elif not aggregate_cell_lines and not aggregate_drugs:
        fit_params = fit_params.sort_values(by=fit_params_sort,
                                            na_position='first')
        groups = fit_params['label']
        yvals = fit_params[fit_params_sort]

        text = None
        marker_cols = colours[1]

        if fit_params_sort in ('ec50', 'auc', 'aa'):
            msg = EC50_OUT_OF_RANGE_MSG
            if fit_params_sort != 'ec50':
                msg = 'Based on ' + EC50_OUT_OF_RANGE_MSG
            text = [msg if x else None for x in fit_params[
                'ec50_out_of_range']]
            marker_cols = [colours[0] if est else colours[1] for
                           est in fit_params['ec50_out_of_range']]
        elif fit_params_sort == 'ic50':
            text = [IC50_OUT_OF_RANGE_MSG if x else None for x in fit_params[
                'ic50_out_of_range']]
            marker_cols = [colours[0] if est else colours[1] for
                           est in fit_params['ic50_out_of_range']]
        elif fit_params_sort == 'emax':
            emax_truncated = fit_params['einf'] < fit_params['emax']
            text = [EMAX_TRUNCATED_MSG if x else None for x in emax_truncated]
            marker_cols = [colours[0] if x else colours[1] for x in
                           emax_truncated]

        data = [go.Bar(x=groups, y=yvals,
                       name=fit_params_sort,
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
        yvals = fit_params[fit_params_sort]

        data = []

        if aggregate_cell_lines:
            if aggregate_cell_lines is True:
                cell_lines = yvals.index.get_level_values(
                    level='cell_line').unique().tolist()
                aggregate_cell_lines = {
                    _create_label_max_items(cell_lines, 5): cell_lines
                }

            for cl_tag_name, cl_names in aggregate_cell_lines.items():
                yvals_tmp = yvals.iloc[yvals.index.isin(cl_names,
                                                        level='cell_line')]
                num_cell_lines = len(yvals_tmp.index.get_level_values(
                    'cell_line').unique())
                cl_tag_label = '{} [{}]'.format(cl_tag_name, num_cell_lines)
                if aggregate_drugs:
                    data.extend(_agg_drugs(
                        yvals_tmp,
                        aggregate_drugs,
                        x=np.repeat(cl_tag_label, len(yvals_tmp))
                    ))
                else:
                    for dr_name, grp in yvals_tmp.groupby(level='drug'):
                        data.append(go.Box(x=np.repeat(cl_tag_label, len(grp)),
                                           y=grp,
                                           name=dr_name))

        else:
            data.extend(_agg_drugs(yvals, aggregate_drugs))

        if aggregate_drugs and \
                (aggregate_drugs is True or len(aggregate_drugs) == 1):
            drugs = yvals.index.get_level_values('drug').unique().tolist()
            annotation_label = '{} [{}]'.format(
                _create_label_max_items(drugs, 3),
                len(drugs)
            )
            layout['annotations'] = [{'x': 0.5, 'y': 1.0, 'xref': 'paper',
                                      'yanchor': 'bottom', 'yref': 'paper',
                                      'showarrow': False,
                                      'text': annotation_label
                                      }]

    layout = go.Layout(layout)

    return go.Figure(data=data, layout=layout)


def _agg_drugs(series, aggregate_drugs, x=None):
    data = []

    if aggregate_drugs is True:
        drugs = series.index.get_level_values(
            'drug').unique().tolist()
        aggregate_drugs = {_create_label_max_items(drugs, 1): drugs}

    for tag_name, drug_names in aggregate_drugs.items():
        y = series.iloc[series.index.isin(
            drug_names, level='drug')]
        num_drugs = len(y.index.get_level_values('drug').unique())
        tag_label = '{} [{}]'.format(tag_name, num_drugs)
        if x is None:
            x = y.index.get_level_values('cell_line')
        data.append(go.Box(x=x, y=y, name=tag_label))

    return data


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
