import collections
import pandas as pd


def format_dose(num, sig_digits=12, array_as_string=None):
    """
    Formats a numeric dose like 1.2e-9 into 1.2 nM
    """
    if not isinstance(num, str) and isinstance(num, collections.Iterable):
        retval = [format_dose(each_num) for each_num in num]
        if array_as_string is not None:
            return array_as_string.join(retval)
        return retval

    if num is None:
        return 'N/A'

    num = float(num)

    _prefix = {1e-12: 'p',
               1e-9: 'n',
               1e-6: 'μ',
               1e-3: 'm',
               1: ''}
    multiplier = 1
    for i in sorted(_prefix.keys()):
        if num >= i:
            multiplier = i
    return ('{0:.' + str(sig_digits) + 'g} {1}M').format(
        num/multiplier, _prefix[multiplier])


def plotly_to_dataframe(plot_fig):
    series = []
    for trace in plot_fig['data']:
        try:
            trace_name = trace['customdata']['csvname']
        except (TypeError, KeyError):
            try:
                trace_name = trace['name']
                if trace_name is not None:
                    trace_name = trace_name.replace('\\n', '')
            except KeyError:
                trace_name = None
        yvals = trace['y']
        xvals = trace['x']
        if isinstance(yvals, (pd.DataFrame, pd.Series)):
            if not isinstance(xvals, pd.Index) or xvals.is_unique:
                yvals = yvals.values
                series.append(pd.Series(yvals, index=xvals, name=trace_name))
            else:
                index_name = yvals.index.names[0]
                series_name = yvals.name
                yvals = pd.DataFrame(yvals)
                yvals['__replicate__'] = yvals.groupby(level=0).cumcount()
                yvals.reset_index(inplace=True)
                ptable = yvals.pivot(columns='__replicate__',
                                     index=index_name,
                                     values=series_name)
                ptable.rename(columns=lambda i: '{} {}'.format(trace_name, i),
                              inplace=True)
                series.extend(col for _, col in ptable.iteritems())
        else:
            series.append(pd.Series(yvals, index=xvals, name=trace_name))

    return pd.concat(series, axis=1)
