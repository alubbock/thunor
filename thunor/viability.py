from datetime import timedelta

SECONDS_IN_HOUR = 3600


def viability(df_data, time_hrs=72, assay_name=None, include_controls=True):
    """
    Calculate viability at the specified time point

    Viability is calculated as the assay value over the mean of controls
    from the same plate, cell line, and time point

    Parameters
    ----------
    df_data: HtsPandas
        HTS dataset
    time_hrs: float
        Time in hours to use for viability. The closest time point in each
        well to the one specified is used.
    assay_name: str, optional
        The assay name to use for viability calculation, or None to use the
        default proliferation assay
    include_controls: bool
        Return the control values for reference as a the second entry in a
        two-tuple, if True

    Returns
    -------
    pd.DataFrame, pd.Series or None
        A DataFrame containing the viability results and a Series containing
        the control values, if requested (None is returned as the second
        return value otherwise)
    """
    if df_data.controls is None:
        raise ValueError('Control wells not found, and are needed for '
                         'viability calculation')

    time = timedelta(hours=time_hrs)

    # Select assay per dataset
    if 'dataset' in df_data.doses.index.names:
        assays = df_data.assays.reset_index('assay')
    else:
        if assay_name is None:
            assay_name = df_data.dip_assay_name
        assays = df_data.assays.loc[assay_name]

    # Filter assays by nearest timepoint
    assays = _get_closest_timepoint_for_each_well(assays, time)
    assays.reset_index('timepoint', inplace=True)

    # Merge counts with well annotations
    df = df_data.doses.merge(assays, left_on='well_id', right_index=True)

    if 'dataset' in df_data.doses.index.names:
        dataset_assays = df.groupby('dataset')['assay'].unique()
        if not (dataset_assays.apply(len) == 1).all():
            raise NotImplementedError('Cannot calculate viability across two '
                                      'datasets when the datasets are '
                                      'multi-assay')
    if 'dataset' in df_data.controls.index.names:
        controls = df_data.controls.reset_index('assay', drop=True)
    else:
        controls = df_data.controls.loc[assay_name]
    controls = _get_closest_timepoint_for_each_well(controls, time)

    # Get average control cell count on each plate for each time point
    idx_cols = ['plate', 'cell_line', 'timepoint']
    if 'dataset' in df.index.names:
        idx_cols = ['dataset'] + idx_cols

    controls_means = controls['value'].groupby(level=idx_cols).mean()

    if include_controls:
        controls['value'] = controls['value'].groupby(level=idx_cols).apply(
            lambda x: x / x.mean())

    df.reset_index(inplace=True)
    df.rename(columns={'plate_id': 'plate'}, inplace=True)
    df.set_index(idx_cols, inplace=True)

    df = df.join(controls_means.to_frame(), rsuffix='_ctrl')
    df.reset_index(inplace=True)
    final_idx_cols = ['drug', 'cell_line', 'dose', 'well_id']
    if 'dataset' in df.columns:
        final_idx_cols = ['dataset'] + final_idx_cols
    df.set_index(final_idx_cols, inplace=True)

    df['viability'] = df['value'] / df['value_ctrl']

    timepoints = df['timepoint'].unique()
    if len(timepoints) == 1:
        time_hrs = timepoints[0].astype('timedelta64[h]').item()\
                       .total_seconds() / SECONDS_IN_HOUR

    df._viability_time = time_hrs
    df._viability_assay = assay_name

    if include_controls:
        return df, controls['value']
    else:
        return df, None


def _get_closest_timepoint_for_each_well(dataframe, timediff):
    dataframe['timediff'] = abs(dataframe.index.get_level_values('timepoint') -
                                timediff)
    dataframe = dataframe.loc[dataframe.groupby('well_id')[
        'timediff'].idxmin()]

    return dataframe
