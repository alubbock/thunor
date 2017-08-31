import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import itertools
import re
from .dip import choose_dip_assay


class PlateMap(object):
    def __init__(self, **kwargs):
        if 'width' in kwargs:
            self.width = kwargs['width']
        if 'height' in kwargs:
            self.height = kwargs['height']
            if self.height > 26:
                # TODO: Fail for now - would need row names like AA, AB etc.
                raise ValueError('Plates with height >26 are not yet '
                                 'supported')

    @property
    def num_wells(self):
        return self.width * self.height

    def row_iterator(self):
        return map(chr, range(65, 65 + self.height))

    def col_iterator(self):
        return range(1, self.width + 1)

    def well_id_to_name(self, well_id):
        return '{}{}'.format(chr(65 + (well_id // self.width)),
                             (well_id % self.width) + 1)

    def well_name_to_id(self, well_name, raise_error=True):
        try:
            row_num = ord(well_name[0]) - 65  # zero-based
            if row_num < 0 or row_num > (self.height - 1):
                raise ValueError('Unable to parse well name {} for plate with '
                                 '{} rows'.format(well_name, self.height))

            col_num = int(well_name[1:]) - 1
            if col_num < 0 or col_num > (self.width - 1):
                raise ValueError('Unable to parse well name {} for plate with '
                                 '{} cols'.format(well_name, self.width))

            return row_num * self.width + col_num
        except ValueError as e:
            if raise_error:
                raise ValueError('Invalid well name: {}'.format(well_name))
            else:
                return -1

    def well_iterator(self):
        row_it = iter(np.repeat(list(self.row_iterator()), self.width))
        col_it = itertools.cycle(self.col_iterator())
        for i in range(self.num_wells):
            yield {'well': i,
                   'row': next(row_it),
                   'col': next(col_it)}

    def well_list(self):
        return list(self.well_iterator())


class HtsPandas(object):
    def __init__(self, doses, assays, controls):
        self.doses = doses
        self.assays = assays
        self.controls = controls

    def __getitem__(self, item):
        if item in ('doses', 'assays', 'controls'):
            return self.__getattribute__(item)

    def filter(self, cell_lines=None, drugs=None):
        """
        Filter by cell lines and/or drugs

        "None" means "no filter"
        """
        doses = self.doses.copy()
        controls = self.controls.copy()
        if cell_lines is not None:
            doses = doses.iloc[doses.index.isin(
                cell_lines, level='cell_line'), :]
            controls = controls.iloc[controls.index.isin(
                cell_lines, level='cell_line'), :]

        if drugs is not None:
            doses = doses.iloc[doses.index.isin(
                drugs, level='drug'
            ), :]

        doses.index = doses.index.remove_unused_levels()
        controls.index = controls.index.remove_unused_levels()

        assays = self.assays.copy()
        assays = assays.iloc[assays.index.isin(doses['well_id'].unique(),
                                               level='well_id'), :]

        return self.__class__(doses, assays, controls)

    def __repr__(self):
        num_cell_lines = len(self.doses.index.get_level_values(
                             "cell_line").unique())
        num_drugs = len(self.doses.index.get_level_values("drug").unique())

        return "HTS Dataset ({} drugs/combos, {} cell lines)".format(
            num_drugs,
            num_cell_lines
        )

    def doses_unstacked(self):
        """ Split multiple drugs/doses into separate columns """
        doses = self.doses.reset_index()
        drug_cols = doses['drug'].apply(pd.Series)
        dose_cols = doses['dose'].apply(pd.Series)
        n_drugs = len(drug_cols.columns)
        drug_cols.rename(columns={n: 'drug%d' % (n + 1) for n in range(
            n_drugs)},
                         inplace=True)
        dose_cols.rename(columns={n: 'dose%d' % (n + 1) for n in range(
            n_drugs)},
                         inplace=True)
        doses.drop(['drug', 'dose'], axis=1, inplace=True)
        doses = pd.concat([doses, drug_cols, dose_cols], axis=1)
        doses.set_index(['drug%d' % (n + 1) for n in range(n_drugs)]
                        + ['cell_line'] +
                        ['dose%d' % (n + 1) for n in range(n_drugs)],
                        inplace=True)
        return doses

    @property
    def cell_lines(self):
        return sorted(self.doses.index.get_level_values("cell_line").unique())

    @property
    def drugs(self):
        return sorted(self.doses.index.get_level_values("drug").unique())

    @property
    def assay_names(self):
        return sorted(self.assays.index.get_level_values("assay").unique())

    @property
    def dip_assay_name(self):
        return choose_dip_assay(self.assay_names)


def read_vanderbilt_hts_single_df(file_or_source, plate_width=24,
                                  plate_height=16, sep='\t'):
    pm = PlateMap(width=plate_width, height=plate_height)

    df = pd.read_csv(file_or_source,
                     encoding='utf8',
                     dtype={
                         'expt.id': str,
                         'upid': str,
                         'cell.line': str,
                         'drug1': str,
                         'drug1.conc': np.float64,
                         'drug1.units': str,
                         'drug2': str,
                         'drug2.conc': np.float64,
                         'drug2.units': str,
                         'cell.count': np.int64,
                     },
                     converters={
                         'time': lambda t: timedelta(
                             hours=float(t)),
                         'well': lambda w: pm.well_name_to_id(w),
                         'expt.date': lambda
                             d: datetime.strptime(
                             d, '%Y-%m-%d').date()
                     },
                     sep=sep
                     )

    df.set_index(['upid', 'well'], inplace=True)

    return df


def read_vanderbilt_hts(file_or_source, plate_width=24, plate_height=16,
                        sep='\t'):
    df = read_vanderbilt_hts_single_df(file_or_source, plate_width,
                                       plate_height, sep=sep)

    assay_name = 'Cell count'

    multi_drug = False
    if 'drug2' in df.columns:
        multi_drug = True

    assert df["drug1.units"].unique() == 'M'
    if multi_drug:
        assert df["drug2.units"].unique() == 'M'

    doses_cols = ["drug1.conc", "cell.line", "drug1"]

    if multi_drug:
        doses_cols += ["drug2", "drug2.conc"]

    df_doses = df[doses_cols]
    if multi_drug:
        df_doses = df_doses[
            np.logical_or(df_doses["drug1.conc"] > 0,
                          df_doses["drug2.conc"] > 0)
        ]
    else:
        df_doses = df_doses[df_doses["drug1.conc"] > 0]
    # Suppress warnings about altering a dataframe slice
    df_doses.is_copy = False
    df_doses.reset_index(inplace=True)
    df_doses = df_doses.assign(well=list(
        ["{}__{}".format(a_, b_) for a_, b_ in
         zip(df_doses["upid"], df_doses["well"])]))

    if multi_drug:
        df_doses['drug1'] = df_doses[['drug1', 'drug2']].apply(tuple, axis=1)
        df_doses['drug1.conc'] = df_doses[['drug1.conc', 'drug2.conc']].apply(
            tuple, axis=1)
        df_doses.drop(['drug2', 'drug2.conc'], axis=1, inplace=True)

    df_doses.columns = ('plate_id', 'well_id', 'dose', 'cell_line', 'drug')
    df_doses.set_index(['drug', 'cell_line', 'dose', 'well_id'],
                       inplace=True)
    df_doses.drop('plate_id', axis=1, inplace=True)

    df_doses = df_doses[~df_doses.index.duplicated(keep='first')]
    df_doses.reset_index(level='well_id', inplace=True)
    df_doses.sort_index(inplace=True)

    # df_controls
    try:
        df_controls = df[df['drug1.conc'] == 0.0]
        if multi_drug:
            df_controls = df_controls[df_controls['drug2.conc'] == 0.0]
        df_controls = df_controls[["cell.line", "time", 'cell.count']]
    except KeyError:
        df_controls = None

    if df_controls is not None:
        df_controls.reset_index(inplace=True)
        df_controls = df_controls.assign(well=list(
            ["{}__{}".format(a_, b_) for a_, b_ in
             zip(df_controls["upid"], df_controls["well"])]))
        df_controls.columns = ['plate', 'well_id', 'cell_line', 'timepoint',
                               'value']
        df_controls['assay'] = assay_name
        df_controls.set_index(['assay', 'cell_line', 'plate', 'well_id',
                               'timepoint'], inplace=True)
        df_controls.sort_index(inplace=True)

    # df_vals
    df_vals = df[['time', 'cell.count']]
    df_vals = df_vals[df_vals.index.get_level_values(level='well') != 0]
    df_vals.index = ["{}__{}".format(a_, b_) for a_, b_ in
                     df_vals.index.tolist()]
    df_vals.index.name = 'well_id'
    df_vals.columns = ['timepoint', 'value']
    df_vals['assay'] = assay_name
    df_vals.reset_index(inplace=True)
    df_vals.set_index(['assay', 'well_id', 'timepoint'], inplace=True)

    return HtsPandas(df_doses, df_vals, df_controls)


def write_hdf(df_data, filename):
    with pd.HDFStore(filename, 'w', complib='zlib', complevel=9) as hdf:
        hdf.put('doses', df_data.doses_unstacked())
        hdf.put('assays', df_data.assays)
        hdf.put('controls', df_data.controls)


def read_hdf(filename_or_buffer):
    hdf_kwargs = {'mode': 'r'}
    if isinstance(filename_or_buffer, str):
        hdf_kwargs['path'] = filename_or_buffer
    else:
        hdf_kwargs.update({
            'path': 'data.h5',
            'driver': 'H5FD_CORE',
            'driver_core_backing_store': 0,
            'driver_core_image': filename_or_buffer
        })
    with pd.HDFStore(**hdf_kwargs) as hdf:
        df_assays = hdf['assays']
        try:
            df_controls = hdf['controls']
        except KeyError:
            df_controls = None
        df_doses = hdf['doses']

    df_doses.reset_index(inplace=True)
    if 'drug' not in df_doses.columns:
        df_doses['drug'] = df_doses.filter(regex='^drug[0-9]+$', axis=1).apply(
            tuple, axis=1)
    if 'dose' not in df_doses.columns:
        df_doses['dose'] = df_doses.filter(regex='^dose[0-9]+$', axis=1).apply(
            tuple, axis=1)
    df_doses = df_doses.select(lambda col: not re.match('^(dose|drug)[0-9]+$',
                                                        col),
                               axis=1)
    df_doses.set_index(['drug', 'cell_line', 'dose'], inplace=True)

    return HtsPandas(df_doses, df_assays, df_controls)
