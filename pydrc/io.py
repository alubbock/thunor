import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import itertools
import pickle


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


def read_vanderbilt_hts_single_df(file_or_source, plate_width=24,
                                  plate_height=16):
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
                     sep="\t"
                     )

    df.set_index(['upid', 'well'], inplace=True)

    return df


def read_vanderbilt_hts(file_or_source, plate_width=24, plate_height=16):
    df = read_vanderbilt_hts_single_df(file_or_source, plate_width,
                                       plate_height)

    assay_name = 'Cell count'

    # df_doses
    df_doses = df[["drug1.conc", "cell.line", "drug1"]]
    # Suppress warnings about altering a dataframe slice
    df_doses.is_copy = False
    df_doses.drop(0, level='well', inplace=True)
    df_doses.reset_index(inplace=True)
    df_doses = df_doses.assign(well=list(
        ["{}__{}".format(a_, b_) for a_, b_ in
         zip(df_doses["upid"], df_doses["well"])]))
    df_doses.columns = ('plate_id', 'well_id', 'dose', 'cell_line', 'drug')
    df_doses.set_index(['drug', 'cell_line', 'dose', 'well_id'], inplace=True)
    df_doses.drop('plate_id', axis=1, inplace=True)

    df_doses = df_doses[~df_doses.index.duplicated(keep='first')]
    df_doses.reset_index(level='well_id', inplace=True)
    df_doses.sort_index(inplace=True)

    # df_controls
    try:
        df_controls = df[["cell.line", "time", 'cell.count']].xs(0, level='well')
    except KeyError:
        df_controls = None

    if df_controls is not None:
        df_controls.reset_index(inplace=True)
        df_controls.columns = ['plate', 'cell_line', 'timepoint', 'value']
        df_controls = df_controls.assign(well_id=list(
            ["{}__{}".format(a_, b_) for a_, b_ in
             zip(df_controls['plate'], itertools.repeat(0))]))
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

    return {'controls': df_controls, 'doses': df_doses, 'assays': df_vals,
            'dip_assay_name': assay_name}


def write_hdf(df_data, filename):
    with pd.HDFStore(filename, 'w', complib='zlib', complevel=9) as hdf:
        for key, val in df_data.items():
            if isinstance(val, pd.DataFrame):
                hdf.put(key, val, format='table')
            elif isinstance(val, str):
                hdf.root._v_attrs[key] = val
            elif val is None:
                pass
            else:
                raise ValueError('Type not supported: {}'.format(type(val)))


def read_hdf(filename_or_buffer):
    df_data = {}
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
        df_data['assays'] = hdf['assays']
        try:
            df_data['controls'] = hdf['controls']
        except KeyError:
            df_data['controls'] = None
        df_data['doses'] = hdf['doses']
        df_data['dip_assay_name'] = hdf.root._v_attrs.dip_assay_name

    return df_data
