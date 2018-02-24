import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import itertools
from .dip import _choose_dip_assay, dip_rates

# assume 3x2 aspect ratio for plates


class PlateMap(object):
    """
    Representation of a High Throughput Screening plate

    Parameters
    ----------
    kwargs: dict, optional
        Optionally supply "width" and "height" values for the plate
    """
    PLATE_ASPECT_RATIO_W = 3
    PLATE_ASPECT_RATIO_H = 2

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
        """
        Number of wells in the plate
        """
        return self.width * self.height

    def row_iterator(self):
        """
        Iterate over the row letters in the plate

        Returns
        -------
        Iterator of str
            Iterator over the row letters (A, B, C, etc.)
        """
        return map(chr, range(65, 65 + self.height))

    def col_iterator(self):
        """
        Iterate over the column numbers in the plate

        Returns
        -------
        Iterator of int
            Iterator over the column numbers (1, 2, 3, etc.)
        """
        return range(1, self.width + 1)

    def well_id_to_name(self, well_id):
        """
        Convert a Well ID into a well name

        Well IDs use a numerical counter from left to right, top to
        bottom, and are zero based.

        Parameters
        ----------
        well_id: int
            Well ID on this plate

        Returns
        -------
        str
            Name for this well, e.g. A1
        """
        return '{}{}'.format(chr(65 + (well_id // self.width)),
                             (well_id % self.width) + 1)

    def well_name_to_id(self, well_name, raise_error=True):
        """
        Convert a well name to a Well ID

        Parameters
        ----------
        well_name: str
            A well name, e.g. A1
        raise_error: bool
            Raise an error if the well name is invalid if True (default),
            otherwise return -1 for invalid well names
        Returns
        -------
        int
            Well ID for this well. See also :func:`well_id_to_name`
        """
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
        """
        Iterator over the plate's wells

        Returns
        -------
        Iterator of dict
            Iterator over the wells in the plate. Each well is given as a dict
            of 'well' (well ID), 'row' (row character) and 'col' (column number)
        """
        row_it = iter(np.repeat(list(self.row_iterator()), self.width))
        col_it = itertools.cycle(self.col_iterator())
        for i in range(self.num_wells):
            yield {'well': i,
                   'row': next(row_it),
                   'col': next(col_it)}

    def well_list(self):
        """
        List of the plate's wells

        Returns
        -------
        list
            The return value of :func:`well_iterator` as a list
        """
        return list(self.well_iterator())

    @classmethod
    def plate_size_from_num_wells(cls, num_wells):
        """
        Calculate plate size from number of wells, assuming 3x2 ratio

        Parameters
        ----------
        num_wells: int
            Number of wells in a plate

        Returns
        -------
        tuple
            Width and height of plate (numbers of wells)
        """
        plate_base_unit = np.sqrt(num_wells / (cls.PLATE_ASPECT_RATIO_W *
                                               cls.PLATE_ASPECT_RATIO_H))
        plate_width = int(plate_base_unit * cls.PLATE_ASPECT_RATIO_W)
        plate_height = int(plate_base_unit * cls.PLATE_ASPECT_RATIO_H)
        return plate_width, plate_height


class PlateData(PlateMap):
    """
    A High Throughput Screening Plate with Data
    """
    def __init__(self, width=24, height=16, dataset_name=None,
                 plate_name=None, cell_lines=[], drugs=[], doses=[],
                 dip_rates=[]):
        super(PlateData, self).__init__(width=width, height=height)
        self.dataset_name = dataset_name
        self.plate_name = plate_name
        self.cell_lines = cell_lines
        self.drugs = drugs
        self.doses = doses
        self.dip_rates = dip_rates

    @classmethod
    def from_dict(cls, d):
        return cls(dataset_name=d['datasetName'],
                   plate_name=d['plateName'],
                   width=d['numCols'],
                   height=d['numRows'],
                   drugs=[w['drugs'] for w in d['wells']],
                   doses=[w['doses'] for w in d['wells']],
                   cell_lines=[w['cellLine'] for w in d['wells']],
                   dip_rates=[w['dipRate'] for w in d['wells']]
                   )


class HtsPandas(object):
    """
    High throughput screen dataset

    Represented internally using pandas dataframes

    Parameters
    ----------
    doses: pd.DataFrame
        DataFrame of doses
    assays: pd.DataFrame
        DataFrame of assays
    controls: pd.DataFrame
        DataFrame of controls

    Attributes
    ----------
    cell_lines: list
        List of cell lines in the dataset
    drugs: list
        List of drugs in the dataset
    assay_names: list
        List of assay names in the dataset
    dip_assay_name: str
        The assay name used for DIP rate calculations, e.g. "Cell count"
    """
    def __init__(self, doses, assays, controls):
        self.doses = doses
        self.assays = assays
        self.controls = controls

    def __getitem__(self, item):
        if item in ('doses', 'assays', 'controls'):
            return self.__getattribute__(item)

    def filter(self, cell_lines=None, drugs=None, plate=None):
        """
        Filter by cell lines and/or drugs

        "None" means "no filter"
        
        Parameters
        ----------
        cell_lines: Iterable, optional
            List of cell lines to filter on
        drugs: Iterable, optional
            List of drugs to filter on
        plate_name: Iterable, optional

        Returns
        -------
        HtsPandas
            A new dataset filtered using the supplied arguments
        """
        # Convert drugs to tuples if not already
        if drugs is not None:
            drugs = [(drug, ) if isinstance(drug, str) else drug
                     for drug in drugs]

        doses = self.doses.copy()
        controls = self.controls.copy()
        if plate is not None:
            if isinstance(plate, str):
                plate = [plate, ]
            doses = doses.loc[doses['plate_id'].isin(plate), :]
            controls = controls.iloc[controls.index.isin(
                plate, level='plate'), :]

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
        cell_lines = set(self.doses.index.get_level_values(
            "cell_line").unique())
        if self.controls is not None:
            cell_lines.update(self.controls.index.get_level_values(
                "cell_line").unique())
        return sorted(cell_lines)

    @property
    def drugs(self):
        return sorted(self.doses.index.get_level_values("drug").unique())

    @property
    def assay_names(self):
        return sorted(self.assays.index.get_level_values("assay").unique())

    @property
    def dip_assay_name(self):
        return _choose_dip_assay(self.assay_names)

    def plate(self, plate_name, plate_size=384, include_dip_rates=False):
        """
        Return a single plate in PlateData format

        Parameters
        ----------
        plate_name: str
            The name of a plate in the dataset
        plate_size: int
            The number of wells on the plate (default: 384)
        include_dip_rates: bool
            Calculate and include DIP rates for each well if True

        Returns
        -------
        PlateData
            The plate data for the requested plate name
        """
        new_dset = self.filter(plate=plate_name)

        dip_values = None
        if include_dip_rates:
            ctrl_dip, expt_dip = dip_rates(new_dset)

            expt_dip.reset_index(inplace=True)
            expt_dip.set_index('well_num', inplace=True)
            expt_dip = expt_dip['dip_rate']

            if ctrl_dip is not None:
                # Need to re-merge in the well numbers in the control data
                controls = new_dset.controls.loc[new_dset.dip_assay_name,
                                                 'well_num']
                controls.reset_index(['cell_line', 'plate', 'timepoint'],
                                     drop=True, inplace=True)
                controls.drop_duplicates(inplace=True)
                controls = controls.to_frame()
                ctrl_dip.reset_index(['cell_line', 'plate'], drop=True,
                                     inplace=True)

                # Merge in the well_num column
                ctrl_dip = ctrl_dip.merge(controls, left_index=True,
                                          right_index=True, how='outer')

                # Set the index to the well_num column
                ctrl_dip.reset_index(drop=True, inplace=True)
                ctrl_dip.set_index('well_num', inplace=True)
                ctrl_dip = ctrl_dip['dip_rate']

                # Merge ctrl_dip into expt_dip by well_num
                expt_dip = pd.concat([ctrl_dip, expt_dip])

            dip_values = list(expt_dip.reindex(range(plate_size)).values)

        new_dset.doses.reset_index(inplace=True)
        new_dset.doses.set_index('well_num', inplace=True)
        doses = new_dset.doses.reindex(range(plate_size))
        # Replace NaN with None
        doses = doses.where((pd.notnull(doses)), None)

        width, height = PlateMap.plate_size_from_num_wells(plate_size)

        return PlateData(
            width=width,
            height=height,
            dataset_name=None,
            plate_name=plate_name,
            cell_lines=list(doses.cell_line),
            drugs=list(doses.drug),
            doses=list(doses.dose),
            dip_rates=dip_values
        )


def read_vanderbilt_hts_single_df(file_or_source, plate_width=24,
                                  plate_height=16, sep='\t'):
    """
    Read a Vanderbilt HTS format file as a single dataframe

    See the wiki for a file format description
    
    Parameters
    ----------
    file_or_source: str or object
        Source for CSV data
    plate_width: int
        Width of the microtiter plates (default: 24, for 384 well plate)
    plate_height: int
        Width of the microtiter plates (default: 16, for 384 well plate)
    sep: str
        Source file delimiter (default: tab)

    Returns
    -------
    pd.DataFrame
        DataFrame representing the source CSV
    """
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
                        sep=None):
    """
    Read a Vanderbilt HTS format file

    See the wiki for a file format description

    Parameters
    ----------
    file_or_source: str or object
        Source for CSV data
    plate_width: int
        Width of the microtiter plates (default: 24, for 384 well plate)
    plate_height: int
        Width of the microtiter plates (default: 16, for 384 well plate)
    sep: str
        Source file delimiter (default: tab)

    Returns
    -------
    HtsPandas
        HTS Dataset containing the data read from the CSV
    """
    if sep is None:
        if not isinstance(file_or_source, str):
            raise ValueError('Need to specify file separator (\\t or ,)')
        if file_or_source.endswith('.csv'):
            sep = ','
        elif file_or_source.endswith('.tsv') or file_or_source.endswith(
                '.txt'):
            sep = '\t'
        else:
            raise ValueError('Failed to detected file separator from name. '
                             'Specify sep=\'\\t\', \',\', or other.')

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
    df_doses['well_num'] = df_doses['well']
    df_doses = df_doses.assign(well=list(
        ["{}__{}".format(a_, b_) for a_, b_ in
         zip(df_doses["upid"], df_doses["well"])]))

    if multi_drug:
        df_doses['drug1'] = df_doses[['drug1', 'drug2']].apply(tuple, axis=1)
        df_doses['drug1.conc'] = df_doses[['drug1.conc', 'drug2.conc']].apply(
            tuple, axis=1)
        df_doses.drop(['drug2', 'drug2.conc'], axis=1, inplace=True)
    else:
        df_doses[['drug1.conc', 'drug1']] = \
            df_doses.transform({'drug1.conc': lambda x: (x, ),
                                'drug1': lambda x: (x, )})

    df_doses.columns = ('plate_id', 'well_id', 'dose', 'cell_line', 'drug',
                        'well_num')
    df_doses.set_index(['drug', 'cell_line', 'dose', 'well_id'],
                       inplace=True)
    # df_doses.drop('plate_id', axis=1, inplace=True)

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
        df_controls['well_num'] = df_controls['well']
        df_controls = df_controls.assign(well=list(
            ["{}__{}".format(a_, b_) for a_, b_ in
             zip(df_controls["upid"], df_controls["well"])]))
        df_controls.columns = ['plate', 'well_id', 'cell_line', 'timepoint',
                               'value', 'well_num']
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


def write_hdf(df_data, filename, dataset_format='fixed'):
    """
    Save a dataset to Thunor HDF5 format

    Parameters
    ----------
    df_data: HtsPandas
        HTS dataset
    filename: str
        Output filename
    dataset_format: str
        One of 'fixed' or 'table'. See pandas HDFStore docs for details
    """
    with pd.HDFStore(filename, 'w', complib='zlib', complevel=9) as hdf:
        hdf.put('doses', df_data.doses_unstacked(), format=dataset_format)
        hdf.put('assays', df_data.assays, format=dataset_format)
        if df_data.controls is not None:
            hdf.put('controls', df_data.controls, format=dataset_format)


def read_hdf(filename_or_buffer):
    """
    Read a HtsPandas dataset from Thunor HDF5 format file

    Parameters
    ----------
    filename_or_buffer: str or object
        Filename or buffer from which to read the data

    Returns
    -------
    HtsPandas
        Thunor HTS dataset
    """
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

    # Aggregate multi-drugs into single column and drop the separates
    df_doses.reset_index(inplace=True)
    if 'drug' not in df_doses.columns:
        df_doses['drug'] = df_doses.filter(regex='^drug[0-9]+$', axis=1).apply(
            tuple, axis=1)
    else:
        df_doses['drug'] = df_doses['drug'].transform(lambda x: (x, ))
    if 'dose' not in df_doses.columns:
        df_doses['dose'] = df_doses.filter(regex='^dose[0-9]+$', axis=1).apply(
            tuple, axis=1)
    else:
        df_doses['dose'] = df_doses['dose'].transform(lambda x: (x, ))
    df_doses.drop(list(df_doses.filter(regex='^(dose|drug)[0-9]+$')),
                  axis=1, inplace=True)
    df_doses.set_index(['drug', 'cell_line', 'dose'], inplace=True)

    return HtsPandas(df_doses, df_assays, df_controls)
