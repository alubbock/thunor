from thunor import __name__ as package_name
from thunor import __version__
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import itertools
import io
import re
from .dip import _choose_dip_assay, dip_rates

SECONDS_IN_HOUR = 3600
ZERO_TIMEDELTA = timedelta(0)
ASCII_A = 65
ALPHABET_LENGTH = 26


STANDARD_PLATE_SIZES = (96, 384, 1536)

ANNOTATION_MSG = 'Annotation information (cell.line, drug, ' + \
    'drug concentrations) must either all be present or all be absent.'


class PlateFileParseException(Exception):
    pass


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
            if self.height > 676:
                # TODO: Fail for now - would need row names like AAA, AAB etc.
                raise ValueError('Plates with height >676 are not '
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
        for i in range(min(self.height, ALPHABET_LENGTH)):
            yield chr(ASCII_A + i % ALPHABET_LENGTH)

        for i in range(ALPHABET_LENGTH, self.height):
            yield chr(ASCII_A - 1 + i // ALPHABET_LENGTH) + \
                chr(ASCII_A + i % ALPHABET_LENGTH)

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
            if len(well_name) < 2:
                raise ValueError('Well name too short')

            if len(well_name) > 2 and well_name[1].isalpha():
                row_num_mult = ord(well_name[0]) - 64 # one-based
                if row_num_mult < 0 or row_num_mult > 25:
                    raise ValueError('First letter is not capital alphanumeric')
                row_num = ord(well_name[1]) - 65  # zero-based
                row_num += (row_num_mult * 26)
                col_num_start = 2
            else:
                row_num = ord(well_name[0]) - 65  # zero-based
                col_num_start = 1

            if row_num < 0 or row_num > (self.height - 1):
                raise ValueError('Unable to parse well name {} for plate with '
                                 '{} rows'.format(well_name, self.height))

            col_num = int(well_name[col_num_start:]) - 1
            if col_num < 0 or col_num > (self.width - 1):
                raise ValueError('Unable to parse well name {} for plate with '
                                 '{} cols'.format(well_name, self.width))

            return row_num * self.width + col_num
        except ValueError:
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
        plate: Iterable, optional

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
        controls = self.controls.copy() if self.controls is not None else None
        if plate is not None:
            if isinstance(plate, str):
                plate = [plate, ]

            if 'plate' in doses.columns:
                doses.set_index('plate', append=True, inplace=True)

            doses = doses[doses.index.isin(plate, level='plate')]
            if controls is not None:
                controls = controls[controls.index.isin(plate, level='plate')]

        if cell_lines is not None:
            doses = doses.iloc[doses.index.isin(
                cell_lines, level='cell_line'), :]
            if controls is not None:
                controls = controls.iloc[controls.index.isin(
                    cell_lines, level='cell_line'), :]

        if drugs is not None:
            doses = doses.iloc[doses.index.isin(
                drugs, level='drug'
            ), :]

        doses.index = doses.index.remove_unused_levels()
        if controls is not None:
            controls.index = controls.index.remove_unused_levels()

        assays = self.assays.copy()
        assays = assays.iloc[assays.index.isin(doses['well_id'].unique(),
                                               level='well_id'), :]

        return self.__class__(doses, assays, controls)

    def __repr__(self):
        if self.doses is None:
            return "Unannotated HTS Dataset"

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

        n_drugs = doses['drug'].apply(len).max()

        if n_drugs == 1:
            # Single drug dataset
            doses['drug1'] = doses['drug'].apply(lambda x: x[0])
            doses['dose1'] = doses['dose'].apply(lambda x: x[0])
            doses.drop(['drug', 'dose'], axis=1, inplace=True)
        else:
            # Multi-drug dataset
            drug_cols = doses['drug'].apply(pd.Series)
            dose_cols = doses['dose'].apply(pd.Series)
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
                controls = new_dset.controls
                if 'dataset' in controls.index.names:
                    controls = controls.reset_index('dataset', drop=True)
                controls = controls.loc[new_dset.dip_assay_name, 'well_num']
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

            dip_values = expt_dip.reindex(range(plate_size))
            dip_values = list(dip_values.where((pd.notnull(dip_values)), None))

        new_dset.doses.reset_index(inplace=True)
        new_dset.doses.set_index('well_num', inplace=True)
        doses = new_dset.doses.reindex(range(plate_size))
        # Replace NaN with None
        doses = doses.where((pd.notnull(doses)), None)

        cell_lines = []
        ctrls = new_dset.controls.reset_index().set_index('well_num')
        for i in range(plate_size):
            if doses.cell_line[i] is not None:
                cell_lines.append(doses.cell_line[i])
            else:
                try:
                    cell_lines.append(ctrls.loc[i]['cell_line'].iloc[0])
                except KeyError:
                    cell_lines.append(None)

        width, height = PlateMap.plate_size_from_num_wells(plate_size)

        return PlateData(
            width=width,
            height=height,
            dataset_name=None,
            plate_name=plate_name,
            cell_lines=cell_lines,
            drugs=list(doses.drug),
            doses=list(doses.dose),
            dip_rates=dip_values
        )


def _time_parser(t):
    try:
        t = float(t)
    except ValueError:
        raise PlateFileParseException(
            'Error parsing time value: "{}"'.format(t))
    return timedelta(hours=t)


def _read_vanderbilt_hts_single_df(file_or_source, plate_width=24,
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

    try:
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
                             'cell.count': np.float64,
                         },
                         converters={
                             'time': _time_parser,
                             'well': lambda w: pm.well_name_to_id(w),
                             'expt.date': lambda
                                 d: datetime.strptime(
                                 d, '%Y-%m-%d').date()
                         },
                         sep=sep
                         )
    except ValueError as ve:
        errstr = str(ve)
        if errstr.startswith('Invalid well name'):
            raise PlateFileParseException(ve)
        elif errstr.startswith('could not convert string to float'):
            raise PlateFileParseException(
                'Invalid value for drug concentration ({})'.format(errstr))
        elif errstr.startswith('invalid literal for int() with base 10'):
            raise PlateFileParseException(
                'Invalid value for cell count ({})'.format(errstr))
        elif errstr.startswith('time data') and 'does not match format' in errstr:
            raise PlateFileParseException(
                'Date format should be YYYY-MM-DD ({})'.format(errstr))
        else:
            raise

    try:
        df.set_index(['upid', 'well'], inplace=True)
    except KeyError:
        raise PlateFileParseException('Please ensure columns "upid" and "well" are present')

    required_columns = {'upid', 'cell.count', 'time'}
    missing_cols = required_columns.difference(set(df.columns))
    if len(missing_cols) > 1:
        raise PlateFileParseException(
            'The following required columns are missing: {}'.format(
                ', '.join(missing_cols))
        )

    return df


def _select_csv_separator(file_or_buf):
    if not isinstance(file_or_buf, str):
        raise ValueError('Need to specify file separator (\\t or ,)')
    if file_or_buf.endswith('.csv'):
        return ','
    elif file_or_buf.endswith('.tsv') or file_or_buf.endswith('.txt'):
        return '\t'
    else:
        raise ValueError('Failed to detected file separator from name. '
                         'Specify sep=\'\\t\', \',\', or other.')


def read_vanderbilt_hts(file_or_source, plate_width=24, plate_height=16,
                        sep=None, _unstacked=False):
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
        Source file delimiter (default: detect from file extension)

    Returns
    -------
    HtsPandas
        HTS Dataset containing the data read from the CSV
    """
    if isinstance(file_or_source, pd.DataFrame):
        df = file_or_source
    else:
        if sep is None:
            sep = _select_csv_separator(file_or_source)

        df = _read_vanderbilt_hts_single_df(file_or_source, plate_width,
                                            plate_height, sep=sep)

    pm = PlateMap(width=plate_width, height=plate_height)

    # Sanity checks
    columns_with_na = set(df.columns[df.isnull().any()])
    columns_with_na = columns_with_na.difference({'drug1', 'drug2', 'drug3',
                                                  'expt.id', 'expt.date'})

    if len(columns_with_na) > 0:
        raise PlateFileParseException(
            'The following column(s) contain blank, NA, or NaN values: {}'
            .format(', '.join(columns_with_na))
        )

    try:
        if (df['cell.count'] < 0).any():
            raise PlateFileParseException('cell.count contains negative '
                                          'values')
    except KeyError:
        raise PlateFileParseException('Check for "cell.count" column header')

    try:
        if (df['time'] < ZERO_TIMEDELTA).any():
            raise PlateFileParseException('time contains negative value(s)')
    except KeyError:
        raise PlateFileParseException('Check for "time" column header')

    drug_no = 1
    drug_nums = []
    while ('drug%d' % drug_no) in df.columns.values:
        try:
            if (df['drug%d.conc' % drug_no] < 0).any():
                raise PlateFileParseException('drug%d.conc contains negative '
                                              'value(s)' % drug_no)
        except KeyError:
            raise PlateFileParseException(
                'Check for "drug{}.conc" column header'.format(drug_no))

        null_drug_names = df['drug%d' % drug_no].isnull()
        null_dose_positions = df['drug%d.conc' % drug_no].loc[null_drug_names]
        if (~null_dose_positions.isnull() & null_dose_positions != 0.0).any():
            raise PlateFileParseException(
                'Check that blank drug{} entries have blank or zero '
                'concentration also'.format(drug_no))

        if 'drug%d.units' % drug_no not in df.columns:
            raise PlateFileParseException(
                'Check for "drug{}.units" column header'.format(drug_no))

        if null_drug_names.all() and drug_no == 2 and 'drug3' not in \
                df.columns:
            break

        for du in df['drug%d.units' % drug_no].unique():
            if not isinstance(du, str) and np.isnan(du):
                continue

            if du != 'M':
                raise PlateFileParseException(
                    'Only supported drug concentration unit is M (not {})'.
                        format(du))
        drug_nums.append(drug_no)
        drug_no += 1

    has_annotation = True
    if drug_nums:
        if 'cell.line' not in df.columns:
            raise PlateFileParseException(
                'cell.line column is not present, but drug and/or dose columns '
                'are present. ' + ANNOTATION_MSG
            )
    else:
        if 'cell.line' in df.columns:
            raise PlateFileParseException(
                'drug and/or dose columns not present, but cell.line is '
                'present. ' + ANNOTATION_MSG
            )
        if 'drug1.conc' in df.columns:
            raise PlateFileParseException(
                'drug1.conc column(s) present, but drug1 column '
                'is not. ' + ANNOTATION_MSG
            )
        has_annotation = False

    # Check for duplicate drugs in any row
    if len(drug_nums) == 2:
        # Ignore rows where both concentrations are zero
        dup_drugs = df.loc[
                    ((df['drug1.conc'] != 0) | (df['drug2.conc'] != 0)) &
                    df['drug1'] == df['drug2'], :]
        if not dup_drugs.empty:
            ind_val = dup_drugs.index.tolist()[0]
            well_name = pm.well_id_to_name(ind_val[1])
            raise PlateFileParseException(
                '{} entries have the same drug listed in the same well, '
                'e.g. plate "{}", well {}'.format(
                    len(dup_drugs),
                    ind_val[0],
                    well_name
                )
            )

    # Check for duplicate time point definitions
    dup_timepoints = df.set_index('time', append=True)
    if dup_timepoints.index.duplicated().any():
        dups = dup_timepoints.loc[dup_timepoints.index.duplicated(),
               :].index.tolist()
        n_dups = len(dups)
        first_dup = dups[0]

        raise PlateFileParseException(
            'There are {} duplicate time points defined, e.g. plate "{}"'
            ', well {}, time {}'.format(
                n_dups,
                first_dup[0],
                pm.well_id_to_name(first_dup[1]),
                first_dup[2]
            )
        )

    assay_name = 'Cell count'

    if has_annotation:
        doses_cols = ["cell.line"]

        for n in drug_nums:
            doses_cols.extend(['drug{}'.format(n), 'drug{}.conc'.format(n)])
        expt_rows = np.logical_or.reduce([df["drug{}.conc".format(n)] > 0
                                         for n in drug_nums])

        df_doses = df.loc[expt_rows, doses_cols]
        # Suppress warnings about altering a dataframe slice
        df_doses.is_copy = False
        df_doses.reset_index(inplace=True)
        df_doses['well_num'] = df_doses['well']
        df_doses = df_doses.assign(well=list(
            ["{}__{}".format(a_, b_) for a_, b_ in
             zip(df_doses["upid"], df_doses["well"])]))
        df_doses = df_doses.drop_duplicates(subset='well')
        col_renames = {'drug{}.conc'.format(n): 'dose{}'.format(n) for
                                 n in drug_nums}
        col_renames.update({
            'cell.line': 'cell_line',
            'well': 'well_id',
            'upid': 'plate'
        })
        df_doses.rename(columns=col_renames, inplace=True)

        if not _unstacked:
            _stack_doses(df_doses, inplace=True)
        else:
            index_cols = ['drug{}'.format(n) for n in drug_nums]
            index_cols += ['cell_line']
            index_cols.extend(['dose{}'.format(n) for n in drug_nums])
            df_doses.set_index(index_cols, inplace=True)

        df_doses.sort_index(inplace=True)
    else:
        df_doses = None
        df['cell.line'] = None
        expt_rows = [False] * len(df.index)

    df_controls = df[np.logical_not(expt_rows)]

    if df_controls.empty:
        df_controls = None

    if df_controls is not None:
        df_controls = df_controls[["cell.line", "time", 'cell.count']]
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
    df_vals = df_vals[expt_rows]
    df_vals.index = ["{}__{}".format(a_, b_) for a_, b_ in
                     df_vals.index.tolist()]
    df_vals.index.name = 'well_id'
    df_vals.columns = ['timepoint', 'value']
    df_vals['assay'] = assay_name
    df_vals.reset_index(inplace=True)
    df_vals.set_index(['assay', 'well_id', 'timepoint'], inplace=True)

    return HtsPandas(df_doses, df_vals, df_controls)


def write_vanderbilt_hts(df_data, filename, plate_width=24,
                         plate_height=16, sep=None):
    """
    Read a Vanderbilt HTS format file

    See the wiki for a file format description

    Parameters
    ----------
    df_data: HtsPandas
        HtsPandas - HTS dataset
    filename: str or object
        filename or buffer to write into
    plate_width: int
        plate width (number of wells)
    plate_height: int
        plate height (number of wells)
    sep: str
        Source file delimiter (default: detect from file extension)
    """
    if sep is None:
        sep = _select_csv_separator(filename)

    # Check the object contains only a single dataset
    doses = df_data.doses_unstacked().reset_index()
    if 'dataset' in doses.columns:
        if len(doses['dataset'].unique()) != 1:
            raise ValueError('Cannot save object containing more than one HTS '
                             'dataset to this file format')
        doses = doses.drop(columns='dataset')

    # Construct a unified dataframe
    assays = df_data.assays.loc[df_data.dip_assay_name].reset_index()
    assays.set_index('well_id', inplace=True)

    doses.set_index('well_id', inplace=True)

    df = doses.merge(assays, how='outer', left_index=True, right_index=True)

    # Add controls, if applicable
    if df_data.controls is not None:
        controls = df_data.controls
        if 'dataset' in controls.index.names:
            if len(controls.index.get_level_values('dataset').unique()) != 1:
                raise ValueError('Cannot save object containing more than one '
                                 'HTS dataset to this file format')
            controls = controls.reset_index('dataset', drop=True)

        controls = controls.loc[df_data.dip_assay_name].reset_index()
        controls.set_index('well_id', inplace=True)
        controls['drug1'] = 'control'
        controls['dose1'] = 0.0
        if 'drug2' in df.columns:
            controls['drug2'] = 'control'
            controls['dose2'] = 0.0

        df = pd.concat([df, controls])

    pm = PlateMap(width=plate_width, height=plate_height)
    df.reset_index(drop=True, inplace=True)
    df['well_num'] = [pm.well_id_to_name(wn) for wn in
                      df['well_num'].astype(int)]
    df.rename({'plate': 'upid',
               'dose1': 'drug1.conc',
               'dose2': 'drug2.conc',
               'timepoint': 'time',
               'well_num': 'well',
               'value': 'cell.count',
               'cell_line': 'cell.line'
               },
              axis='columns', inplace=True)

    df['time'] = [td.total_seconds() / SECONDS_IN_HOUR for td in df['time']]
    df['drug1.units'] = 'M'
    if 'drug2' in df.columns:
        df['drug2.units'] = 'M'

    df.to_csv(path_or_buf=filename, sep=sep, index=False)


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
        hdf.root._v_attrs.generator = package_name
        hdf.root._v_attrs.generator_version = __version__
        hdf.put('doses', df_data.doses_unstacked(), format=dataset_format)
        hdf.put('assays', df_data.assays, format=dataset_format)
        if df_data.controls is not None:
            hdf.put('controls', df_data.controls, format=dataset_format)


def _stack_doses(df_doses, inplace=True):
    if not inplace:
        df_doses = df_doses.copy()

    # Aggregate multi-drugs into single column and drop the separates
    df_doses.reset_index(inplace=True)
    drug_cols = df_doses.filter(regex='^drug[0-9]+$', axis=1)
    dose_cols = df_doses.filter(regex='^dose[0-9]+$', axis=1)
    n_drugs = len(drug_cols.columns)
    assert n_drugs == len(dose_cols.columns)

    if n_drugs > 1:
        df_doses['drug'] = df_doses.filter(regex='^drug[0-9]+$', axis=1).apply(
            tuple, axis=1)
        df_doses['dose'] = df_doses.filter(regex='^dose[0-9]+$', axis=1).apply(
            tuple, axis=1)
    else:
        lbl_drug = 'drug' if n_drugs == 0 else 'drug1'
        df_doses['drug'] = df_doses[lbl_drug].transform(lambda x: (x, ))
        lbl_dose = 'dose' if n_drugs == 0 else 'dose1'
        df_doses['dose'] = df_doses[lbl_dose].transform(lambda x: (x, ))

    df_doses.drop(list(df_doses.filter(regex='^(dose|drug)[0-9]+$')),
                  axis=1, inplace=True)
    df_doses.set_index(['drug', 'cell_line', 'dose'], inplace=True)

    if not inplace:
        return df_doses


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
    hts_pandas = _read_hdf_unstacked(filename_or_buffer)
    _stack_doses(hts_pandas.doses, inplace=True)

    if 'dataset' in hts_pandas.doses.columns:
        hts_pandas.doses.set_index('dataset', append=True, inplace=True)

    return hts_pandas


def _read_hdf_unstacked(filename_or_buffer):
    hdf_kwargs = {'mode': 'r'}
    if isinstance(filename_or_buffer, str):
        hdf_kwargs['path'] = filename_or_buffer
    else:
        if hasattr(filename_or_buffer, 'read') and \
                callable(filename_or_buffer.read):
            filename_or_buffer = filename_or_buffer.read()
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

    # Change from plate_id to plate
    if 'plate_id' in df_doses.index.names:
        df_doses.index.rename('plate', level='plate_id', inplace=True)
    elif 'plate_id' in df_doses.columns:
        df_doses.rename(columns={'plate_id': 'plate'}, inplace=True)

    return HtsPandas(df_doses, df_assays, df_controls)


def read_incucyte(filename_or_buffer, plate_width=24, plate_height=16):
    LABEL_STR = 'Label: '
    CELL_TYPE_STR = 'Cell Type: '
    TSV_START_STR = 'Date Time\tElapsed\t'

    plate_name = 'Unnamed plate'
    cell_type = None
    if isinstance(filename_or_buffer, str):
        plate_name = filename_or_buffer
    elif hasattr(filename_or_buffer, 'name'):
        plate_name = filename_or_buffer.name

    def _incucyte_header(filedat):
        for line_no, line in enumerate(filedat):
            if line.startswith(LABEL_STR):
                new_plate_name = line[len(LABEL_STR):].strip()
                if new_plate_name:
                    plate_name = new_plate_name
            elif line.startswith(CELL_TYPE_STR):
                cell_type = line[len(CELL_TYPE_STR):].strip()
            elif line.startswith(TSV_START_STR):
                return line_no
        return None

    if isinstance(filename_or_buffer, io.BytesIO):
        filedat = io.TextIOWrapper(filename_or_buffer,
                                   encoding='utf-8')
        line_no = _incucyte_header(filedat)
        filedat.detach()
        filename_or_buffer.seek(0)
    else:
        with open(filename_or_buffer, 'r') as f:
            line_no = _incucyte_header(f)

    if line_no is None:
        raise PlateFileParseException('Does not appear to be an Incucyte '
                                      'Zoom generated file')

    dat = pd.read_csv(filename_or_buffer, skiprows=line_no, sep='\t')

    dat = dat.drop(dat.columns[0], axis=1)
    dat = dat.set_index(['Elapsed'])
    # Aggregate columns representing the same well
    dat = dat.rename(columns=lambda x: re.sub(',.*$', '', x))
    dat = dat.groupby(by=dat.columns, axis=1).agg(np.sum)

    dat = dat.stack()
    dat = dat.reset_index()
    dat.columns = ['time', 'well', 'cell.count']

    dat['time'] = pd.to_timedelta(dat['time'], unit='H')

    pm = PlateMap(width=plate_width, height=plate_height)

    try:
        dat['well'] = dat['well'].apply(pm.well_name_to_id)
    except ValueError as e:
        raise PlateFileParseException(e)

    if cell_type:
        dat['cell.line'] = cell_type

    dat['upid'] = plate_name
    dat.set_index(['upid', 'well'], inplace=True)

    return read_vanderbilt_hts(dat)
