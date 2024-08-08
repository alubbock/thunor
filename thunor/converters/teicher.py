from thunor.io import HtsPandas, write_hdf
import pandas as pd
import datetime
import os

ASSAY = 'lum:Lum'
TIMEPOINT = datetime.timedelta(hours=96)

FILE_MAIN = 'sclc_export_experiment_dose_resp_element_19FEB2016.csv'
FILE_CL = 'sclc_export_experiment_19FEB2016.csv'
FILE_DR = 'sclc_export_compound_19FEB2016.csv'


def import_teicher(directory):
    df = pd.read_csv(os.path.join(directory, FILE_MAIN),
                     dtype={'concentration': float})

    df_cl = pd.read_csv(os.path.join(directory, FILE_CL),
                        dtype={'cell_line': str})
    df = df.merge(df_cl, left_on='experiment_id', right_on='id')

    df_dr = pd.read_csv(os.path.join(directory, FILE_DR),
                        dtype={'drug_name': str})
    df = df.merge(df_dr, on='nsc')

    df = df.rename(columns={'cell_line': 'cell.line', 'drug_name': 'drug'})

    df['cell.line'] = df['cell.line'].str.replace('-', '')
    df['cell.line'] = df['cell.line'].str.replace(' ', '')
    df['cell.line'] = df['cell.line'].str.strip()
    df['drug'] = df['drug'].str.strip()
    df = df[~df['drug'].isna()]

    df['concentration'] = [(d,) for d in df['concentration'].values]
    df['drug'] = [(d,) for d in df['drug'].values]

    df.rename(columns={
        'experiment_id': 'plate',
        'concentration': 'dose',
        'cell.line': 'cell_line',
        'mean_pct_ctrl': 'value'
    }, inplace=True)

    df['well_num'] = df.groupby('plate').cumcount() + 1
    df['well_id'] = df['plate'].astype(str) + '__' + df['well_num'].astype(str)

    controls_list = []
    for plate, dat in df.groupby('plate'):
        cell_lines = dat['cell_line'].unique()
        assert len(cell_lines == 1)
        cell_line = cell_lines[0]
        controls_list.append({
            'assay': ASSAY,
            'cell_line': cell_line,
            'plate': plate,
            'well_id': '{}__{}'.format(plate, 0),
            'timepoint': TIMEPOINT,
            'value': 100.0,
            'well_num': 0
        })

    controls = pd.DataFrame(controls_list)
    controls = controls.set_index(['assay', 'cell_line', 'plate', 'well_id',
                                   'timepoint'])

    doses = df.loc[:, ['drug', 'cell_line', 'dose', 'well_id', 'plate',
                       'well_num']]
    doses = doses.set_index(['drug', 'cell_line', 'dose'])

    assays = df.loc[:, ['well_id', 'value']]
    assays['timepoint'] = TIMEPOINT
    assays['assay'] = ASSAY
    assays = assays.set_index(['assay', 'well_id', 'timepoint'])

    return HtsPandas(doses, assays, controls)


def convert_teicher(directory='.', output_file='teicher.h5'):
    """
    Convert Teicher data to Thunor format

    The "Teicher" data is a dataset of dose-response data on a panel of
    small cell lung cancer (SCLC) cell lines. The data can be downloaded from
    the following link (select the Compound Concentration/Response Data link):

    https://sclccelllines.cancer.gov/sclc/downloads.xhtml

    Unzip the downloaded file. The dataset can then be converted on the command
    line::

        python -c "from thunor.converters import convert_teicher; \
                   convert_teicher()"

    Please note that the layout of wells in each plate after conversion is
    arbitrary, since this information is not in the original files.

    This will output a file called (by default) :file:`teicher.h5`,
    which can be opened with :func:`thunor.io.read_hdf()`, or used with Thunor
    Web.

    Parameters
    ----------
    directory: str
        Directory containing the Teicher dataset
    output_file: str
        Filename of output file (Thunor HDF5 format)

    """
    hts = import_teicher(directory)
    print('Writing HDF5 file...')
    write_hdf(hts, output_file)
    print('Done!')
