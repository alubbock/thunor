import pandas as pd
import numpy as np
from datetime import timedelta
import collections
from thunor.io import HtsPandas, write_hdf

# The data are 72 hour viability
TIMEPOINT = timedelta(hours=72)
# Assay name (generic luminescence)
ASSAY = 'lum:Lum'
# Number of measurement wells
NUM_RAW = 9
# Number of control wells
NUM_CONTROLS = 48
# The first well to use for experimental data (must be > NUM_CONTROLS)
START_WELL_EXPT = 50
# Conversion factor for well concentrations to molar (from micromolar)
WELL_CONVERSION = 1e-6


def _ctrl_well(df, num_controls):
    well_vals = {}
    for row in df.itertuples():
        for i in range(num_controls):
            well_id = '{}__{}'.format(row.BARCODE, i)
            ctrl_val = getattr(row, 'control{}'.format(i + 1))
            if ctrl_val == 'qc_fail' or np.isnan(ctrl_val):
                continue
            if well_id in well_vals:
                assert well_vals[well_id] == ctrl_val
            else:
                well_vals[well_id] = ctrl_val
                yield {
                    'assay': ASSAY,
                    'cell_line': row.CELL_LINE_NAME,
                    'plate': row.BARCODE,
                    'well_id': well_id,
                    'timepoint': TIMEPOINT,
                    'value': ctrl_val,
                    'well_num': i
                }


def _get_controls(df, num_controls):
    controls = pd.DataFrame(_ctrl_well(df, num_controls))
    controls.set_index(['assay', 'cell_line', 'plate', 'well_id',
                        'timepoint'], inplace=True)

    return controls


def import_gdsc(drug_list_file, screen_data_file):
    print('This process may take several minutes, please be patient.')
    print('Reading drug names...')
    drug_list = pd.read_excel(drug_list_file, converters={
        'Drug Name': str
    })

    drug_ids = drug_list.loc[:, ('Drug ID', 'Drug Name')]
    drug_ids.set_index('Drug ID', inplace=True)

    print('Reading HTS data...')
    screen_data = pd.read_excel(screen_data_file, converters={
        'BARCODE': str,
        'CELL_LINE_NAME': str
    })

    print('Merging data...')
    df = screen_data

    # Drop the blank wells (no cells, no drugs)
    df.drop(list(df.filter(regex='blank\d+')), axis=1, inplace=True)

    # Merge in the drug names
    df = df.merge(drug_ids, left_on='DRUG_ID', right_index=True)

    df.rename(columns={'Drug Name': 'DRUG_NAME',
                       'raw_max': 'raw1'}, inplace=True)

    powers = list(range(NUM_RAW))

    doses_list = []
    assay_list = []

    plate_counter = collections.Counter()

    print('Extracting controls...')
    controls = _get_controls(df, NUM_CONTROLS)

    print('Extracting data...')
    for row in df.itertuples():
        # Create the dilution series for this row
        concs = 1 / np.power(row.FOLD_DILUTION, powers) * WELL_CONVERSION * \
                row.MAX_CONC

        start_well = plate_counter[row.BARCODE]
        for i, conc in enumerate(concs):
            well_num = i + START_WELL_EXPT + start_well
            well_id = '{}__{}'.format(row.BARCODE, well_num)
            well_val = getattr(row, 'raw{}'.format(i+1))
            doses_list.append({
                'drug': (row.DRUG_NAME, ),
                'cell_line': row.CELL_LINE_NAME,
                'dose': (conc, ),
                'well_id': well_id,
                'plate': row.BARCODE,
                'well_num': well_num
            })

            if not np.isnan(well_val):
                assay_list.append({
                    'assay': ASSAY,
                    'well_id': well_id,
                    'timepoint': TIMEPOINT,
                    'value': well_val
                })
                # assert well_vals.setdefault(well_id, well_val) == well_val

        plate_counter[row.BARCODE] += NUM_RAW

    print('Creating Thunor dataset...')
    doses = pd.DataFrame(doses_list)
    doses.set_index(['drug', 'cell_line', 'dose'], inplace=True)
    assays = pd.DataFrame(assay_list)
    assays.set_index(['assay', 'well_id', 'timepoint'], inplace=True)

    return HtsPandas(doses, assays, controls)


def convert_gdsc_tags(cell_line_file='Cell_Lines_Details.xlsx',
                      output_file='gdsc_cell_line_primary_site_tags.txt'):
    """
    Convert GDSC cell line tissue descriptors to Thunor tags

    GDSC is the Genomics of Drug Sensitivity in Cancer, a project which has
    generated a large quantity of viability data.

    The data are freely available under the license agreement described on
    their website:

    https://www.cancerrxgene.org/downloads

    The required files can be downloaded from here:

    ftp://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/release-6.0/

    You'll need to download one file:

    * Cell line details, "Cell_Lines_Details.xlsx"

    You can run this function at the command line to convert the files;
    assuming the downloaded file is in the current directory, simply run::

        python -c "from thunor.converters import convert_gdsc_tags; convert_gdsc_tags()"

    This will output a file called (by default)
    :file:`gdsc_cell_line_primary_site_tags.txt`, which can be loaded into
    Thunor Web using the "Upload cell line tags" function.

    Parameters
    ----------
    cell_line_file: str
        Filename of GDSC cell line details (Excel .xlsx format)
    output_file: str
        Filename of output file (tab separated values format)

    """
    df = pd.read_excel(cell_line_file,
                       sheet_name='Cell line details')
    cl_column = 'Sample Name'
    tissue_column = 'GDSC\nTissue descriptor 1'
    df = df[[cl_column, tissue_column]]
    df.rename(columns={cl_column: 'cell_line',
                       tissue_column: 'tag_name'},
              inplace=True)
    df['tag_category'] = 'GDSC primary site'
    df.to_csv(output_file, sep='\t', index=False)


def convert_gdsc(drug_list_file='Screened_Compounds.xlsx',
                 screen_data_file='v17a_public_raw_data.xlsx',
                 output_file='gdsc-v17a.h5'):
    """
    Convert GDSC data to Thunor format

    GDSC is the Genomics of Drug Sensitivity in Cancer, a project which has
    generated a large quantity of viability data.

    The data are freely available under the license agreement described on
    their website:

    https://www.cancerrxgene.org/downloads

    The required files can be downloaded from here:

    ftp://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/release-6.0/

    You'll need to download two files to convert to Thunor format:

    * The list of drugs, "Screened_Compounds.xlsx"
    * Sensitivity data, "v17a_public_raw_data.xlsx"

    Please note that the layout of wells in each plate after conversion is
    arbitrary, since this information is not in the original files.

    Please make sure you have the "tables" and "xlrd" python packages installed,
    in addition to the standard Thunor Core requirements.

    You can run this function at the command line to convert the files;
    assuming the two files are in the current directory, simply run::

        python -c "from thunor.converters import convert_gdsc; convert_gdsc()"

    This script will take several minutes to run, please be patient. It is also
    resource-intensive, due to the size of the dataset. We recommend you utilize
    the highest-spec machine that you have available.

    This will output a file called (by default) :file:`gdsc-v17a.h5`,
    which can be opened with :func:`thunor.io.read_hdf()`, or used with Thunor
    Web.

    Parameters
    ----------
    drug_list_file: str
        Filename of GDSC list of drugs, to convert drug IDs to names
    screen_data_file: str
        Filename of GDSC sensitivity data
    output_file: str
        Filename of output file (Thunor HDF5 format)

    """
    hts = import_gdsc(drug_list_file, screen_data_file)
    print('Writing HDF5 file...')
    write_hdf(hts, output_file)
    print('Done!')
