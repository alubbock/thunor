Conversion tools for external formats and databases (:py:mod:`thunor.converters`)
=================================================================================

The converter functions reshape external public datasets into the Thunor HDF5
format so that they can be opened with :func:`thunor.io.read_hdf` and analysed
with the rest of the package. The following datasets are supported:

- **GDSC** (Genomics of Drug Sensitivity in Cancer): a large-scale 72-hour
  viability screen from the Wellcome Sanger Institute.
  :func:`thunor.converters.convert_gdsc`

- **CTRP v2.0** (Cancer Therapeutics Response Portal): a 72-hour viability
  screen from the Broad Institute CTD² project.
  :func:`thunor.converters.convert_ctrp`

- **Teicher SCLC panel**: 96-hour viability data for a small cell lung cancer
  cell line panel from the NCI.
  :func:`thunor.converters.convert_teicher`

Each converter function's docstring describes where to download the required
source files.

.. automodule:: thunor.converters
    :members:
