# Thunor

**Thunor** (pronounced THOO-nor) is a free software platform to manage,
visualise, and analyse high throughput screen (HTS) data, which measures
the dose-dependent response of cells to one or more drugs.

This repository, Thunor Core, is a Python package which can be used for
standalone analysis or integration into computational pipelines. There is
also a web interface, [Thunor Web](https://github.com/alubbock/thunor-web),
built around this package with added database, multi-user capabilities,
drag-and-drop upload of cell count data, automatic calculation of dose
response curves, and an interactive multi-panelled plot system
([demo](https://demo.thunor.net)).

## Overview

Thunor makes extensive use of [pandas](http://pandas.pydata.org/) and
[plotly](http://plot.ly/python/) to manage HTS data at scale.

The data model centres around the `thunor.io.HtsPandas` container, which
internally keeps three aligned tables:

- `doses`: per-well experiment annotations (drug identity, cell line, dose)
- `assays`: time-series measurements indexed by assay name, well, and timepoint
- `controls`: untreated wells stored separately, available for normalisation

A typical analysis looks like this:

```python
from thunor.io import read_hdf
from thunor.dip import dip_rates
from thunor.curve_fit import fit_params
from thunor.plots import plot_drc

dataset = read_hdf('my_data.h5')
ctrl_dip_data, expt_dip_data = dip_rates(dataset)
fp = fit_params(ctrl_dip_data, expt_dip_data)
plot_drc(fp).show()
```

1. Load a dataset with `read_hdf()` or `read_vanderbilt_hts()`.
2. Derive response metrics with `dip_rates()` or `viability()`.
3. Fit dose-response curves with `fit_params()`.
4. Visualise the result with one of the plotting helpers in `thunor.plots`.

## Input formats

Thunor reads two formats:

- **HDF5** (`.h5`): the native round-trip format, written by `write_hdf()`.
- **Vanderbilt HTS** (`.txt`/`.tsv`/`.csv`): a tab-separated text format
  where each row is one well at one timepoint.

A minimal Vanderbilt HTS file (columns are tab-separated):

```
upid    well  cell.line  drug1          drug1.conc  drug1.units  time  cell.count
Plate1  A1    MCF7       Staurosporine  1e-9        M            0     1000
Plate1  A1    MCF7       Staurosporine  1e-9        M            24    1250
Plate1  B1    MCF7                      0           M            0     1010
Plate1  B1    MCF7                      0           M            24    2020
```

Rows where `drug1.conc` is 0 are treated as control wells. See the
[format reference](https://core.thunor.net/en/latest/vanderbilt_hts_format.html)
for full column details.

## Installation

Thunor Core requires Python 3.11 or later. Install using `pip`:

```
pip install thunor
```

## Examples and documentation

View the [Thunor Core documentation](https://core.thunor.net) online,
or you can build it locally for offline use. To do so, clone this git
repository, then run:

    pip install -e '.[docs]'
    cd doc
    make html

After the build completes, open `_build/html/index.html` in your web browser.

The docs include:

- a worked notebook tutorial using `hts007`, a bundled breast cancer DIP rate screen
- a Vanderbilt HTS format reference with column definitions and an example
- an implementation guide covering the internal data model and analysis pipeline
- API reference pages for I/O, DIP rates, viability, curve fitting, plotting,
  and format converters

## Tutorial

The [tutorial](https://core.thunor.net/en/latest/tutorial.html) walks through
a complete DIP rate analysis using `hts007`, a bundled dataset of 27 drugs
tested on 8 breast cancer cell lines with cell count measurements taken over
approximately 5 days.

To work through it locally, install the package and open the notebook directly:

    pip install thunor
    jupyter notebook doc/tutorial.ipynb

## Citation

Lubbock A.L.R., Harris L.A., Quaranta V., Tyson D.R., Lopez C.F.
[Thunor: visualization and analysis of high-throughput dose–response datasets](https://doi.org/10.1093/nar/gkab424)
Nucleic Acids Research (2021), gkab424.

## Further help and resources

See the [Thunor website](https://www.thunor.net) for further links,
documentation and related projects.
