# Thunor

**Thunor** (pronounced THOO-nor) is a free software platform for managing,
visualizing, and analyzing high throughput screen (HTS) data, which measure
the dose-dependent response of cells to one or more drug(s).

This repository, Thunor Core, is a Python package which can be used for
standalone analysis or integration into computational pipelines. There
is also a web interface, [Thunor Web](https://github.com/alubbock/thunor-web),
built around this package with added database, multi-user capabilities, drag-and-drop upload of cell count data,
automatic calculation of dose response curves, and an interactive
multi-panelled plot system ([demo](https://demo.thunor.net)).

## Implementation

Thunor makes extensive use of [pandas](http://pandas.pydata.org/) and
[plotly](http://plot.ly/python/).

The package centres around the `thunor.io.HtsPandas` container, which keeps
three aligned tables:

- `doses`: per-well annotations such as drug, dose, cell line, and plate
- `assays`: time-series measurements keyed by assay name, well, and timepoint
- `controls`: untreated or reference wells stored separately from experiment wells

Typical analysis flows look like this:

1. Load a dataset with `read_hdf()` or `read_vanderbilt_hts()`.
2. Derive response metrics with `dip_rates()` or `viability()`.
3. Fit dose-response curves with `fit_params()`.
4. Visualise the result with one of the plotting helpers in `thunor.plots`.

Internally, curve fitting uses Hill/log-logistic models from
`thunor.curve_fit`, while DIP calculations operate on log2-transformed assay
values and select the best regression window for each well.

## Installation

Thunor Core requires Python 3.11 or later. Install using `pip`:

```
pip install thunor
```

## Examples and documentation

The Thunor Core documentation is available [online](https://core.thunor.net),
or you can build it locally for offline use. To do so, clone this git
repository, then run:

    pip install -e '.[docs]'
    cd doc
    make html

After the build completes, open `_build/html/index.html` in your web browser.

The docs include:

- a worked notebook tutorial using bundled example data
- an implementation guide covering the internal data model and analysis pipeline
- API reference pages for I/O, DIP rates, viability, curve fitting, plotting,
  and format converters

## Tutorial

The [tutorial](https://core.thunor.net/en/latest/tutorial.html) is available online.
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
