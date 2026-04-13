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
