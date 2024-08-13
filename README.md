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

Thunor is written in pure Python and is compatible with Python 3 only.
It makes extensive use of [pandas](http://pandas.pydata.org/) and
[plotly](http://plot.ly/python/).

## Installation

Thunor Core is tested against the three most recent stable releases
of Python (currently 3.10-3.12), so one of these versions is
recommended. Install Thunor Core using `pip`:

```
pip install thunor
```

## Examples and documentation

The Thunor Core documentation is available [online](https://core.thunor.net),
or you can build it locally for offline use. To do so, clone this git
repository and change into the `thunor` directory.

To build documentation locally, you'll need a few software dependencies:

    pip install -e '.[docs]'

You'll also need to install [pandoc](https://pandoc.org/installing.html).

Then, you can build the documentation like so:

    cd doc
    make html

After the build completes, open _build/html/index.html in your web browser.

## Tutorial

Like the docs, you can also view the [tutorial](https://core.thunor.net/en/latest/tutorial.html) online.
To work through the tutorial locally, build the documentation as per the previous section. You can then open the file with Jupyter Notebook:

    jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10 doc/tutorial.ipynb

## Citation

Lubbock A.L.R., Harris L.A., Quaranta V., Tyson D.R., Lopez C.F.
[Thunor: visualization and analysis of high-throughput dose–response datasets](https://doi.org/10.1093/nar/gkab424)
Nucleic Acids Research (2021), gkab424.

## Further help and resources

See the [Thunor website](https://www.thunor.net) for further links,
documentation and related projects.
