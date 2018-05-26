# Thunor

**Thunor** (pronounced THOO-nor) is a free software platform for managing,
visualizing, and analyzing high throughput screen (HTS) data, which measure
the dose-dependent response of cells to one or more drug(s).
Thunor has a web interface for drag-and-drop upload of cell count data, 
automatic calculation of dose response curves, and an interactive
multi-panelled plot system.

This repository, Thunor Core, is a Python package which can be used for
standalone analysis or integration into computational pipelines.

Also available is [Thunor Web](https://github.com/alubbock/thunor-web),
which builds on Thunor Core. Thunor Web is a
[Django](https://www.djangoproject.com/)-based web application which
stores datasets in a database, provides group-based access control,
a tagging system for cell lines and drugs, and an interactive plot
interface.

## Implementation

Thunor is written in pure Python and is compatible with Python 3 only.
It makes extensive use of [pandas](http://pandas.pydata.org/) and 
[plotly](http://plot.ly/python/).

## Examples

See the Jupyter notebook `tutorial.ipynb` for worked examples.
