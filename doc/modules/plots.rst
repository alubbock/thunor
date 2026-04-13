Plots and visualization (:py:mod:`thunor.plots`)
================================================

The plotting helpers build Plotly figures from the standard DataFrame shapes
produced by :mod:`thunor.dip`, :mod:`thunor.viability`, and
:mod:`thunor.curve_fit`. They are designed to sit at the end of the analysis
pipeline rather than accept arbitrary user tables.

In practice, :func:`thunor.plots.plot_drc` is the main entry point for fitted
dose-response curves, while the other helpers focus on parameter summaries,
time courses, and plate-level views.

.. automodule:: thunor.plots
    :members:
