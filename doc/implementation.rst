Implementation notes
====================

This page covers the internal data structures and analysis pipeline for
readers who want to extend the package or understand how the pieces fit
together.

Data model
----------

The central container is :class:`thunor.io.HtsPandas`. It wraps three pandas
objects that move through the rest of the package together:

- ``doses`` stores per-well annotations for experiment wells. The index
  encodes drug identity, cell line, and dose; a ``well_id`` column links each
  row to the corresponding measurements in ``assays``.
- ``assays`` stores measurement time series. Its index includes assay name,
  well identifier, and timepoint.
- ``controls`` stores untreated or reference wells separately from experiment
  wells because they do not have a standard drug-dose annotation.

This separation lets downstream functions merge only the pieces they need.
For example, DIP calculations combine assay values with well annotations,
whereas viability calculations also join in plate-matched controls.

Input and output formats
------------------------

Two formats are used most often:

- :func:`thunor.io.read_hdf` and :func:`thunor.io.write_hdf` preserve the full
  internal structure and are the preferred format for round-tripping datasets.
- :func:`thunor.io.read_vanderbilt_hts` and
  :func:`thunor.io.write_vanderbilt_hts` provide a tabular interchange format
  for Vanderbilt-style HTS exports (see :doc:`vanderbilt_hts_format` for
  column definitions and an example).

The Vanderbilt reader performs substantial validation while parsing, including
well-name decoding, unit checks, duplicate well/timepoint detection, and
consistency checks between drug names and concentrations. That validation layer
is important because many later functions assume the dataset is already
structurally sound.

Analysis pipeline
-----------------

Most workflows follow the same sequence:

1. Load or construct an :class:`thunor.io.HtsPandas` dataset.
2. Choose a response metric.
3. Fit dose-response curves on that metric.
4. Plot or export the resulting parameter table.

For dynamic assays, :func:`thunor.dip.dip_rates` calculates drug-induced
proliferation (DIP) rates from time-series values. For endpoint-style analyses,
:func:`thunor.viability.viability` computes each well's response relative to the
mean control value from the same plate, cell line, and nearest matching
timepoint.

Curve fitting details
---------------------

Dose-response fitting lives in :mod:`thunor.curve_fit`. The usual entry point is
:func:`thunor.curve_fit.fit_params`, which takes control and experiment response
tables and returns a parameter DataFrame with values such as IC50, EC50, AUC,
AA, Hill coefficient, and Emax.

The lower-level :func:`thunor.curve_fit.fit_drc` helper performs a single curve
fit using a Hill/log-logistic model. The implementation includes a few practical
details worth knowing:

- it can fit in a log-EC50 parameterisation for numerical stability
- doses may be rescaled on a log scale before optimisation to reduce precision
  loss across very wide concentration ranges
- a fallback linear-EC50 fit is attempted when the preferred fit path fails
- fits that are underdetermined or numerically degenerate return ``None``
  instead of producing a partial result

``fit_params`` calls ``fit_drc`` for each drug–cell combination and then
attaches derived summary parameters and the fitted curve values needed by the
plotting code.

DIP fitting strategy
--------------------

The DIP code in :mod:`thunor.dip` works with log2-transformed assay values.
For experiment wells, the default selector evaluates every suffix of the time
series as a candidate regression window and scores each one with the
``tyson1`` function, a composite of adjusted R², RMSE, and window length that
rewards good fits over longer windows. The selection is fully vectorised,
which keeps large screens practical to analyse.

Control wells are handled separately, and the returned control and experiment
tables are shaped so that they can be fed directly into
:func:`thunor.curve_fit.fit_params`.

Plotting layer
--------------

Plot functions in :mod:`thunor.plots` expect the DataFrame conventions produced
by the analysis functions rather than arbitrary user-provided tables.
For example, :func:`thunor.plots.plot_drc` expects the parameter columns added
by :func:`thunor.curve_fit.fit_params`, including fitted models and measured
dose bounds.

Because the plotting layer is built on Plotly, the returned figures can be
shown interactively, embedded in notebooks, or further customised by callers.
