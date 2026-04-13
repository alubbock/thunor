Viability calculations and statistics (:py:mod:`thunor.viability`)
==================================================================

The :func:`thunor.viability.viability` function calculates endpoint viability by
normalising each well to the mean control value from the same plate, cell line,
and nearest matching timepoint. This makes the function suitable for endpoint
assays or for extracting a single time slice from a time course.

The returned DataFrame is shaped so that it can be passed on to
:func:`thunor.curve_fit.fit_params` in the same way as DIP-rate outputs.

.. automodule:: thunor.viability
    :members:
