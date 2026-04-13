DIP calculations and statistics (:py:mod:`thunor.dip`)
======================================================

The DIP module converts time-series assay measurements into drug-induced
proliferation rates. In the common workflow, :func:`thunor.dip.dip_rates`
returns a pair of tables containing control DIP rates and experiment DIP rates.
Those tables can then be passed directly to :func:`thunor.curve_fit.fit_params`.

For experiment wells, the default implementation evaluates multiple candidate
regression windows and chooses the best suffix of the time series. The current
code uses a vectorised implementation for the default selector to avoid the
cost of repeated per-well, per-window regressions.

.. automodule:: thunor.dip
    :members:
