Dose Response Curve Fitting (:py:mod:`thunor.curve_fit`)
========================================================

This module fits Hill/log-logistic dose-response curves and derives summary parameters
from them. The usual high-level entry point is :func:`thunor.curve_fit.fit_params`.

Internally, fitting code tries to remain robust in the face of sparse or noisy
screening data. Fits may be performed in an alternative parameterisation for
better numerical stability, doses can be rescaled before optimisation, and
underdetermined fits return ``None`` rather than exposing partially fitted
results.

.. automodule:: thunor.curve_fit
    :members:
