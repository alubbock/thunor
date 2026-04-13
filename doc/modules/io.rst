I/O, file reading and writing, core formats  (:py:mod:`thunor.io`)
==================================================================

The :mod:`thunor.io` module defines the package's core dataset container,
:class:`thunor.io.HtsPandas`, together with readers and writers for the HDF5
and Vanderbilt HTS formats.

The most important implementation detail is that experiment wells and control
wells are stored separately. Experiment wells live in ``doses`` plus ``assays``;
control wells live in ``controls`` plus ``assays``. This makes it possible to
keep control data available for viability normalisation and plotting without
forcing fake drug annotations onto untreated wells.

Use :func:`thunor.io.read_hdf` when you want to preserve the full internal data
model, and :func:`thunor.io.read_vanderbilt_hts` when importing tabular plate
exports. For a full description of the Vanderbilt HTS file format, including
column definitions and an example, see :doc:`../vanderbilt_hts_format`.

.. automodule:: thunor.io
    :members:
