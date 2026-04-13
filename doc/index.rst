Introduction
============

Thunor Core is a Python package for managing, analysing, and visualising
high throughput screen (HTS) data. It supports both single-timepoint
viability and the multi-timepoint drug-induced proliferation (DIP) rate
metric, a dynamic measure of drug response.

For further information on Thunor, related projects (including a
web interface, Thunor Web), and further help see the `Thunor website`_.

.. toctree::
   :hidden:
   :maxdepth: 3

   self
   installation
   vanderbilt_hts_format
   implementation
   tutorial.ipynb
   modules/index

Quickstart
----------

.. code-block:: python

    from thunor.io import read_hdf
    from thunor.dip import dip_rates
    from thunor.curve_fit import fit_params
    from thunor.plots import plot_drc

    dataset = read_hdf('my_data.h5')
    ctrl_dip_data, expt_dip_data = dip_rates(dataset)
    fp = fit_params(ctrl_dip_data, expt_dip_data)
    plot_drc(fp).show()

See the :doc:`tutorial` for a full worked example using ``hts007``, a bundled
dataset of 27 drugs on 8 breast cancer cell lines.

If you want to understand the package internals, data structures, or fitting
workflow, see :doc:`implementation`.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

.. _Thunor website: https://www.thunor.net
