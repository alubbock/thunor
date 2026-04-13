Introduction
============

Thunor Core is a Python package for managing and viewing high
throughput screen data. It can calculate and visualize both single-timepoint
viability calculations, and the multi-timepoint drug-induced proliferation
rate (DIP rate) metric, which is a dynamic measure of drug response.

For further information on Thunor, related projects (including a
web interface, Thunor Web), and further help see the `Thunor website`_.

.. toctree::
   :hidden:
   :maxdepth: 3

   self
   installation
   tutorial.ipynb
   modules/index

Quickstart
----------

::

    from thunor.io import read_hdf
    from thunor.dip import dip_rates
    from thunor.curve_fit import fit_params
    from thunor.plots import plot_drc

    dataset = read_hdf('my_data.h5')
    ctrl_dip_data, expt_dip_data = dip_rates(dataset)
    fp = fit_params(ctrl_dip_data, expt_dip_data)
    plot_drc(fp).show()

See the :doc:`tutorial` for a full worked example with an included dataset.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

.. _Thunor website: https://www.thunor.net
