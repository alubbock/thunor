Vanderbilt HTS file format
==========================

The Vanderbilt HTS format is the plain-text interchange format for Thunor.
Each row represents a single measurement: one well at one timepoint. Files are
tab-separated by default; use a ``.csv`` extension for comma-separated files.

Load a file with :func:`thunor.io.read_vanderbilt_hts` and save one with
:func:`thunor.io.write_vanderbilt_hts`.

Columns
-------

Required
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Column
     - Type
     - Description
   * - ``upid``
     - string
     - Unique plate identifier
   * - ``well``
     - string
     - Well name composed of a row letter and column number, e.g. ``A1``,
       ``H12``, ``P24``
   * - ``time``
     - float
     - Elapsed time in hours from the start of the experiment
   * - ``cell.count``
     - float
     - Cell count; must be ≥ 0

Drug annotation
~~~~~~~~~~~~~~~

Drug annotation columns must either all be present or all be absent. When
present, rows where all drug concentrations are zero are treated as control
wells; all other rows are experiment wells.

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Column
     - Type
     - Description
   * - ``cell.line``
     - string
     - Cell line name
   * - ``drug1``
     - string
     - Drug name; leave blank for control wells
   * - ``drug1.conc``
     - float
     - Drug concentration in molar; set to ``0`` for control wells
   * - ``drug1.units``
     - string
     - Concentration units; must be ``M``

For combination screens, include a second set: ``drug2``, ``drug2.conc``,
``drug2.units``.

Optional metadata
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Column
     - Type
     - Description
   * - ``expt.id``
     - string
     - Experiment identifier
   * - ``expt.date``
     - string
     - Experiment date in ``YYYY-MM-DD`` format

Example
-------

A minimal single-drug file with two experiment wells at two concentrations and
one control well, each measured at two timepoints (columns are tab-separated):

.. code-block:: text

   upid    well  cell.line  drug1          drug1.conc  drug1.units  time  cell.count
   Plate1  A1    MCF7       Staurosporine  1e-9        M            0     1000
   Plate1  A1    MCF7       Staurosporine  1e-9        M            24    1250
   Plate1  B1    MCF7       Staurosporine  1e-8        M            0     990
   Plate1  B1    MCF7       Staurosporine  1e-8        M            24    450
   Plate1  C1    MCF7                      0           M            0     1010
   Plate1  C1    MCF7                      0           M            24    2020

Reading this file:

.. code-block:: python

   from thunor.io import read_vanderbilt_hts
   dataset = read_vanderbilt_hts('data.txt')

Plate size
----------

By default, :func:`thunor.io.read_vanderbilt_hts` assumes a 384-well plate
(24 columns × 16 rows, so wells run from ``A1`` to ``P24``). Pass
``plate_width`` and ``plate_height`` explicitly for other sizes:

.. code-block:: python

   # 96-well plate
   dataset = read_vanderbilt_hts('data.txt', plate_width=12, plate_height=8)
