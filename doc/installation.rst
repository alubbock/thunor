Installation
============

Thunor Core requires Python 3.11 or later. Install using pip:

    :command:`pip install thunor`

For local development, install the package in editable mode with the optional
documentation dependencies:

.. code-block:: bash

    pip install -e '.[docs]'

To build the documentation locally:

.. code-block:: bash

    cd doc
    make html

To run the test suite, including notebook validation:

.. code-block:: bash

    uv run pytest
    uv run pytest --nbval
