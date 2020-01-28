Performing a fit from the terminal
==================================

To perform a fit directly from the terminal, only a ``UVTable`` with the data to
be fit and a ``.json`` parameter file are needed. A ``UVTable`` can be extracted
from :py:obj:`CASA` via xx. The default parameter file is
``default_parameters.json``.

Given these files, perform a fit simply with

.. code-block:: bash

    python fit.py


A custom parameter file can alternatively be provided with

.. code-block:: bash

    python fit.py --p <parameter_filename>.json

By default :py:mod:`frank` saves the fitted brightness profile as a ``.txt``,
the visibility domain fit as a ``.npz``, ``UVTables`` for the **reprojected**
fit and its residuals, and a figure showing the fit and its diagnostics:

xx add figure with caption xx
