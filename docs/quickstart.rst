Performing a fit
================

You can interface with Frankenstein (:py:mod:`frank`) to perform a fit in 2 ways:
**(1)** run the code directly from the terminal or **(2)** use the code as a library.

Perform a fit from the terminal
-------------------------------

To perform a quick fit from the terminal, only a ``UVTable`` with the data to
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
fit and its residuals as ``.dat``, and a figure showing the fit and its diagnostics:

xx add figure with caption xx

Perform a fit using the code as a library
-----------------------------------------

To interface with the code more directly, let's use it as a library.

First we'll load a ``UVTable`` with some example data to be fitted,
the DSHARP observations of AS 209, available as a ``UVTable``
`here <https://github.com/discsim/frankenstein/blob/master/tutorials/AS209_continuum.dat>`_.

.. code-block:: python

    u, v, vis, weights = np.genfromtxt('AS209_continuum.dat').T

Now choose an outer radius out to which we'll fit.

.. code-block:: python

    Rmax = 1.6 / rad_to_arcsec

And run the fit using the ``FrankFitter`` class. We'll choose determine the disc's
geometry and deproject the visibilities using ``FitGeometryGaussian()``.
Then for the brightness profile fit we'll use 250 collocation points and the code's
default ``alpha`` and ``weights_smooth`` hyperprior values.

.. code-block:: python

    FF = FrankFitter(Rmax, 250, FitGeometryGaussian(),
                     alpha=1.05, weights_smooth=1e-4)

    sol = FF.fit(u, v, vis, weights)

    geom = sol.geometry

We'll save the fit using xx and plot the result as xx.
