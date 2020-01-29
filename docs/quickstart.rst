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

Then perform a fit from `/frankenstein` [xx update xx] with

.. code-block:: bash

    python fit -m full_path_to_uvtable.dat

where `-m` imports and runs :py:mod:`frank` as a package, and the ``UVTable`` can be
given without the full path if it's in `/frankenstein` [xx update xx]. You can also
specify the load and save directories and the ``UVTable`` filename in the parameter file.

Want to change other parameters? Provide a custom parameter file with

.. code-block:: bash

    python fit -m --p <parameter_filename>

By default :py:mod:`frank` saves the fitted brightness profile as a ``.txt``,
the visibility domain fit as a ``.npz``, ``UVTables`` for the **reprojected**
fit and its residuals as ``.dat``, and a figure showing the fit and its diagnostics:

xx add figure with caption xx

Perform a fit using the code as a library
-----------------------------------------

To interface with the code more directly, you can use it as a library.

First import some basic stuff from ``frank`` and load a ``UVTable`` with the data to be fit
(here we'll use the DSHARP observations of AS 209, available as a ``UVTable``
`here <https://github.com/discsim/frankenstein/blob/master/tutorials/AS209_continuum.dat>`_). For ease we'll use the
:py:func:`load_uvdata` function.

.. code-block:: python

    import numpy as np

    import frankenstein as frank
    from frank.constants import rad_to_arcsec
    from frank.radial_fitters import FrankFitter
    from frank.geometry import FitGeometryGaussian
    from fit import load_uvdata xx update xx

    u, v, vis, weights = load_uvdata('AS209_continuum.dat')

Now choose an outer radius out to which :py:mod:`frank` will fit.

.. code-block:: python

    Rmax = 1.6 / rad_to_arcsec

Then run the fit using the ``FrankFitter`` class. Here we'll determine the disc's
geometry and deproject the visibilities using ``FitGeometryGaussian()``.
For the brightness profile fit we'll use 250 collocation points and the code's
default ``alpha`` and ``weights_smooth`` hyperprior values.

.. code-block:: python

    FF = FrankFitter(Rmax, 250, FitGeometryGaussian(),
                     alpha=1.05, weights_smooth=1e-4)

    sol = FF.fit(u, v, vis, weights)

Finally we'll plot the real space and visibility domain fits using xx and save them using xx.
