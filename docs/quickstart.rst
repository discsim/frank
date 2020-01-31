Performing a fit
================

You can interface with Frankenstein (``frank``) to perform a fit in 2 ways:
**(1)** run the code directly from the terminal or **(2)** use the code as a library.

Perform a fit from the terminal
-------------------------------

To perform a quick fit from the terminal, only a UVTable with the data to
be fit and a *.json* parameter file (see below) are needed. A UVTable can be extracted
from CASA via xx as demonstrated in `this tutorial <tutorials/xx>`_.

If you specify `load_dir`, `save_dir` and `uvtable_filename` in the default parameter file,
you can perform a fit using the default parameters with

.. code-block:: bash

    python -m frank.fit

where `-m` imports and runs frank as a package.

Or you can leave the load and save directories and the UVTable filename empty in the parameter file,
in which case the load and save directories are assumed to be your current working directory,
and you perform a fit by explicitly passing the UVTable filename with

.. code-block:: bash

    python -m frank.fit <uvtable_filename.txt>

If you want to change the default parameters, provide a custom parameter file with

.. code-block:: bash

    python -m frank.fit [uvtable_filename.txt] --p <parameter_filename.json>

The default parameter file is ``default_parameters.json``, and it looks like this:

.. literalinclude:: ../frank/default_parameters.json
    :linenos:
    :language: json

You can get a description for each parameter with

.. code-block:: bash

    python -c 'import frank.fit; frank.fit.helper()'

which returns

.. code-block:: bash

    {
        "input_output": {
            "uvtable_filename": "UV table with data to be fit. (columns: u, v, Re(V), Im(V), weights)",
            "load_dir": "Directory containing UV table",
            "save_dir": "Directory in which output datafiles and figures are saved",
            "save_profile_fit": "Whether to save fitted brightness profile",
            "save_vis_fit": "Whether to save fitted visibility distribution",
            "save_uvtables": "Whether to save fitted and residual UV tables (these are reprojected)",
            "make_plots": "Whether to make figures showing the fit and diagnostics",
            "save_plots": "Whether to save figures",
            "dist": "Distance to source, optionally used for plotting. [AU]"
        },
        "modify_data": {
            "cut_data": "Whether to truncate the visibilities at a given maximum baseline prior to fitting",
            "cut_baseline": "Maximum baseline at which visibilities are truncated"
        },
        "geometry": {
            "fit_geometry": "Whether to fit for the source's geometry (on-sky projection)",
            "known_geometry": "Whether to manually specify a geometry (if False, geometry will be fitted)",
            "fit_phase_offset": "Whether to fit for the phase center or just the inclination and position angle",
            "inc": "Inclination. [deg]",
            "pa": "Position angle. [deg]",
            "dra": "Delta (offset from 0) right ascension. [arcsec]",
            "ddec": "Delta declination. [arcsec]"
        },
        "hyperpriors": {
            "n": "Number of collocation points used in the fit (suggested range 100 - 300)",
            "rout": "Maximum disc radius in the fit (best to overestimate size of source). [arcsec]",
            "alpha": "Order parameter for the power spectrum's inverse Gamma prior (suggested range 1.00 - 1.50)",
            "p0": "Scale parameter for the power spectrum's inverse Gamma prior (suggested >0, <<1)",
            "wsmooth": "Strength of smoothing applied to the power spectrum (suggested range 10^-4 - 10^-1)"
        }
    }

That's it! By default frank saves the fitted brightness profile as a *.txt*,
the visibility domain fit as a *.npz*, UVTables for the **reprojected**
fit and its residuals as *.txt*, and 2 figures showing the fit and its diagnostics.

Here are the figures produced by a frank fit to the DSHARP continuum observations of the protoplanetary disc
AS 209 (`Andrews et al. 2018 <https://ui.adsabs.harvard.edu/abs/2018ApJ...869L..41A/abstract>`_).

 xx add figure with caption xx

Modify the `fit.py` script
##########################
We've run this example using `fit.py`; if you'd like to modify this file, you can get it `here <https://raw.githubusercontent.com/discsim/frank/master/frank/fit.py>`_.
For an 'under the hood' look at what this script does, see `this tutorial <tutorials/using_frank_as_library.ipynb>`_.
If you'd like a more qualitative overview of the code (with sound), see `here <https://www.youtube.com/watch?v=xMxsLKQidY4&t=5>`_.

Perform a fit using the code as a Python module
-----------------------------------------------

To interface with the code more directly, you can use it as a module.

First import some basic stuff from frank and load the data
(again using the DSHARP observations of AS 209, available as a UVTable
`here <https://github.com/discsim/frank/blob/master/tutorials/AS209_continuum.txt>`_).
Note that the wrapper functions in ``fit.py`` can do all this for us; we're not using them here just to show how to directly interface
with the code's internal classes.

.. code-block:: python

    from frank.constants import rad_to_arcsec
    from frank.radial_fitters import FrankFitter
    from frank.geometry import FitGeometryGaussian
    from frank.fit import load_uvdata, output_results

    u, v, vis, weights = load_uvdata('AS209_continuum.txt')

Now run the fit using the `FrankFitter <https://github.com/discsim/frank/blob/master/frank/docs/_build/html/py_API.html#frank.radial_fitters.FrankFitter>` class.
Here we'll determine the disc's geometry and deproject the visibilities
using the `FitGeometryGaussian <https://github.com/discsim/frank/blob/master/frank/docs/_build/html/py_API.html#frank.geometry.FitGeometryGaussian>` class.
For the brightness profile reconstruction we'll fit out to 1.6" using 250 collocation points and the code's
default ``alpha`` and ``weights_smooth`` hyperprior values.

.. code-block:: python

    FF = FrankFitter(Rmax=1.6/rad_to_arcsec, 250, FitGeometryGaussian(),
                     alpha=1.05, weights_smooth=1e-4)

    sol = FF.fit(u, v, vis, weights)

Finally we'll plot the real space and visibility domain fits and save them.

.. code-block:: python

    output_results(u, v, vis, weights, sol)

    xx simple figure xx
