Performing a fit
================

You can interface with Frankenstein (``frank``) to perform a fit in 2 ways:
**(1)** run the code directly from the terminal or **(2)** use the code as a library.

Perform a fit from the terminal
-------------------------------

To perform a quick fit from the terminal, only a UVTable with the data to
be fit and a *.json* parameter file (see below) are needed. A UVTable can be extracted
from CASA via xx as demonstrated in `this tutorial <tutorials/xx>`_.
The column format should be `u [m]     v [m]      Re(V) [Jy]     Im(V) [Jy]     Weight`.

If you specify `load_dir`, `save_dir` and `uvtable_filename` in the default parameter file,
you can perform a fit using the default parameters with

.. code-block:: bash

    python -m frank.fit

where `-m` runs the `frank/fit` module as a script.

Alternatively you can leave any/all of the load directory, save directory and UVTable filename empty in the parameter file.
If so, the load directory will be set to your current working directory, the save directory to your load directory,
and you pass in the UVTable filename with the `-uv` option.

.. code-block:: bash

    python -m frank.fit -uv <uvtable_filename.txt>

If you want to change the default parameters, provide a custom parameter file with

.. code-block:: bash

    python -m frank.fit [-uv uvtable_filename.txt] -p <parameter_filename.json>

The default parameter file is ``default_parameters.json``, and it looks like this:

.. literalinclude:: ../default_parameters.json
    :linenos:
    :language: json

You can get a description for each parameter with

.. code-block:: bash

    python -c 'import frank.fit; frank.fit.helper()'

which returns

.. literalinclude:: ../parameter_descriptions.json
    :linenos:
    :language: json

That's it! By default frank saves (in `save_dir`) the parameter file you use as `frank_used_pars.json`,
the fitted brightness profile as `<uvtable_filename>_frank_profile_fit.txt`,
the visibility domain fit as `<uvtable_filename>_frank_vis_fit.npz`, UVTables for the **reprojected**
fit and its residuals as `<uvtable_filename>_frank_uv_fit.txt` and `<uvtable_filename>_frank_uv_resid.txt`,
and two figures showing the fit and its diagnostics as `<uvtable_filename>_fit.png` and `<uvtable_filename>_diag.png`.

Here are those figures for a frank fit to the DSHARP continuum observations of the protoplanetary disc
AS 209 (`Andrews et al. 2018 <https://ui.adsabs.harvard.edu/abs/2018ApJ...869L..41A/abstract>`_).

 xx add figure with caption xx

Perform multiple fits in a loop
###############################
You can run multiple fits in a single call to frank (e.g., to check a fit's sensitivity to hyperpriors or run a self-consistent analysis on multiple sources)
by setting one or more of the parameters in the parameter file as a list.
See `this tutorial <tutorials/running_fits_in_a_loop.ipynb>`_ for an example.

Modify the `fit.py` script
##########################
We've run this example using `fit.py`; if you'd like to modify this file, you can get it `here <https://raw.githubusercontent.com/discsim/frank/master/frank/fit.py>`_.
For an 'under the hood' look at what this script does, see `this tutorial <tutorials/using_frank_as_library.ipynb>`_.
And if you'd like a more qualitative overview of the script (with sound), see `here <https://www.youtube.com/watch?v=xMxsLKQidY4&t=5>`_.

Perform a fit using the code as a Python module
-----------------------------------------------

To interface with the code more directly, you can use it as a module.

Let's first import some basic stuff from frank and load the data
(again using the DSHARP observations of AS 209, available as a UVTable
`here <https://github.com/discsim/frank/blob/master/tutorials/AS209_continuum.txt>`_).
Note that the wrapper functions in ``fit.py`` can do all this for us; of those,
here we're not using `parse_parameters` because we'll explicitly pass the parameters we need,
and we're also not using `determine_geometry` or `perform_fit`
just to show how to directly interface with the code's internal classes.

.. code-block:: python

    from frank.constants import rad_to_arcsec
    from frank.radial_fitters import FrankFitter
    from frank.geometry import FitGeometryGaussian
    from frank.fit import load_uvdata, output_results

    u, v, vis, weights = load_uvdata('AS209_continuum.txt')

Now run the fit using the `FrankFitter <https://github.com/discsim/frank/blob/master/frank/docs/_build/html/py_API.html#frank.radial_fitters.FrankFitter>`_ class.
In this example we'll ask frank to fit for the disc's geometry using the `FitGeometryGaussian <https://github.com/discsim/frank/blob/master/frank/docs/_build/html/py_API.html#frank.geometry.FitGeometryGaussian>`_ class.
`FrankFitter <https://github.com/discsim/frank/blob/master/frank/docs/_build/html/py_API.html#frank.radial_fitters.FrankFitter>`_ will then deproject the visibilities
and fit for the brightness profile. We'll fit out to 1.6" using 250 collocation points and the code's default ``alpha`` and ``weights_smooth`` hyperprior values.

.. code-block:: python

    FF = FrankFitter(Rmax=1.6/rad_to_arcsec, 250, FitGeometryGaussian(),
                     alpha=1.05, weights_smooth=1e-4)

    sol = FF.fit(u, v, vis, weights)

Finally we'll make a simple figure of the fit and save the fit results.

.. code-block:: python

    output_results(u, v, vis, weights, sol, diag_fig=False)

xx add simple fig with caption xx
