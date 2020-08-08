.. |br| raw:: html

    <br>

Performing a fit
================

You can interface with Frankenstein (``frank``) to perform a fit in 2 ways:
**(1)** run the code directly from the terminal or **(2)** use the code as a Python module.

Perform a fit from the terminal
-------------------------------

To perform a quick fit from the terminal, only a UVTable with the data to
be fit and a *.json* parameter file (see below) are needed. A UVTable can be extracted
with CASA as demonstrated in `this tutorial <tutorials/mstable_to_uvtable.rst>`_.
The column format is assumed to be `u [\lambda]     v [\lambda]      Re(V) [Jy]     Im(V) [Jy]     Weight [Jy^-2]`
if the UVTable is in ASCII format (`.txt` or `.dat`),
or `u [\lambda]     v [\lambda]      V (Re + Im * 1j) [Jy]     Weight [Jy^-2]`
if the UVTable is in binary format (`.npz`).

You can quickly run a fit with the default parameter file, `default_parameters.json` (see below),
by just passing in the filename of the UVTable to be fit with the `-uv` option. The UVTable can be a `.npz`, `.txt` or `.dat`.

.. code-block:: bash

    python -m frank.fit -uv <uvtable_filename.npz>

If you want to change the default parameters, provide a custom parameter file with

.. code-block:: bash

    python -m frank.fit [-uv uvtable_filename.npz] -p <parameter_filename.json>

The default parameter file is ``default_parameters.json``. You can get it
`here <https://github.com/discsim/frank/blob/master/frank/default_parameters.json>`_,
and it looks like this,

.. literalinclude:: ../frank/default_parameters.json
    :linenos:
    :language: json

Note that anytime you run a fit without specifying `-p`, frank's internal `default_parameters.json` will be used.

You can get a description for each parameter with

.. code-block:: bash

    python -c 'import frank.fit; frank.fit.helper()'

which returns

.. literalinclude:: ../frank/parameter_descriptions.json
    :linenos:
    :language: json

That's it! frank saves (in `save_dir`) these fit outputs: |br|
- the logged messages printed during the fit as `<uvtable_filename>_frank_fit.log`, |br|
- the parameter file used in the fit as `<uvtable_filename>_frank_used_pars.json`, |br|
- optionally, the fitted brightness profile as `<uvtable_filename>_frank_profile_fit.txt`, |br|
- optionally, the visibility domain fit as `<uvtable_filename>_frank_vis_fit.npz`, |br|
- optionally, the `sol` (solution) object (see `FrankFitter <py_API.rst#frank.radial_fitters.FrankFitter>`_) as `<uvtable_filename>_frank_sol.obj` and the `iteration_diagnostics` object as `<uvtable_filename>_frank_iteration_diagnostics.obj`, |br|
- optionally, the UVTables for the **reprojected** fit and its residuals as `<uvtable_filename>_frank_uv_fit.npz` and `<uvtable_filename>_frank_uv_resid.npz`, |br|
- optionally, figures showing the fit and its diagnostics as `<uvtable_filename>_*.png`.

Here's the full figure frank produces (if `full_plot=True` in your parameter file) for a fit to the DSHARP continuum observations of the protoplanetary disc
AS 209 (`Andrews et al. 2018 <https://ui.adsabs.harvard.edu/abs/2018ApJ...869L..41A/abstract>`_).

.. figure:: plots/AS209_continuum_frank_fit_full.png
   :align: left
   :figwidth: 700

**a)** The fitted frank brightness profile. |br|
**b)** As in (a), on a log scale. The oscillations below :math:`\approx 10^9\ {\rm Jy\ sr}^{-1}` indicate the fit's noise floor. |br|
**c)** The frank profile swept over :math:`2\pi`. Note this image is not convolved with any beam. |br|
**d)** The visibility domain fit and the data in 1 and 50 :math:`{\rm k}\lambda` bins. |br|
**e)** As in (d), zooming on the longer baselines. |br|
**f)** Residuals between the binned data and the fit. The residuals' RMSE is given in the legend;
note this is being increased by the residuals beyond the baseline at which the fit walks off the data. |br|
**g)** As in (d), on a log scale. The positive and negative data and fit regions are distinguished.
On this scale it is more apparent that frank walks off the visibilities as their binned noise begins to grow strongly beyond :math:`\approx 4\ {\rm M}\lambda`. |br|
**h)** The fit's reconstructed power spectrum, the prior on the fitted brightness profile. |br|
**i)** Histogram of the binned real component of the visibilities.
Note how the bin counts drop sharply beyond :math:`\approx 4.5\ {\rm M}\lambda`,
a consequence of sparser sampling at the longest baselines. |br|
**j)** The (binned) imaginary component of the visibilities. frank only fits the real component, so if Im(V) is large,
it could indicate azimuthal asymmetry in the disc that frank will average over.

Fit for the source geometry
###########################
frank fits for the source geometry to deproject the visibilities before fitting for the brightness profile.
See `here <tutorials/fitting_procedure.ipynb#2.-Determinine-the-disc-geometry-and-deproject-the-visibilities>`_ for approaches to the deprojection.

Test the fit's sensitivity to the hyperparameters
#################################################
It's **always** important to check a fit's sensitivity to the hyperparameters :math:`\alpha` and :math:`w_{\rm smooth}`.
Often the sensitivity is quite weak, but for lower resolution or particularly noisy datasets,
the location and amplitude of substructure in the brightness profile can be sensitive to :math:`\alpha` and :math:`w_{\rm smooth}`.
You can quickly check this sensitivity by running and overplotting multiple fits in a single call to frank.
Just set `alpha` and/or `wsmooth` in the parameter file as a list of values.
See `this tutorial <tutorials/prior_sensitivity.rst>`_ for an example.

Understand the model's limitations
##################################
See `this tutorial <tutorials/model_limitations.rst>`_ for an explanation and discussion of the model's limitations,
briefly summarized here: |br|
- Noise in the visibilities can imprint noise on the reconstructed brightness profile
(in the above fit to AS 209, this is limited to the regions of flux :math:`\lesssim 10^9` Jy sr :math:`^{-1}`). |br|
- The fit does not  by default prevent regions of negative brightness in the reconstructed profile
(as seen above in the AS 209 fit's innermost gap). |br|
- The model yields a fitted brightness profile whose uncertainty is typically underestimated.
For this reason we do not show the uncertainty by default (note its absence in the above AS 209 fit).
See `here <tutorials/model_limitations.rst#an-underestimated-brightness-profile-uncertainty>`_ for a fuller discussion and mitigation approaches.

Examine the fit's convergence
#############################
Once a fit has been performed, it can be useful to check its convergence.
A convergence test on the inferred power spectrum is performed as the fit iterates,
but you can additionally examine convergence of the inferred brightness profile by setting
`diag_plot=True` in your parameter file.
frank will then produce a diagnostic figure to assess the fit's convergence.
See `this tutorial <tutorials/fit_convergence.rst>`_ for an example.

Modify the `fit.py` script
##########################
We've run this example using `frank/fit.py`; if you'd like to modify this file, you can get it `here <https://raw.githubusercontent.com/discsim/frank/master/frank/fit.py>`_.
For an 'under the hood' look at what this script does, see `this tutorial <tutorials/fitting_procedure.ipynb>`_.
If you'd like a video demonstration of the same tutorial (with sound), see `here <https://www.youtube.com/watch?v=xMxsLKQidY4&t=5>`_.

Perform a fit using `frank` as a Python module
-----------------------------------------------

To interface with the code more directly, you can use it as a module.
The wrapper functions in ``fit.py`` do everything we'll show below,
but we're going to mostly avoid those here,
just to show how to directly interface with the code's internal classes.

First import some basic stuff from frank and load the data
(again using the DSHARP observations of AS 209, available as a UVTable - with some reasonable time- and channel-averaging -
`here <https://github.com/discsim/frank/blob/master/docs/tutorials/AS209_continuum.npz>`_).

.. code-block:: python

    import matplotlib.pyplot as plt
    from frank.io import load_uvtable, save_fit
    from frank.radial_fitters import FrankFitter
    from frank.geometry import FitGeometryGaussian
    from frank.make_figs import make_quick_fig

    save_prefix = 'AS209_continuum'
    uvtable_filename = save_prefix + '.npz'
    u, v, vis, weights = load_uvtable(uvtable_filename)

Now run the fit using the `FrankFitter <py_API.rst#frank.radial_fitters.FrankFitter>`_ class.
In this example we'll fit for the disc's geometry using the `FitGeometryGaussian <py_API.rst#frank.geometry.FitGeometryGaussian>`_ class.
`FrankFitter` will then deproject the visibilities
and fit for the brightness profile. We'll fit out to 1.6" using 250 collocation points and the code's default ``alpha`` and ``weights_smooth`` hyperparameter values.

.. code-block:: python

    FF = FrankFitter(Rmax=1.6, N=250, geometry=FitGeometryGaussian(),
                     alpha=1.05, weights_smooth=1e-4)

    sol = FF.fit(u, v, vis, weights)

Now make a simplified figure showing the fit (with only subplots (a), (b), (d), (f) from the full figure above;
when running from the terminal, frank produces this figure if `quick_plot=True` in your parameter file).

.. code-block:: python

    fig, axes = make_quick_fig(u, v, vis, weights, sol, bin_widths=[1e3, 5e4], force_style=True)
    plt.savefig(save_prefix + '_frank_fit_quick.png')

which makes this,

.. figure:: plots/AS209_continuum_frank_fit_quick.png
   :align: left
   :figwidth: 700

Finally we'll save the fit results.

.. code-block:: python

    save_fit(u, v, vis, weights, sol, prefix=save_prefix)
