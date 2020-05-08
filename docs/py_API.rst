The Frankenstein Python API
===============================

Geometry classes
----------------

Given a set of visibilities, together these classes: **(1)** optionally fit for the source geometry and
**(2)** deproject the visibilities by the given or fitted geometry.

.. autoclass:: frank.geometry.FixedGeometry

.. autoclass:: frank.geometry.FitGeometryGaussian

.. autoclass:: frank.geometry.SourceGeometry
 :members: apply_correction, undo_correction, deproject, reproject, fit

Fitting classes
---------------

Together these classes reconstruct the 1D radial brightness profile of a source by fitting
the deprojected visibilities.

.. autoclass:: frank.radial_fitters.FrankFitter
  :members: fit, MAP_solution, MAP_spectrum, MAP_spectrum_covariance, r, Rmax, q, Qmax, size, geometry

.. autoclass:: frank.radial_fitters._HankelRegressor
  :members: mean, covariance, power_spectrum, r, q, Rmax, Qmax, size, geometry, predict, log_likelihood

Plotting functions: Figure generation
-------------------------------------

These functions make the figures frank will produce when `quick_plot`, `full_plot` and/or `diag_plot` are `True` in your parameter file.

.. autofunction:: frank.make_figs.make_quick_fig

.. autofunction:: frank.make_figs.make_full_fig

.. autofunction:: frank.make_figs.make_diag_fig

Plotting functions: Individual plots
####################################

And these are the plotting functions those figures call.

.. autofunction:: frank.plot.plot_brightness_profile

.. autofunction:: frank.plot.plot_vis_fit

.. autofunction:: frank.plot.plot_vis

.. autofunction:: frank.plot.plot_vis_resid

.. autofunction:: frank.plot.plot_pwr_spec

.. autofunction:: frank.plot.plot_convergence_criterion

.. autofunction:: frank.plot.make_colorbar

.. autofunction:: frank.plot.plot_profile_iterations

.. autofunction:: frank.plot.plot_pwr_spec_iterations

.. autofunction:: frank.plot.plot_2dsweep

Utility functions
-----------------

These are some useful functions for various aspects of fitting and analysis.

.. autofunction:: frank.utilities.estimate_weights

.. autofunction:: frank.utilities.cut_data_by_baseline

.. autofunction:: frank.utilities.draw_bootstrap_sample

.. autofunction:: frank.utilities.sweep_profile

.. autofunction:: frank.utilities.convolve_profile
