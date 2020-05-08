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

Utility functions and classes
-----------------------------

These are some useful functions and classes for various aspects of fitting and analysis.

.. autofunction:: frank.utilities.arcsec_baseline


.. autofunction:: frank.utilities.normalize_uv

.. autofunction:: frank.utilities.cut_data_by_baseline

.. autofunction:: frank.utilities.estimate_weights

.. autofunction:: frank.utilities.draw_bootstrap_sample

.. autofunction:: frank.utilities.sweep_profile

.. autofunction:: frank.utilities.convolve_profile

.. autoclass:: frank.utilities.UVDataBinner
