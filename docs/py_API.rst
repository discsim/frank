The Frankenstein Python API
===========================

Geometry classes
----------------

Given a set of visibilities, together these classes: **(1)** optionally fit for the source geometry and
**(2)** deproject the visibilities by the given or fitted geometry.

.. autoclass:: frank.geometry.FixedGeometry

.. autoclass:: frank.geometry.FitGeometryGaussian

.. autoclass:: frank.geometry.FitGeometryFourierBessel

.. autoclass:: frank.geometry.SourceGeometry
 :members: apply_correction, undo_correction, deproject, reproject, fit

Fitting classes
---------------

Together these classes reconstruct the 1D radial brightness profile of a source by fitting
the deprojected visibilities.

.. autoclass:: frank.radial_fitters.FrankFitter
  :members: fit, MAP_solution, MAP_spectrum, MAP_spectrum_covariance, r, Rmax, q, Qmax, size, geometry

.. autoclass:: frank.radial_fitters.FrankRadialFit
  :members: I, MAP, r, q, Rmax, Qmax, size, geometry, predict, predict_deprojected

.. autoclass:: frank.radial_fitters.FrankGaussianFit
  :members: I, mean, MAP, covariance, power_spectrum, r, q, Rmax, Qmax, size, geometry, predict, predict_deprojected, log_likelihood, solve_non_negative

.. autoclass:: frank.radial_fitters.FrankLogNormalFit
  :members: I, MAP, covariance, power_spectrum, r, q, Rmax, Qmax, size, geometry, predict, predict_deprojected, log_likelihood

.. autoclass:: frank.debris_fitters.FrankDebrisFitter
  :members: fit, MAP_solution, MAP_spectrum, MAP_spectrum_covariance, r, Rmax, q, Qmax, size, geometry


Utility functions and classes
-----------------------------

These are some useful functions and classes for various aspects of fitting and analysis.

Hankel transform
````````````````

.. autoclass:: frank.hankel.DiscreteHankelTransform
  :members: r, Rmax, q, Qmax, size, order, transform, coefficients

.. autofunction:: frank.utilities.generic_dht

.. autofunction:: frank.utilities.get_collocation_points

Unit conversion
```````````````

.. autofunction:: frank.utilities.arcsec_baseline

.. autofunction:: frank.utilities.radius_convert

.. autofunction:: frank.utilities.jy_convert

Data alteration
```````````````

.. autofunction:: frank.utilities.normalize_uv

.. autofunction:: frank.utilities.cut_data_by_baseline

Visibility binning and weights estimation
`````````````````````````````````````````

.. autoclass:: frank.utilities.UVDataBinner

.. autofunction:: frank.utilities.estimate_weights

Imaging
```````

.. autofunction:: frank.utilities.sweep_profile

.. autofunction:: frank.utilities.make_image

.. autofunction:: frank.utilities.convolve_profile

Mock data routines
``````````````````

.. autofunction:: frank.utilities.add_vis_noise

.. autofunction:: frank.utilities.make_mock_data

Statistical analysis
````````````````````

.. autofunction:: frank.utilities.draw_bootstrap_sample
