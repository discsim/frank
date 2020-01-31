The Frankenstein Python API
===============================

To interface with the code's fitting routine, use the
:func:`FrankFitter <frank.radial_fitters.FrankFitter>` object.

.. autofunction:: frank.radial_fitters.FrankFitter

.. note ::

    Documentation such as this can also be accessed in Python with, e.g.,

    .. code:: python

Determining the disc geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first task before fitting the radial profile with Frankenstein is to determine the geometry of the disc. This is needed to de-project the visibilities so that Frankenstein can be applied to an axisymmetric source. To achieve this, an object specifying how the geometry is determined must be passed to the :class:`FrankFitter <frank.radial_fitters.FrankFitter>`.

Currently, we provide two classes that can determine the geometry, :class:`FixedGeometry <frank.geometry.FixedGeometry>` and :class:`FitGeometryGaussian <frank.geometry.FitGeometryGaussian>`.
:class:`FixedGeometry <frank.geometry.FixedGeometry>` uses a known inclination, position angle, and phase centre.

Geometry classes
----------------

Given a set of visibilities, together these classes: **(1)** optionally fit for the source geometry and
**(2)** deproject the visibilities by the given or fitted geometry.

>>>>>>> format API docs page
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
