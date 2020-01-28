Python API reference
====================

The documentation of each function that is reported in this page can also be directly accessed from Python with, e.g.:

.. code:: python

    from frankenstein.radial_fitters import FrankFitter

    help(FrankFitter)


Determining the disc geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first task before fitting the radial profile with Frankenstein is to determine the geometry of the disc. This is needed to de-project the visibilities so that Frankenstein can be applied to an axisymmetric source. To achieve this, an object specifying how the geometry is determined must be passed to the :class:`FrankFitter <frankenstein.radial_fitters.FrankFitter>`.

Currently, we provide two classes that can determine the geometry, :class:`FixedGeometry <frankenstein.geometry.FixedGeometry>` and :class:`FitGeometryGaussian <frankenstein.geometry.FitGeometryGaussian>`. 
:class:`FixedGeometry <frankenstein.geometry.FixedGeometry>` uses a known inclination, position angle, and phase centre. 

.. autoclass:: frankenstein.geometry.FixedGeometry

:class:`FitGeometryGaussian <frankenstein.geometry.FitGeometryGaussian>` determines the geometry by fitting a Gaussian directly to the visibilities. In this case it is only necessary to specify whether the phase centre is already known or should be fitted for.

.. autoclass:: frankenstein.geometry.FitGeometryGaussian

Adding your own geometry fit routine
####################################
It is possible to extend Frankenstein's geometry fitting capabilities with your own routines. To do this you should write your own class that inherits from the :class:`SourceGeometry <frankenstein.geometry.SourceGeometry>` base class. The :class:`SourceGeometry <frankenstein.geometry.SourceGeometry>` base class provides the interface used by Frankenstein to de-project the data, but you must implement your own :func:`fit() <frankenstein.geometry.SourceGeometry.fit>` method. This method will be called internally by :class:`FrankFitter <frankenstein.radial_fitters.FrankFitter>` to determine the geometry. 

The :func:`fit() <frankenstein.geometry.SourceGeometry.fit>` method should set the attributes :data:`_inc`,  :data:`_PA`, :data:`_dRA`, and :data:`_dDec`, which are used by the 
:func:`apply_correction() <frankenstein.geometry.SourceGeometry.apply_correction>`, 
:func:`undo_correction() <frankenstein.geometry.SourceGeometry.apply_correction>`, 
:func:`deproject <frankenstein.geometry.SourceGeometry.apply_correction>`,
and  
:func:`reproject <frankenstein.geometry.SourceGeometry.apply_correction>` methods. For the call signature of :func:`fit() <frankenstein.geometry.SourceGeometry.fit>`, see below.

.. autoclass:: frankenstein.geometry.SourceGeometry
 :members: apply_correction, undo_correction, deproject, reproject, fit


Running a fit
~~~~~~~~~~~~~
Fits are done with Frankenstein using the :class:`FrankFitter <frankenstein.radial_fitters.FrankFitter>` class. The main parameters for :class:`FrankFitter <frankenstein.radial_fitters.FrankFitter>` are `Rmax` and `N`, controlling the size of the region fit and the number of points to fit and the `geometry` object (see above). The hyperprior is controlled through the parameters `alpha`, `p_0` and `smooth`.

The fit is performed by calling the  :func:`fit() <frankenstein.radial_fitters.FrankFitter.fit>` of :class:`FrankFitter <frankenstein.radial_fitters.FrankFitter>`.

.. autoclass:: frankenstein.radial_fitters.FrankFitter
  :members: fit, MAP_solution, MAP_spectrum, MAP_spectrum_covariance, r, Rmax, q, Qmax, size, geometry

Once the fit is complete the best-fit parameters will be returned and stored in the :data:`MAP_solutution <frankenstein.radial_fitters.FrankFitter.MAP_solution>` attribute of the  :class:`FrankFitter <frankenstein.radial_fitters.FrankFitter>`. The fit is returned as a :class:`_HankelRegressor <frankenstein.radial_fitters._HankelRegressor>` object, which provides the posterior :data:`mean <frankenstein.radial_fitters._HankelRegressor.mean>`, :data:`covariance <frankenstein.radial_fitters._HankelRegressor.covariance>`, and  :data:`power spectrum <frankenstein.radial_fitters._HankelRegressor.power_spectrum>`, as well as the :data:`radius points <frankenstein.radial_fitters._HankelRegressor.r>` of the fit and the corresponding  :data:`frequency points <frankenstein.radial_fitters._HankelRegressor.q>`. Additionally, the solution object provides methods to compute the :func:`visibilities <frankenstein.radial_fitters._HankelRegressor.predict>` of the best fit model and evaluate its :func:`log-likelihood <frankenstein.radial_fitters._HankelRegressor.log_likelihood>`.

.. autoclass:: frankenstein.radial_fitters._HankelRegressor
  :members: mean, covariance, power_spectrum, r, q, Rmax, Qmax, size, geometry, predict, log_likelihood

