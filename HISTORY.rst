.. :history:

Changelog
+++++++++

v.1.2.0
+++++++
*Introduction of log-normal fits, large amount of code refactoring to support both 'Gaussian' and 'LogNormal' methods*

- default_parameters.json, parameter_descriptions.json:
    - Adds parameters: 'rescale_flux', 'method'
- filter.py:
    - New module that now hosts the routine for optimizing for power spectrum priors, 'CriticalFilter'
- fit.py:
    - Adds ability to run either a standard or log-normal fit
- geometry.py:
    - Adds routine to rescale the total flux according to the source geometry, 'rescale_total_flux'
    - Adds 'rescale_factor' property to 'SourceGeometry'
- minimizer.py:
    - New module that hosts routines for solving non-linear minimization problems: 'BaseLineSearch', 'LineSearch', 'MinimizeNewton'
- radial_fitters.py:
    - Code refactoring:
        * Removes '_HankelRegressor' class
        * Adds 'FrankGaussianFit' and 'FrankLogNormalFit' classes
        * Adds 'FrankRadialFit' class
        * Moves some core functionalities to 'frank.filter' and 'frank.statistical_models'
    - Adds 'MAP', 'I', 'info' properties, and 'assume_optically_thick' parameter, to 'FrankRadialFit'
- statistical_models.py:
   - New module that now hosts 'GaussianModel' class (containing much of the functionality of the now deprecated '_HankelRegressor'), and adds analogous 'LogNormalMAPModel' class
- tests.py:
    - Adds test for a log-normal fit
- Docs:
    - Adds tutorial for log-normal fits
    - Updates API
- Miscellaneous:
    - Minor bug and typo fixes


v.1.1.0
+++++++

*A number of bug fixes, some increased flexibility in fits from terminal and figure generation, several new optional fit parameters*

- default_parameters.json, parameter_descriptions.json:
    - Adds additional parameters: 'asinh_a', 'fit_inc_pa', 'gamma', 'norm_residuals', 'norm_wle', 'plot_in_logx', 'stretch', 'use_median_weight'
- fit.py:
    - Fixes bug in which 'norm_wle' parameter was not checked
- geometry.py:
    - Fixes bug in 'deproject' that was redefining global variable 'u'
    - Fixes unit conversion bug in 'FitGeometryGaussian' when inc and/or PA are user-provided, fixes 'guess' bug in '__init__'
    - Adds option to only fit for dRA and dDec
- io.py:
    - Adds more careful checks of UVtable format
- make_figs.py:
    - Fixes a few plot generation bugs in figures
    - Adds some more flexibility to figure generation, including arcsinh colorscale for 2D image
- tests.py:
    - Adds a few tests
- utilities.py:
    - Fixes bug in 'convolve_profile'
    - Fixes bug in 'estimate_weights', adds optional 'q' argument
    - Updates 'sweep_profile' to optionally return reprojected 2D image, adds optional 'dr' argument
    - Updates 'UVDataBinner' to optionally only intake real component of visibilities
    - Adds a couple logging messages
- Docs:
    - Updates jupyter notebooks and figures to be consistent with current code version
    - Updates paper links to point to ADS
    - Adds changelog ('HISTORY.rst')
    - Adds video tutorial
    - Adds more verbose descriptions in some parts of docs
- Miscellaneous:
    - Fixes 'MaskedArray' warnings in 'utilities.py', 'plot.py'
    - Fixes a few other minor bugs

v.1.0.0
+++++++

*Initial production/stable release as used in* `Jennings et al. 2020 <https://academic.oup.com/mnras/advance-article/doi/10.1093/mnras/staa1365/5838058?guestAccessKey=7f163a1f-c12f-4771-8e54-928636794a5b>`_

- default_parameters.json, parameter_descriptions.json:
    - Adds 'initial_guess' parameter for geometry fitting routines
- geometry.py:
    - Adds nonparametric geometry fitting routine
    - Adds routine to clip inclination and PA to expected range
- tests.py:
    - Adds several tests
- Docs:
    - Adds hyperlinks, badges to README
    - Adds code coverage
- Miscellaneous:
    - Fixes a few other minor bugs

v.0.1.0
+++++++

*Initial release of frank codebase for submission to MNRAS*

- In progress: geometry fitting routines, README, docs
