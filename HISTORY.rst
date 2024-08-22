.. :history:

Changelog
+++++++++

v.1.2.3
+++++++
*Two bug fixes; new functions; minor, backward-compatible API changes; additional tests*
- fit.py
    - Adds call to 'utilities.check_uv' in fit runner
- test.py
    - Adds tests for `utilities` objects and for `radial_fitters.FrankFitter.log_evidence_laplace`
- utilities.py 
    - Adds function `check_uv`
    - Adds option in `get_fit_stat_uncer` to return log uncertainty for LogNormal model
    - Adds option in `make_mock_data` to reproject
    - Fixes bug in `normalize_uv`
    - Fixes two bugs in `sweep_profile`
- setup.cfg
    - Excludes scipy versions with known bug for scipy.optimize.nnls
    - Updates supported Python versions
- Docs:
    - Minor updates to tutorials using `utilities.make_mock_data`

v.1.2.2
+++++++
*One bug fix, some code refactoring, a couple new functionalities, integrates scale-height fits into command-line UI*

- default_parameters.json, parameter_descriptions.json:
    - Adds parameters: 'convergence_failure', 'I_scale', 'save_figures', 'scale_height'
- filter.py
    - Breaks 'spectral_smoothing_matrix' out of 'CriticalFilter'; some restructuring of 'covariance_MAP' and 'log_prior'
- fit.py
    - Adds 'get_scale_height' function
- hankel.py
    - Adds 'interpolation_coefficients' and 'interpolate' functions
- io.py
    - Fixed a bug that was saving incorrect uncertainty for LogNormal brightness profile
- make_figs.py
    - Adds non-negative fit to figures
- radial_fitters.py
    - Adds 'convergence_failure' arg to FrankFitter
    - Adds 'log_evidence_laplace' function to FrankFitter
- statistical_models.py
    - Adds 'DHT_coefficients' and 'interpolate' functions to 'VisibilityMapping'
- test.py
    - Adds tests for non-negative solver ('test_solve_non_negative') and solution object IO ('test_save_load_sol')
- utilities.py
    - Adds 'get_fit_stat_uncer' function

v.1.2.1
+++++++
*Fixed a bug that caused non-python files to not be installed through pip*

v.1.2.0
+++++++
*Introduction of log-normal fits, large amount of code refactoring to support both 'Gaussian' and 'LogNormal' methods*

- default_parameters.json, parameter_descriptions.json:
    - Adds parameters: 'rescale_flux', 'method'
- debris_fitters.py
    - Adds support for fitting optically thin but geometrically thick disks with a known Gaussian scale height.
- filter.py:
    - New module that now hosts the routine for optimizing for power spectrum priors, 'CriticalFilter'
- fit.py:
    - Adds ability to run either a standard or log-normal fit
- geometry.py:
    - Adds routine to rescale the total flux according to the source geometry, 'rescale_total_flux'
    - Adds 'rescale_factor' property to 'SourceGeometry'
    - Adds the option to keep all three Fourier components (u,v,w) when deprojecting
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
   - Adds a VisibilityMapping that abstracts the mapping between the brightness profile and visibilities. Handles optically thick (default), optically thin, and debris disk models.
- tests.py:
    - Adds test for a log-normal fit
- Docs:
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
