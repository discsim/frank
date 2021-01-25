.. :history:

*Changelog*

**v.0.1.0** - *Initial stable release of frank codebase for submission to MNRAS*
#. In progress: geometry fitting routines, README, docs

**v.1.0.0** - *Initial production/stable release (version used in `Jennings et al. 2020 <https://academic.oup.com/mnras/advance-article/doi/10.1093/mnras/staa1365/5838058?guestAccessKey=7f163a1f-c12f-4771-8e54-928636794a5b>`_)*
#. default_parameters.json, parameter_descriptions.json:
   #. Adds 'initial_guess' parameter for geometry fitting routines
#. geometry.py:
   #. Adds nonparametric geometry fitting routine
   #. Adds routine to clip inclination and PA to expected range
   #. Streamlines code syntax
#. tests.py:
   #. Adds several tests
#. Docs:
   #. Adds hyperlinks, badges to README
   #. Adds code coverage
#. Miscellaneous:
   #. A few other minor bug fixes

**v.1.1.0** - *A number of big fixes, some increased flexibility in fits from terminal and figure generation, several new optional fit parameters*
#. default_parameters.json, parameter_descriptions.json:
   #. Adds additional parameters: 'asinh_a', 'fit_inc_pa', 'gamma', 'norm_residuals', 'norm_wle', 'plot_in_logx', 'stretch', 'use_median_weight'
#. fit.py:
   #. Fixes bug in which 'norm_wle' parameter was not checked
#. geometry.py:
   #. Fixes bug in 'deproject' that was redefining global variable 'u'
   #. Fixes unit conversion bug in 'FitGeometryGaussian' when inc and/or PA are user-provided, fixes 'guess' bug in __init__
   #. Adds option to only fit for dRA and dDec
#. io.py:
   #. Adds more careful checks of UVtable format
#. make_figs.py:
   #. Adds some more flexibility to figure generation, including arcsinh colorscale for 2D images
   #. Fixes a few plot generation bugs in figures
#. tests.py:
    #. Adds a few tests
#. utilities.py:
    #. Fixes bug in convolve_profile
    #. Fixes bug in estimate_weights, adds optional 'q' argument
    #. Updates sweep_profile to optionally return reprojected 2D image, adds optional 'dr' argument
    #. Updates UVDataBinner to optionally only intake real component of visibilities
    #. Adds a couple logging messages
#. Docs:
   #. Adds changelog (HISTORY.rst)
   #. Adds video tutorial
   #. Adds more verbose descriptions in some parts of docs
   #. Updates jupyter notebooks and figures to be consistent with current code version
   #. Updates paper links to point to ADS
#. Miscellaneous:
   #. Fixes 'MaskedArray' warnings in 'utilities.py', 'plot.py'
   #. A few other minor bug fixes
