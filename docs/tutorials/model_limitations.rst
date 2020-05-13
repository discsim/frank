Examining the model's limitations
=================================

Noisy oscillations imprinted on brightness profile
--------------------------------------------------
Regions in the visibilities with sparse and/or sufficiently noisy sampling can cause a lack of constraint on the local spatial frequency scale,
inducing oscillations in the reconstructed brightness profile on the corresponding spatial scale.
This can potentially mimic real structure.
frank typically prevents this by damping power on scales with low SNR,
but when it does occur the oscillations in the brightness profile can be diagnosed by their frequency,
which corresponds to the unconstrained spatial frequency scale.
Varying the hyperparameter values for a fit as detailed in `this tutorial <./prior_sensitivity.rst>`_
is useful to assess and potentially suppress this behavior.

Allowed regions of negative brightness
--------------------------------------
The fitted brightness profile can have negative regions corresponding to spatial scales un- or underconstrained by the visibilities.
You can perform a fit in which the solution is forced to be positive (given the maximum a posteori powerspectrum) by using the `solve_non_negative` method provided by the solution returned by `FrankFitter` (if running frank from the terminal, set `hyperparameters : nonnegative` to `true` in your parameter file).
In tests we've seen the effect on the recovered brightness profile to typically be localized to the regions of negative flux,
with otherwise minor differences. Since enforcing the profile to be non-negative requires some extrapolation beyond the data's longest baseline the non-negative fit can be more strongly affected by the hyperprior parameter choices (particularly :math:`w_{\rm smooth}` because it affects how steeply the fit drops off at long baselines). Therefore, it is always best to compare it to the standard fit (that allows negative brightness) and check that your hyperprior parameters are reasonable.

An underestimated brightness profile uncertainty
------------------------------------------------
The uncertainty on the fitted brightness profile is typically underestimated.
For this reason we do not show the uncertainty on the reconstructed brightness profile by default.
The model framework produces an estimate of the uncertainty on the brightness profile,
but this is not reliable because reconstructing the brightness from Fourier data is an ill-posed problem.
The model's confidence interval does not typically capture a fit's ill-defined systematic uncertainty,
i.e., that due to sparse sampling in the $(u,v)$ plane.

For example if the visibility amplitude were to spike at any point beyond the data's maximum baseline,
this would imprint high amplitude variations in the brightness profile on small spatial scales.
Unless we know a priori (which is not generally the case) that the visibilities are decreasing sufficiently rapidly with increasing baseline,
the uncertainty is therefore formally infinite.
While it is reasonable to assume that for real disc brightness profiles the visibilities do decrease rapidly at long baseline,
it is not straightforward to generically extrapolate the slope of this decline beyond a dataset's longest baseline;
a robust error estimate is thus difficult to obtain.

Two effective ways to (at least coarsely) assess a brightness profile's uncertainty are:
- vary the model hyperparameters (see `this tutorial <./prior_sensitivity.rst>`_)
and examine the variation in the brightness across these different fits,
- Perform the fit for a given dataset by first truncating the data at increasingly shorter maximum baseline
(say, in steps of :math:`100\ {\rm k}\lambda`).
The variation in a given feature in the brightness profile as the maximum baseline in the dataset *increases*
can give a sense of how well resolved this feature is.
If it is highly resolved, the local brightness uncertainty can be expected to be small.
