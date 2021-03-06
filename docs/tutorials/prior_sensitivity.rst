.. |br| raw:: html

    <br>

Testing a fit's sensitivity to the hyperparameters :math:`\alpha` and :math:`w_{\rm smooth}`
============================================================================================

Context: Interpreting the model's hyperparameters
-------------------------------------------------

The hyperparameters :math:`\alpha` and :math:`w_{\rm smooth}` affect the fit's power spectrum estimate
(which is itself a prior on the fitted brightness profile).
In short, :math:`\alpha` acts as a signal-to-noise (SNR) threshold for the maximum baseline out to which frank attempts to fit the visibilities,
with a higher value imposing a more strict SNR threshold (thus fitting less of the noisiest data).
:math:`w_{\rm smooth}` sets the strength of smoothing applied to the power spectrum estimate,
with a higher value more strongly smoothing the power spectrum.

Note that the model's other three hyperparameters, :math:`R_{\rm max}`, :math:`N` and :math:`p_0`, have a negligible effect on the fit so long as: |br|
- :math:`R_{\max}` is larger than the disc's outer edge (a good choice is :math:`R_{\rm max} \gtrsim 1.5 \times` the outer edge). |br|
- :math:`N` is large enough to yield a grid of spatial frequency collocation points that extend beyond the longest baseline in the dataset
(this also ensures the nominal fit resolution in real space is sufficiently sub-beam).
Typical values are :math:`N = 100 - 300`. |br|
- :math:`p_0` is small (:math:`0 < p_0 \ll 1`; the default is :math:`10^{-15}` Jy :math:`^2`).

See the frank `methods paper <https://academic.oup.com/mnras/advance-article/doi/10.1093/mnras/staa1365/5838058?guestAccessKey=7f163a1f-c12f-4771-8e54-928636794a5b>`_ for an extended discussion of the model's hyperparameters.

Performing multiple fits to vary :math:`\alpha` and :math:`w_{\rm smooth}`
--------------------------------------------------------------------------

The default value of :math:`\alpha` is :math:`1.05`, and  we suggest varying it between :math:`1.00 - 1.30`.
(Though note that choosing :math:`\alpha = 1.00` will cause frank to fit a dataset's entire visibility distribution.
For many real datasets, the binned visibilities become noise-dominated at the longest baselines,
and fitting these baselines will imprint oscillatory artifacts on the reconstructed brightness profile.)

The default value of :math:`w_{\rm smooth}` is :math:`10^{-4}`,
and we suggest varying it between :math:`10^{-4} - 10^{-1}`.

The fastest way to assess the fit's sensitivity to :math:`\alpha` and :math:`w_{\rm smooth}` in practice is to
run a fit from the terminal (see the `Quickstart <../quickstart.rst>`_),
and set `alpha` and/or `wsmooth` as a list of values in your *.json* parameter file.
This will produce a figure that overplots the fits for all combinations of your supplied
`alpha` and `wsmooth`.

Let's do this for the DSHARP AS 209 dataset in the Quickstart,
using the extrema of the ranges we suggest for :math:`\alpha` and :math:`w_{\rm smooth}`.
That is, we'll set `alpha=[1.05, 1.30]` and `wsmooth=[1e-4, 1e-1]` in our parameter file.
This will run four fits (for the four combinations of these chosen values),
generating this figure,

.. figure:: ../plots/AS209_continuum_frank_hyperprior_sensitivity.png
   :align: left
   :figwidth: 700

**a)** The fitted frank brightness profile for each combination of hyperparameter values.
The frank brightness profile for this dataset is evidently only weakly sensitive to our choices for :math:`\alpha` and :math:`w_{\rm smooth}`.
|br|
**b)** As in (a), on a log scale.
|br|
**c)** The visibility domain fits, and the data in 1 and 50 :math:`{\rm k}\lambda` bins.
|br|
**d)** As in (c), on a log scale, which shows the relatively small variation in how the fits
handle the data as they become noise-dominated.
|br|
**e)** The reconstructed power spectrum - which is the prior on the brightness profile - for each fit.
Despite the appreciable difference in the power spectrum substructure between the :math:`\alpha = 1.05` and :math:`1.30` fits,
the visibility domain fits in (d) and brightness profiles in (a) are quite similar,
demonstrating the effect of the power spectrum as a prior is weak in this case.

Always check a fit's sensitivity to :math:`\alpha` and :math:`w_{\rm smooth}`
-----------------------------------------------------------------------------
While these sensitivities can be weak as shown above,
a fit's sensitivity to :math:`\alpha` and :math:`w_{\rm smooth}` can be nontrivial
for lower resolution or particularly noisy datasets.
In these cases, the location and amplitude of substructure in the
brightness profile can vary with :math:`\alpha` and :math:`w_{\rm smooth}`,
so it's **always** worth checking how the fit changes over the ranges `alpha=[1.05, 1.30]` and `wsmooth=[1e-4, 1e-1]`.
