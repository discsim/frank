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
- :math:`p_0` is small (:math:`0 < p_0 \ll 1`; the default is :math:`10^{-15}`).

See `this tutorial <xx.ipynb>`_ for an extended discussion of the model's hyperparameters.

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
`alpha` and/or `wsmooth`.

Let's do this for the DSHARP AS 209 dataset in the Quickstart,
using the extrema of the ranges we suggest for :math:`\alpha` and :math:`w_{\rm smooth}`.
That is, we'll set `alpha=[1.05, 1.30]` and `wsmooth=[1e-4, 1e-1]` in our parameter file,
and frank will thus run four fits
(for the four combinations of these chosen values),
generating this figure,

.. figure:: ../plots/AS209_continuum_frank_hyperparameter_sensitivity.png
   :align: left
   :figwidth: 700

So this is good, the frank brightness profile for this dataset is negligibly sensitive to our choices for :math:`\alpha` and :math:`w_{\rm smooth}`.
Though as we note in the Quickstart, a fit's sensitivity to :math:`\alpha` and :math:`w_{\rm smooth}` can be nontrivial
for lower resolution or particularly noisy datasets.
In these cases, the location and amplitude of substructure in the
brightness profile can be sensitive to :math:`\alpha` and :math:`w_{\rm smooth}`,
so it's **always** worth checking how the fit changes in the ranges `alpha=[1.05, 1.30]` and `wsmooth=[1e-4, 1e-1]`.
