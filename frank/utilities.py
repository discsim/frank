# Frankenstein: 1D disc brightness profile reconstruction from Fourier data
# using non-parametric Gaussian Processes
#
# Copyright (C) 2019-2020  R. Booth, J. Jennings, M. Tazzari
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
#
"""This module has functions that are useful for plotting and analyzing fit
results.
"""
import logging
import numpy as np
from scipy.interpolate import interp1d

def arcsec_baseline(x):
    """
    Provide x as a radial scale [arcsec] to return the corresponding baseline
    [lambda], or vice-versa

    Parameters
    ----------
    x : float
        Radial scale [arcsec] or baseline [lambda]

    Returns
    -------
    converted : float
        Baseline [lambda] or radial scale [arcsec]
    """

    converted = 1 / (x / 60 / 60 * np.pi / 180)

    return converted


class UVDataBinner(object):
    r"""
    Average uv-data into bins of equal size.

    Compute the weighted mean of the visibilities in each bin

    Parameters
    ----------
    uv : array, unit = :math:`\lambda`
        Baselines of the data to bin
    V : array, unit = Jy
        Complex visibilities
    weights : array, unit = Jy^-2
        Weights on the visibility points
    bin_width : float, unit = :math:`\lambda`
        Width of the uv-bins
    """

    def __init__(self, uv, V, weights, bin_width):
        nbins = np.ceil(uv.max() / bin_width).astype('int')
        bins = np.arange(nbins+1, dtype='float64') * bin_width

        self._bins = bins
        self._nbins = nbins
        self._norm =  1 / bin_width

        bin_uv, bin_wgt, bin_vis, bin_n = \
            self.bin_quantities(uv, weights, 
                               uv, np.ones_like(uv), V, 
                               bin_counts=True)

        # Normalize
        idx = bin_n > 0
        w = bin_wgt[idx]
        bin_vis[idx] /= w
        bin_uv[idx] /= w

        # Compute the uncertainty on the means:
        w_sqd, w_sqd_V, w_sqd_V2 = \
            self.bin_quantities(uv, weights**2, 
                               np.ones_like(uv), V, (V.real**2 + 1j*V.imag**2))

        real_err = (w_sqd_V2.real - 2*bin_vis.real*w_sqd_V.imag 
                        + w_sqd*bin_vis.real**2)
        imag_err = (w_sqd_V2.imag - 2*bin_vis.imag*w_sqd_V.imag 
                        + w_sqd*bin_vis.imag**2)

        denom = bin_wgt**2 * (1 - 1 /np.maximum(bin_n, 2))
        real_err /= denom
        imag_err /= denom

        idx2 = bin_n > 2
        bin_vis_err = np.full(nbins, np.nan, dtype=V.dtype)
        bin_vis_err[idx2] = \
            np.sqrt(real_err[idx2]) + 1.j * np.sqrt(imag_err[idx2])
        
        # Use a sensible error for bins with one baseline
        idx1 = bin_n == 1
        bin_vis_err[idx1].real = bin_vis_err[idx1].imag = \
            1 / np.sqrt(bin_wgt[idx1])

        # Mask the empty bins
        self._uv = bin_uv[idx]
        self._V = bin_vis[idx]
        self._w = bin_wgt[idx]
        self._Verr = bin_vis_err[idx]
        self._count = bin_n[idx]

        self._uv_left = bins[:-1][idx]
        self._uv_right = bins[1:][idx]

    def bin_quantities(self, uv, w, *quantities, bin_counts=False):
        """Bin the given quantities according the to uv points and weights.

        Parameters
        ----------
        uv : array, unit = :math:`\lambda`
            Baselines of the data to bin
        weights : array, unit = Jy^-2
            Weights on the visibility points
        quantities : arrays,
            Quantities evaluated at the uv points to bin.
        bin_counts : bool, default=False
            Determines whether to count the number of uv points per bin.
        
        Returns
        -------
        results : arrays, same type as quantities
            Binned data
        bin_counts : array, int64, optional
            If bin_counts=True, then this array contains the number of uv 
            points in each bin. Otherwise, it is not returned.
        """
        bins = self._bins
        nbins = self._nbins
        norm = self._norm

        results = [np.zeros(nbins, dtype=x.dtype) for x in quantities]

        if bin_counts:
            counts = np.zeros(nbins, dtype='int64')

        BLOCK = 65536
        for i in range(0, len(uv), BLOCK):
            tmp_uv = uv[i:i+BLOCK]
            tmp_wgt = w[i:i+BLOCK]

            idx = np.floor(tmp_uv * norm).astype('int32')

            # Only the last bin includes its edge
            increment = (tmp_uv >= bins[idx+1]) & (idx+1 != nbins)
            idx[increment] += 1
            
            if bin_counts:
                counts += np.bincount(idx, minlength=nbins)

            for qty, res in zip(quantities, results):
                tmp_qty = tmp_wgt*qty[i:i+BLOCK]

                if np.iscomplexobj(qty):
                    res.real += np.bincount(idx, weights=tmp_qty.real,
                                            minlength=nbins)
                    res.imag += np.bincount(idx, weights=tmp_qty.imag,
                                            minlength=nbins)
                else:
                    res += np.bincount(idx, weights=tmp_qty, minlength=nbins)

        if bin_counts:
            return results + [counts]
        return results


    @property
    def uv(self):
        r"""Binned uv points, unit = :math:`\lambda`"""
        return self._uv

    @property
    def V(self):
        """Binned visibility, unit = Jy"""
        return self._V

    @property
    def weights(self):
        """Binned weights, unit = Jy^-2"""
        return self._w

    @property
    def error(self):
        """Uncertainty on the binned visibilities, unit = Jy"""
        return self._Verr

    @property
    def bin_counts(self):
        """Number of points in each bin"""
        return self._count

    @property
    def bin_edges(self):
        """Edges of the histogram bins"""
        return [self._uv_left, self._uv_right]


def normalize_uv(u, v, wle):
    r"""
    Normalize data u and v coordinates by the observing wavelength

    Parameters
    ----------
    u, v : array, unit = [m]
        u and v coordinates of observations
    wle : float, unit = [m]
        Observing wavelength of observations

    Returns
    -------
    u_normed, v_normed : array, unit = :math:`\lambda`
        u and v coordinates normalized by observing wavelength
    """

    u_normed = u / wle
    v_normed = v / wle

    return u_normed, v_normed


def cut_data_by_baseline(u, v, vis, weights, cut_range):
    r"""
    Truncate the data to be within a chosen baseline range

    Parameters
    ----------
    u, v : array, unit = [m]
        u and v coordinates of observations
    vis : array, unit = Jy
        Observed visibilities (complex: real + imag * 1j)
    weights : array, unit = Jy^-2
        Weights assigned to observed visibilities, of the form
        :math:`1 / \sigma^2`
    cut_range : list of float, length = 2, unit = [\lambda]
        Lower and upper baseline bounds outside of which visibilities are
        truncated

    Returns
    -------
    u_cut, v_cut : array, unit = :math:`\lambda`
        u and v coordinates in the chosen baseline range
    vis_cut : array, unit = Jy
        Visibilities in the chosen baseline range
    weights_cut : array, unit = Jy^-2
        Weights in the chosen baseline range
    """

    logging.info('  Cutting data outside of the minimum and maximum baselines'
                 ' of {} and {}'
                 ' klambda'.format(cut_range[0] / 1e3,
                                   cut_range[1] / 1e3))

    baselines = np.hypot(u, v)
    above_lo = baselines >= cut_range[0]
    below_hi = baselines <= cut_range[1]
    in_range = above_lo & below_hi
    u_cut, v_cut, vis_cut, weights_cut = [x[in_range] for x in [u, v, vis, weights]]

    return u_cut, v_cut, vis_cut, weights_cut


def apply_correction_to_weights(u, v, ReV, weights, nbins=300):
    r"""
    Estimate and apply a correction factor to the data's weights by comparing
    binnings of the real component of the visibilities under different
    weightings. This is useful for mock datasets in which the weights are all
    unity.

    Parameters
    ----------
    u, v : array, unit = :math:`\lambda`
        u and v coordinates of observations

    ReV : array, unit = Jy
        Real component of observed visibilities

    weights : array, unit = Jy^-2
        Weights assigned to observed visibilities, of the form
        :math:`1 / \sigma^2`

    nbins : int, default=300
        Number of bins used to construct the histograms

    Returns
    -------
    wcorr_estimate : float
        Correction factor by which to adjust the weights

    weights_corrected : array, unit = Jy^-2
        Corrected weights assigned to observed visibilities, of the form
        :math:`1 / \sigma^2`
    """

    logging.info('  Estimating, applying correction factor to visibility weights')

    baselines = np.hypot(u, v)
    mu, _ = np.histogram(np.log10(baselines), weights=ReV, bins=nbins)
    mu2, _ = np.histogram(np.log10(baselines), weights=ReV ** 2, bins=nbins)
    N, _ = np.histogram(np.log10(baselines), bins=nbins)

    mu /= np.maximum(N, 1)
    mu2 /= np.maximum(N, 1)

    sigma = (mu2 - mu ** 2) ** 0.5
    wcorr_estimate = sigma[np.where(sigma > 0)].mean()

    weights_corrected = weights / wcorr_estimate ** 2

    return wcorr_estimate, weights_corrected


def draw_bootstrap_sample(u, v, vis, weights):
    r"""
    Obtain the sample for a bootstrap, drawing, with replacement, N samples from
    a length N dataset

    Parameters
    ----------
    u, v : array, unit = :math:`\lambda`
          u and v coordinates of observations
    vis : array, unit = Jy
          Observed visibilities (complex: real + imag * 1j)
    weights : array, unit = Jy^-2
          Weights on the visibilities, of the form
          :math:`1 / \sigma^2`

    Returns
    -------
    u_boot, v_boot : array, unit = :math:`\lambda`
          Bootstrap sampled u and v coordinates
    vis_boot : array, unit = Jy
          Bootstrap sampled visibilities
    weights_boot : array, unit = Jy^-2
          Boostrap sampled weights on the visibilities
    """
    idxs = np.random.randint(low=0, high=len(u), size=len(u))

    u_boot = u[idxs]
    v_boot = v[idxs]
    vis_boot = vis[idxs]
    weights_boot = weights[idxs]

    return u_boot, v_boot, vis_boot, weights_boot


def sweep_profile(r, I, axis=0):
    r"""
    Sweep a 1D radial brightness profile over :math:`2 \pi` to yield a 2D
    brightness distribution

    Parameters
    ----------
    r : array
          Radial coordinates at which the 1D brightness profile is defined
    I : array
          Brightness values at r
    axis : int, default = 0
          Axis over which to interpolate the 1D profile

    Returns
    -------
    I2D : array, shape = (len(r), len(r))
        2D brightness distribution
    xmax : float
        Maximum x-value of the 2D grid
    ymax : float
        Maximum y-value of the 2D grid
    """

    xmax = ymax = r.max()
    dr = np.mean(np.diff(r))
    x = np.linspace(-xmax, xmax, int(xmax/dr) + 1)
    y = np.linspace(-ymax, ymax, int(ymax/dr) + 1)

    xi, yi = np.meshgrid(x, y)
    r1D = np.hypot(xi, yi)

    im_shape = r1D.shape + I.shape[1:]

    interp = interp1d(r, I, bounds_error=False, fill_value=0., axis=axis)
    I2D = interp(r1D.ravel()).reshape(*im_shape)

    return I2D, xmax, ymax
