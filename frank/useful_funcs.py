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
import numpy as np
from scipy.interpolate import interp1d
__all__ = ['arcsec_baseline', 'BinUVData', 'sweep_profile']

def arcsec_baseline(x):
    """
    Provide x as a radial scale [arcsec] to return the corresponding baseline
    [lambda], or vice-versa.
    """
    converted = 1 / (x / 60 / 60 * np.pi / 180)
    return converted


class BinUVData(object):
    r"""Average the uv-data into bins of equal size.
    Computes the weighted mean of the visibilities in each bin.

    Parameters
    ----------
    uv : array, unit= :math:`\lambda`
        Baselines of the data to bin
    V : array, unit=Jy
        Complex visibilities
    weights : array, unit=Jy^-2
        Weights on the visibility points
    bin_width : float, unit= :math:`\lambda`
        Width of the uv-bins
    """
    def __init__(self, uv, V, weights, bin_width):
        nbins = np.ceil(uv.max() / bin_width).astype('int')
        bins = np.arange(nbins+1, dtype='float64') * bin_width

        # Initialize binned data
        bin_n   = np.zeros(nbins, dtype='int64')
        bin_uv  = np.zeros(nbins, dtype='float64')
        bin_wgt = np.zeros(nbins, dtype='float64')

        bin_vis = np.zeros(nbins, dtype='complex64')
        bin_vis_err = np.zeros(nbins, dtype='complex64')


        norm = 1 / bin_width

        # Use blocking since its faster and requires less memory
        BLOCK = 65536
        for i in range(0, len(uv), BLOCK):
            tmp_uv  = uv[i:i+BLOCK]
            tmp_vis = V[i:i+BLOCK]
            tmp_wgt = weights[i:i+BLOCK]

            idx = np.floor(tmp_uv * norm).astype('int32')

            # Only the last bin includes its edge
            increment = (tmp_uv >= bins[idx+1]) & (idx+1 != nbins)
            idx[increment] += 1

            bin_n   += np.bincount(idx, minlength=nbins)
            bin_wgt += np.bincount(idx, weights=tmp_wgt, minlength=nbins)
            bin_uv += np.bincount(idx, weights=tmp_wgt*tmp_uv, minlength=nbins)

            bin_vis.real += np.bincount(idx, weights=tmp_wgt*tmp_vis.real,
                                        minlength=nbins)
            bin_vis.imag  += np.bincount(idx, weights=tmp_wgt*tmp_vis.imag,
                                         minlength=nbins)

            bin_vis_err.real += np.bincount(idx,
                                            weights=tmp_wgt*tmp_vis.real**2,
                                            minlength=nbins)
            bin_vis_err.imag += np.bincount(idx,
                                            weights=tmp_wgt*tmp_vis.imag**2,
                                            minlength=nbins)

        # Normalize
        idx = bin_n > 0
        w = bin_wgt[idx]
        bin_vis [idx] /= w
        bin_uv[idx] /= w

        bin_vis_err[idx] /= w

        bin_vis_err.real = np.sqrt(bin_vis_err.real - bin_vis.real**2)
        bin_vis_err.imag = np.sqrt(bin_vis_err.imag - bin_vis.imag**2)

        # Use a sensible error for bins with one baseline
        idx1 = bin_n == 1
        bin_vis_err.real[idx1] = bin_vis_err.imag[idx1] = 1 / np.sqrt(bin_wgt[idx1])


        # Mask the empty bins
        self._uv = bin_uv[idx]
        self._V  = bin_vis[idx]
        self._w  = bin_wgt[idx]
        self._Verr = bin_vis_err[idx]
        self._count = bin_n[idx]

    @property
    def uv(self):
        r"""Binned uv points, unit= :math:`\lambda`"""
        return self._uv
    @property
    def V(self):
        """Binned visibility, unit=Jy"""
        return self._V
    @property
    def weights(self):
        """Binned weights, unit=Jy^-2"""
        return self._w
    @property
    def error(self):
        """Uncertainty on the binned visibilities, unit=Jy"""
        return self._Verr
    @property
    def bin_counts(self):
        """Number of points in each bin"""
        return self._count


def sweep_profile(r, I, axis=0):
    rmax = xmax = ymax = r.max()
    dr = np.mean(np.diff(r))
    x = np.linspace(-xmax, xmax, int(xmax/dr) + 1)
    y = np.linspace(-ymax, ymax, int(ymax/dr) + 1)

    xi, yi = np.meshgrid(x,y)
    r1D = np.hypot(xi,yi)

    im_shape = r1D.shape + I.shape[1:]

    interp = interp1d(r, I, bounds_error=False, fill_value=0., axis=axis)
    I2D = interp(r1D.ravel()).reshape(*im_shape)

    return I2D, xmax, ymax
