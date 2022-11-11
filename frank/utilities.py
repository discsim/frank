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
"""This module has various functions useful for plotting and analyzing fit
results.
"""
import logging
import numpy as np
from scipy.fft import fftfreq
from scipy.interpolate import interp1d

from frank.constants import deg_to_rad, sterad_to_arcsec, rad_to_arcsec
from frank.geometry import FixedGeometry
from frank.hankel import DiscreteHankelTransform
from frank.statistical_models import VisibilityMapping

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


def radius_convert(x, dist, conversion='arcsec_au'):
    """
    Provide x as a radius/radii in [arcsec] to convert to [au] (or vice-versa),
    assuming a distance in [pc]

    Parameters
    ----------
    x : float or array
        Radius or radii ([arcsec] or [au])
    dist : float
        Distance to source [pc]
    conversion : {'arcsec_au', 'au_arcsec'}, default = 'arcsec_au'
        The unit conversion to perform, e.g. 'arcsec_au' converts from [arcsec]
        to [au]

    Returns
    -------
    converted : float or array
        Radius or radii ([au] or [arcsec])

    """

    if conversion == 'arcsec_au':
        converted = x * dist
    elif conversion == 'au_arcsec':
        converted = x / dist
    else:
        raise AttributeError("conversion must be one of {}"
                             "".format(['arcsec_au', 'au_arcsec']))

    return converted


def jy_convert(x, conversion, bmaj=None, bmin=None):
    """
    Provide x as a brightness in one of the units [Jy / beam], [Jy / arcsec^2],
    [Jy / sterad] to convert x to another of these units

    Parameters
    ----------
    x : float
        Brightness in one of: [Jy / beam], [Jy / arcsec^2], [Jy / sterad]
    conversion : { 'beam_sterad', 'beam_arcsec2', 'arcsec2_beam',
                   'arcsec2_sterad', 'sterad_beam', 'sterad_arcsec2'}
        The unit conversion to perform, e.g., 'beam_sterad' converts
        [Jy / beam] to [Jy / sterad]
    bmaj : float, optional
        Beam FWHM along the major axis [arcsec]
    bmin : float, optional
        Beam FWHM along the minor axis [arcsec]

    Returns
    -------
    converted : float
        Brightness in converted units

    """
    if (bmaj is None or bmin is None) and conversion in ['beam_sterad',
                                                       'beam_arcsec2',
                                                       'arcsec2_beam',
                                                       'sterad_beam']:
        raise ValueError('bmaj and bmin must be specified to perform the'
        ' conversion {}'.format(conversion))

    if bmaj is not None and bmin is not None:
        beam_solid_angle = np.pi * bmaj  * bmin / (4 * np.log(2))

    if conversion == 'beam_arcsec2':
        converted = x / beam_solid_angle
    elif conversion == 'arcsec2_beam':
        converted = x * beam_solid_angle
    elif conversion == 'arcsec2_sterad':
        converted = x * sterad_to_arcsec
    elif conversion == 'sterad_arcsec2':
        converted = x / sterad_to_arcsec
    elif conversion == 'beam_sterad':
        converted = x / beam_solid_angle * sterad_to_arcsec
    elif conversion == 'sterad_beam':
        converted = x * beam_solid_angle / sterad_to_arcsec
    else:
        raise AttributeError("conversion must be one of {}"
                             "".format(['beam_sterad', 'beam_arcsec',
                             'arcsec_beam', 'arcsec_sterad',
                             'sterad_beam', 'sterad_arcsec']))

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
        Observed visibility. If complex, both the real and imaginary
        components will be binned. Else only the real part will be binned.
    weights : array, unit = Jy^-2
        Weights on the visibility points
    bin_width : float, unit = :math:`\lambda`
        Width of the uv-bins

    Notes
    -----
    Uses numpy masked arrays to mask bins with no uv points.

    """

    def __init__(self, uv, V, weights, bin_width):
        nbins = np.ceil(uv.max() / bin_width).astype('int')
        #Protect against rounding
        if nbins * bin_width < uv.max():
            nbins += 1
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
        bin_uv[idx] /= w
        bin_vis[idx] /= w

        # Store the binned data, masking empty bins:
        mask = (bin_n == 0)
        self._uv = np.ma.masked_where(mask, bin_uv)
        self._V = np.ma.masked_where(mask, bin_vis)
        self._w = np.ma.masked_where(mask, bin_wgt)
        self._count = np.ma.masked_where(mask, bin_n)

        self._uv_left = np.ma.masked_where(mask, bins[:-1])
        self._uv_right = np.ma.masked_where(mask, bins[1:])

        # Compute the uncertainty on the means:
        bin_vis_err = np.full(nbins, np.nan, dtype=V.dtype)

        #   1) Get the binned mean for each vis:
        mu = self._V[self.determine_uv_bin(uv)]

        #   2) Compute weighted error for bins with n > 1
        quantities = (V-mu).real**2
        if np.iscomplexobj(V):
            quantities = quantities + 1j * (V-mu).imag**2
        err = self.bin_quantities(uv, weights**2, quantities)

        idx2 = bin_n > 1
        err[idx2] /= bin_wgt[idx2]**2 * (1 - 1 / bin_n[idx2])

        temp_error = np.sqrt(err.real[idx2])
        if np.iscomplexobj(V):
            temp_error = temp_error + 1.j * np.sqrt(err.imag[idx2])
        bin_vis_err[idx2] = temp_error

        #   3) Use a sensible error for bins with one baseline
        idx1 = bin_n == 1
        bin_vis_err[idx1].real = 1 / np.sqrt(bin_wgt[idx1])
        if np.iscomplexobj(V):
            bin_vis_err[idx1].imag = bin_vis_err[idx1].real
        bin_vis_err[mask] = np.nan

        #   4) Store the error
        self._Verr = np.ma.masked_where(mask, bin_vis_err)


    def determine_uv_bin(self, uv):
        r"""Determine the bin that the given uv points belong too.

        Parameters
        ----------
        uv : array, unit = :math:`\lambda`
            Baselines to determine the bins of

        Returns
        -------
        idx : array,
            Bins that the uv point belongs to. Will be -1 if the bin does not
            exist.

        """

        bins = self._bins
        nbins = self._nbins

        idx = np.floor(uv * self._norm).astype('int32')
        idx[uv < bins[idx]] -= 1    # Fix rounding error
        idx[uv == bins[nbins]] -= 1 # Handle point exaclty on outer boundary

        too_high = idx >= nbins
        idx[too_high] = -1

        # Increase points on the RH boundary, unless it is the outer one
        idx_tmp = idx[~too_high]
        increment = (uv[~too_high] >= bins[idx_tmp+1]) & (idx_tmp+1 < nbins)
        idx_tmp[increment] += 1

        return idx

    def bin_quantities(self, uv, w, *quantities, bin_counts=False):
        r"""Bin the given quantities according the to uv points and weights.

        Parameters
        ----------
        uv : array, unit = :math:`\lambda`
            Baselines of the data to bin
        weights : array, unit = Jy^-2
            Weights on the visibility points
        quantities : arrays,
            Quantities evaluated at the uv points to bin.
        bin_counts : bool, default = False
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

            # Fix rounding:
            idx[tmp_uv < bins[idx]] -= 1
            # Move points exactly on the outer boundary inwards:
            idx[idx == nbins] -= 1

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
        if len(results) == 1:
            return results[0]
        return results

    def __len__(self):
        return len(self._uv)

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
    wle : float or array, unit = [m]
        Observing wavelength of observations. If an array, it should be the
        pointwise wavelength for each (u,v) point

    Returns
    -------
    u_normed, v_normed : array, unit = :math:`\lambda`
        u and v coordinates normalized by observing wavelength

    """

    logging.info('  Normalizing u and v coordinates by provided'
                 ' observing wavelength of {} m'.format(wle))

    wle = np.atleast_1d(wle, dtype='f8')
    if  len(wle) != 1 and len(wle) != len(u):
        raise ValueError("len(wle) = {}. It should be equal to "
                         "len(u) = {} (or 1 if all wavelengths are the same)".format(len(wle), len(u)))
    u_normed = u / wle
    v_normed = v / wle

    return u_normed, v_normed


def cut_data_by_baseline(u, v, vis, weights, cut_range, geometry=None):
    r"""
    Truncate the data to be within a chosen baseline range.

    The cut will be done in deprojected baseline space if the geometry is
    provided.

    Parameters
    ----------
    u, v : array, unit = :math:`\lambda`
        u and v coordinates of observations
    vis : array, unit = Jy
        Observed visibilities (complex: real + imag * 1j)
    weights : array, unit = Jy^-2
        Weights assigned to observed visibilities, of the form
        :math:`1 / \sigma^2`
    cut_range : list of float, length = 2, unit = [\lambda]
        Lower and upper baseline bounds outside of which visibilities are
        truncated
    geometry : SourceGeometry object, optional
        Fitted geometry (see frank.geometry.SourceGeometry).


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
    if geometry is not None:
        up, vp = geometry.deproject(u, v)
    else:
        up, vp = u, v

    baselines = np.hypot(up, vp)
    above_lo = baselines >= cut_range[0]
    below_hi = baselines <= cut_range[1]
    in_range = above_lo & below_hi
    u_cut, v_cut, vis_cut, weights_cut = [x[in_range] for x in [u, v, vis, weights]]

    return u_cut, v_cut, vis_cut, weights_cut


def estimate_weights(u, v=None, V=None, nbins=300, log=True, use_median=False,
                     verbose=True):
    r"""
    Estimate the weights using the variance of the binned visibilities.

    The estimation is done assuming that the variation in each bin is dominated
    by the noise. This will be true if:
    1) The source is axi-symmetric,
    2) The uv-points have been deprojected,
    3) The bins are not too wide,
    Otherwise the variance may be dominated by intrinsic variations in the
    visibilities.

    Parameters
    ----------
    u, v : array, unit = :math:`\lambda`
        u and v coordinates of observations (deprojected). Data will be binned
        by baseline. If v is not None, np.hpot(u,v) will be used instead. Note
        that if V is None the argument v will be intepreted as V instead
    V : array, unit = Jy, default = None
        Observed visibility. If complex, the weights will be computed from the
        average of the variance of the real and imaginary components, as in
        CASA's statwt. Otherwise the variance of the real part is used.
    nbins : int, default = 300
        Number of bins used.
    log : bool, default = True
        If True, the uv bins will be constructed in log space, otherwise linear
        spaced bins will be used.
    use_median : bool, default = False
        If True all of the weights will be set to the median of the variance
        estimated across the bins. Otherwise, the baseline dependent variance
        will be used.
    verbose : bool, default = True
        If true, the logger will record calls to this function, along with
        whether the median estimate was used.

    Returns
    -------
    weights : array
        Estimate of the weight for each uv point.

    Notes
    -----
        - This function does not use the original weights in the estimation.
        - Bins with only one uv point do not have a variance estimate. Thus
          the mean of the variance in the two adjacent bins is used instead.

    Examples
    --------
        All of the following calls will work as expected:
            `estimate_weights(u, v, V) `
            `estimate_weights(u, V)`
            `estimate_weights(u, V=V)`
        In each case the variance of V in the uv-bins is used to estimate the
        weights. The first call will use q = np.hypot(u, v) in the uv-bins. The
        second and third calls are equivalent to the first with u=0.

    """

    if verbose:
        logging.info('  Estimating visibility weights')

    if V is None:
        if v is not None:
            V = v
            q = np.abs(u)
        else:
            raise ValueError("The visibilities, V, must be supplied")
    elif v is not None:
        q = np.hypot(u,v)
    else:
        q = np.abs(u)

    if log:
        q = np.log(q)
        q -= q.min()

    bin_width = (q.max()-q.min()) / nbins

    uvBin = UVDataBinner(q, V, np.ones_like(q), bin_width)

    if uvBin.bin_counts.max() == 1:
        raise ValueError("No bin contains more than one uv point, can't"
                         " estimate the variance. Use fewer bins.")

    if np.iscomplex(V.dtype):
        var = 0.5*(uvBin.error.real**2 + uvBin.error.imag**2) * uvBin.bin_counts
    else:
        var = uvBin.error.real**2 * uvBin.bin_counts

    if use_median:
        if verbose:
            logging.info('    Setting all weights as median binned visibility '
                         'variance')
        return np.full(len(u), 1/np.ma.median(var[uvBin.bin_counts > 1]))
    else:
        if verbose:
            logging.info('    Setting weights according to baseline-dependent '
                         'binned visibility variance')
        # For bins with 1 uv point, use the average of the adjacent bins
        no_var = np.argwhere(uvBin.bin_counts == 1).reshape(-1)
        if len(no_var) > 0:
            # Find the location `loc` of the bad points in the array of good points
            good_var = np.argwhere(uvBin.bin_counts > 1).reshape(-1)
            loc = np.searchsorted(good_var, no_var, side='right')

            # Set the variance to the average of the two adjacent bins
            im = good_var[np.maximum(loc-1, 0)]
            ip = good_var[np.minimum(loc, len(good_var)-1)]
            var[no_var] = 0.5*(var[im] + var[ip])

        bin_id = uvBin.determine_uv_bin(q)
        assert np.all(bin_id != -1), "Error in binning"  # Should never occur

        weights = 1/var[bin_id]

        return weights


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


def sweep_profile(r, I, project=False, phase_shift=False, geom=None, axis=0,
                  xmax=None, ymax=None, dr=None):
    r"""
    Sweep a 1D radial brightness profile over :math:`2 \pi` to yield a 2D
    brightness distribution. Optionally project this sweep by a supplied
    geometry.

    Parameters
    ----------
    r : array
        Radial coordinates at which the 1D brightness profile is defined
    I : array
        Brightness values at r
    project : bool, default = False
        Whether to project the swept profile by the supplied geom
    phase_shift : bool, default = False
        Whether to phase shift the projected profile by the supplied geom.
        If False, the source will be centered in the image
    geom : SourceGeometry object, default=None
        Fitted geometry (see frank.geometry.SourceGeometry). Here we use
        geom.inc [deg], geom.PA [deg], geom.dRA [arcsec], geom.dDec [arcsec] if
        project=True
    axis : int, default = 0
        Axis over which to interpolate the 1D profile
    xmax, ymax : float, optional, default = None
        Value setting the x- and y-bounds of the image (same units as r). The
        positive and negative bounds are both set to this value (modulo sign).
        If not provided, these will be set to r.max()
    dr : float, optional, default = None
        Pixel size (same units as r). If not provided, it will be set at the
        same spatial scale as r

    Returns
    -------
    I2D : array, shape = (len(r), len(r))
        2D brightness distribution (projected if project=True)
    xmax : float
        Maximum x-value of the 2D grid
    ymax : float
        Maximum y-value of the 2D grid

    Notes
    -----
    Sign convention: a negative geom.dRA shifts the source to the right
    in the image

    """
    if project:
        inc, pa, dra, ddec = geom.inc, geom.PA, geom.dRA, geom.dDec
        inc *= deg_to_rad
        pa *= deg_to_rad
        if not phase_shift:
            dra, ddec = 0., 0.

        cos_i = np.cos(inc)
        cos_pa, sin_pa = np.cos(pa), np.sin(pa)

    if xmax is None:
        xmax = r.max()
    if ymax is None:
        ymax = r.max()

    if dr is None:
        dr = np.mean(np.diff(r))

    x = np.linspace(-xmax, xmax, int(xmax/dr) + 1)
    y = np.linspace(-ymax, ymax, int(ymax/dr) + 1)

    if phase_shift:
        xi, yi = np.meshgrid(x + dra, y - ddec)
    else:
        xi, yi = np.meshgrid(x, y)

    if project:
        xp  = xi * cos_pa + yi * sin_pa
        yp  = -xi * sin_pa + yi * cos_pa
        xp /= cos_i
        r1D = np.hypot(xp, yp)
    else:
        r1D = np.hypot(xi, yi)

    im_shape = r1D.shape + I.shape[1:]

    interp = interp1d(r, I, bounds_error=False, fill_value=0., axis=axis)
    I2D = interp(r1D.ravel()).reshape(*im_shape)

    return I2D, xmax, ymax


def make_image(fit, Npix, xmax=None, ymax=None, project=True):
    """Make an image of a model fit.
    
    Parameters
    ----------
    fit : FrankFitter result object
        Fitted profile to make an image of
    Npix : int or list
        Number of pixels in the x-direction, or [x-,y-] direction
    xmax: float or None, unit=arcsec
        Size of the image is [-xmax, xmax]. By default this is twice
        fit.Rmax to avoid aliasing.
    ymax: float or None, unit=arcsec
        Size of the image is [-ymax,ymax]. Defaults to xmax if ymax=None
    project: bool, default=True
        Whether to produce a projected image.

    Returns
    -------
    x : array, 1D; unit=arcsec
        Locations of the x-points in the image.
    y : array, 1D; unit=arcsec
        Locations of the y-points in the image.
    I : array, 2D; unit=Jy/Sr
        Image of the surface brightness.
    """
    if xmax is None:
        xmax = 2*fit.Rmax
    if ymax is None:
        ymax = xmax
    
    try:
        Nx, Ny = Npix
    except TypeError:
        Nx = Npix
        Ny = int(Nx*(ymax/xmax))

    dx = 2*xmax/(Nx*rad_to_arcsec)
    dy = 2*ymax/(Ny*rad_to_arcsec)

    
    # The easiest way to produce an image is to predict the visibilities
    # on a regular grid in the Fourier plane and then transform it back.
    # All frank models must be able to compute the visibilities so this
    # method is completely general.
    u = np.fft.fftfreq(Nx)/dx
    v = np.fft.fftfreq(Ny)/dy
    u, v = np.meshgrid(u,v, indexing='ij')
    
    # Get the visibilities:
    if project:
        Vis = fit.predict(u.reshape(-1), v.reshape(-1)).reshape(*u.shape)
    else:
        q = np.hypot(u,v)
        Vis = fit.predict_deprojected(q.reshape(-1)).reshape(*u.shape)
        
    # Convert to the image plane
    I = np.fft.ifft(np.fft.ifft(Vis, axis=0), axis=1).real
    I /= dx*dy

    # numpy's fft has zero in the corner. We want it in the middle so we need
    # to wrap:
    tmp = I.copy()
    tmp[:Nx//2,], tmp[Nx//2:] = I[Nx//2:], I[:Nx//2]
    I[:,:Ny//2], I[:,Ny//2:] = tmp[:,Ny//2:], tmp[:,:Ny//2]

    xe = np.linspace(-xmax, xmax, Nx+1)
    x = 0.5*(xe[1:] + xe[:-1])
    ye = np.linspace(-ymax, ymax, Ny+1)
    y = 0.5*(ye[1:] + ye[:-1])

    return x, y, I


def convolve_profile(r, I, disc_i, disc_pa, clean_beam,
                    n_per_sigma=5, axis=0):
    r"""
    Convolve a 1D radial brightness profile with a 2D Gaussian beam, degrading
    the profile's resolution

    Parameters
    ----------
    r : array
        Radial coordinates at which the 1D brightness profile is defined
    I : array
        Brightness values at r
    disc_i : float, unit = deg
        Disc inclination
    disc_pa : float, unit = deg
        Disc position angle
    clean_beam : dict
        Dictionary with beam `bmaj` (FWHM of beam along its major axis) [arcsec],
        `bmin` (FWHM of beam along its minor axis) [arcsec],
        `pa` (beam position angle) [deg]
    n_per_sigma : int, default = 50
        Number of points per standard deviation of the Gaussian kernel (used
        for gridding)
    axis : int, default = 0
          Axis over which to interpolate the 1D profile

    Returns
    -------
    I_smooth : array, shape = (len(r), len(r))
        Convolved brightness profile I at coordinates r

    """

    from scipy.constants import c
    from scipy.interpolate import interp1d
    from scipy.ndimage import gaussian_filter

    # Set up the geometry for the smoothing mesh.
    # We align the beam with the grid (major axis aligned) and rotate the
    #  image accordingly

    # Convert beam FWHM to sigma
    bmaj = clean_beam['bmaj'] / np.sqrt(8 * np.log(2))
    bmin = clean_beam['bmin'] / np.sqrt(8 * np.log(2))

    PA = (disc_pa - clean_beam['beam_pa']) * np.pi / 180.

    cos_i = np.cos(disc_i * np.pi/180.)
    cos_PA = np.cos(PA)
    sin_PA = np.sin(PA)

    # Pixel size in terms of bmin
    rmax = r.max()
    dx = bmin / n_per_sigma

    xmax = rmax * (np.abs(cos_i * cos_PA) + np.abs(sin_PA))
    ymax = rmax * (np.abs(cos_i * sin_PA) + np.abs(cos_PA))

    xmax = int(xmax / dx + 1) * dx
    ymax = int(ymax / dx + 1) * dx

    x = np.arange(-xmax, xmax + dx / 2, dx)
    y = np.arange(-ymax, ymax + dx / 2, dx)

    xi, yi = np.meshgrid(x, y)

    xp =  xi * cos_PA + yi * sin_PA
    yp = -xi * sin_PA + yi * cos_PA
    xp /= cos_i

    r1D = np.hypot(xi, yi)

    im_shape = r1D.shape + I.shape[1:]

    # Interpolate to grid and apply smoothing
    interp = interp1d(r, I, bounds_error=False, fill_value=0., axis=axis)

    I2D = interp(r1D.ravel()).reshape(*im_shape)
    sigma = [float(n_per_sigma), (bmaj / bmin) * n_per_sigma]
    I2D = gaussian_filter(I2D, sigma, mode='nearest', cval=0.)

    # Now convert back to a 1D profile
    edges = np.concatenate(([r[0] * r[0] / r[1]], r, [r[-1] * r[-1] / r[-2]]))
    edges = 0.5 * (edges[:-1] + edges[1:])

    I_smooth = np.empty_like(I)
    I_smooth = np.histogram(r1D.ravel(), weights=I2D.ravel(), bins=edges)[0]
    counts = np.maximum(np.histogram(r1D.ravel(), bins=edges)[0], 1)
    I_smooth /= counts

    return I_smooth


def add_vis_noise(vis, weights, seed=None):
    r"""
    Add Gaussian noise to visibilities

    Parameters
    ----------
    vis : array, unit = [Jy]
        Visibilities to add noise to.
        Can be complex (real + imag * 1j) or purely real.
    weights : array, unit = [Jy^-2]
        Weights on the visibilities, of the form :math:`1 / \sigma^2`.
        Injected noise will be scaled proportional to `\sigma`.
    seed : int, default = None
        Number to initialize a pseudorandom number generator for the noise draws

    Returns
    -------
    vis_noisy : array, shape = vis.shape
        Visibilities with added noise

    """
    if seed is not None:
        np.random.seed(seed)

    dim0 = 1
    if np.iscomplexobj(vis):
        dim0 = 2

    vis = np.array(vis)
    noise = np.random.standard_normal((dim0,) + vis.shape)
    noise *= weights ** -0.5

    vis_noisy = vis + noise[0]
    if np.iscomplexobj(vis):
        vis_noisy += 1j * noise[1]

    return vis_noisy


def make_mock_data(r, I, Rmax, u, v, geometry=None, N=500, add_noise=False,
                   weights=None, seed=None):
    r"""
    Generate mock visibilities from a provided brightness profile and (u,v)
    distribution.

    Parameters
    ----------
    r : array, unit = [arcsec]
        Radial coordinates of I(r)
    I : array, unit = [Jy / sr]
        Brightness values at r
    Rmax : float, unit = [arcsec], default=2.0
        Maximum radius beyond which I(r) is zero. This should be larger than the
        disk size
    u, v : array, unit = :math:`\lambda`
        u and v coordinates of observations
    geometry : SourceGeometry object, default=None
        Source geometry (see frank.geometry.SourceGeometry). If supplied, the
        visibilities will be deprojected, and their total flux scaled by the
        inclination.
    N : integer, default=500
        Number of terms to use in the Fourier-Bessel series
    add_noise : bool, default = False
        Whether to add noise to the mock visibilities
    weights : array, unit = Jy^-2
        Visibility weights, of the form :math:`1 / \sigma^2`.
        If provided, injected noise will be scaled proportional to `\sigma`.
    seed : int, default = None
        Number to initialize a pseudorandom number generator for the noise draws

    Returns
    -------
    baselines : array, unit = :math:`\lambda`
        Baseline coordinates of the mock visibilities. These will be equal to
        np.hypot(u, v) if 'geometry' is None (or if its keys are all equal to 0)
    vis : array, unit = Jy
        Mock visibility amplitudes, including noise if 'add_noise' is True

    Notes
    -----
    'r' and 'I' should be sufficiently sampled to ensure an interpolation will
    be accurate, otherwise 'vis' may be a poor estimate of their transform.
    """

    if geometry is None:
        geometry = FixedGeometry(0, 0, 0, 0)
    else:
        u, v = geometry.deproject(u, v)

    baselines = np.hypot(u, v)

    _, vis = generic_dht(r, I, Rmax, N, grid=baselines, inc=geometry.inc)

    if add_noise:
        vis = add_vis_noise(vis, weights, seed)

    return baselines, vis


def get_collocation_points(Rmax=2.0, N=500, direction='forward'):
    """
    Obtain the collocation points of a discrete Hankel transform for a given
    'Rmax' and 'N' (see frank.hankel.DiscreteHankelTransform)

    Parameters
    ----------
    Rmax : float, unit = [arcsec], default=2.0
        Maximum radius beyond which the real space function is zero
    N : integer, default=500
        Number of terms to use in the Fourier-Bessel series
    direction : { 'forward', 'backward' }, default='forward'
        Direction of the transform. 'forward' is real space -> Fourier space,
        returning real space radial collocation points needed for the transform.

    Returns
    -------
    coll_pts : array, unit = [lambda] or [arcsec]
        The DHT collocation points in either real or Fourier space.

    """
    if direction not in ['forward', 'backward']:
        raise AttributeError("direction must be one of ['forward', 'backward']")

    r_pts, q_pts = DiscreteHankelTransform.get_collocation_points(
        Rmax=Rmax / rad_to_arcsec, N=N, nu=0
        )

    if direction == 'forward':
        coll_pts = r_pts * rad_to_arcsec
    else:
        coll_pts = q_pts

    return coll_pts


def generic_dht(x, f, Rmax=2.0, N=500, direction='forward', grid=None,
                inc=0.0):
    """
    Compute the visibilities or brightness of a model by directly applying the
    Discrete Hankel Transform. 
    
    The correction for inclination will also be applied, assuming an optically
    thick disc. For an optically thin disc, setting inc=0 (the default) will
    achieve the correct scaling.

    Parameters
    ----------
    x : array, unit = [arcsec] or [lambda]
        Radial or spatial frequency coordinates of f(x)
    f : array, unit = [Jy / sr] or [Jy]
        Amplitude values of f(x)
    Rmax : float, unit = [arcsec], default=2.0
        Maximum radius beyond which the real space function is zero
    N : integer, default=500
        Number of terms to use in the Fourier-Bessel series
    direction : { 'forward', 'backward' }, default='forward'
        Direction of the transform. 'forward' is real space -> Fourier space.
    grid : array, unit = [arcsec] or [lambda], default=None
        The radial or spatial frequency points at which to sample the DHT.
        If None, the DHT collocation points will be used.
    inc : float, unit = [deg], default = 0.0
        Source inclination. The total flux of the transform of f(x)
        will be scaled by cos(inc); this has no effect if inc=0.

    Returns
    -------
    grid : array, size=N, unit = [arcsec] or [lambda]
        Spatial frequency or radial coordinates of the Hankel transform of f(x)
    f_transform : array, size=N, unit = [Jy / sr] or [Jy]
        Hankel transform of f(x)

    Notes
    -----
    'x' and 'f' should be sufficiently sampled to ensure an interpolation will
    be accurate, otherwise 'f_transform' may be a poor estimate of their
    transform.
    """

    if direction not in ['forward', 'backward']:
        raise AttributeError("direction must be one of ['forward', 'backward']")

    DHT = DiscreteHankelTransform(Rmax=Rmax / rad_to_arcsec, N=N, nu=0)
    geom = FixedGeometry(inc, 0, 0, 0)
    VM = VisibilityMapping(DHT, geom)

    if direction == 'forward':
        # map the profile f(x) onto the DHT collocation points.
        # this is necessary for an accurate transform.
        y = np.interp(VM.r, x, f)

        if grid is None:
            grid = VM.q

        # perform the DHT
        f_transform = VM.predict_visibilities(y, grid, geometry=geom)

    else:
        y = np.interp(VM.q, x, f)

        if grid is None:
            grid = VM.r

        f_transform = VM.invert_visibilities(y, grid, geometry=geom)

    return grid, f_transform
