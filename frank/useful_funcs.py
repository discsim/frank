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


#from galario import arcsec, deg, au
#from galario.double import sweep
#arcsec
arcsec = 4.84813681109536e-06
deg = 0.017453292519943295
au = 14959787070000.0

def sweep_ref(I, Rmin, dR, nrow, ncol, dxy, inc, pa, Dx=0., Dy=0., dtype_image='float64', origin='upper'):
    """
    Compute the intensity map (i.e. the image) given the radial profile I(R).
    We assume an axisymmetric profile.
    The origin of the output image is in the upper left corner.
    Parameters
    ----------
    I: 1D float array
        Intensity radial profile I(R).
    Rmin : float
        Inner edge of the radial grid. At R=Rmin the intensity is intensity[0].
        For R<Rmin the intensity is assumed to be 0.
        **units**: rad
    dR : float
        Size of the cell of the radial grid, assumed linear.
        **units**: rad
    nrow : int
        Number of rows of the output image.
        **units**: pixel
    ncol : int
        Number of columns of the output image.
        **units**: pixel
    dxy : float
        Size of the image cell, assumed equal and uniform in both x and y direction.
        **units**: rad
    inc : float
        Inclination along North-South axis.
        **units**: rad
    Dx : optional, float
        Right Ascension offset (positive towards East, left).
        **units**: rad
    Dy : optional, float
        Declination offset (positive towards North, top).
        **units**: rad
    dtype_image : optional, str
        numpy dtype specification for the output image.
    origin: ['upper' | 'lower'], optional, default: 'upper'
        Set the [0,0] index of the array in the upper left or lower left corner of the axes.
    Returns
    -------
    intensmap: 2D float array
        The intensity map, sweeped by 2pi.
    """
    if origin == 'upper':
        v_origin = 1.
    elif origin == 'lower':
        v_origin = -1.

    inc_cos = np.cos(inc)

    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)

    nrad = len(I)
    gridrad = np.linspace(Rmin, Rmin + dR * (nrad - 1), nrad)

    # create the mesh grid
    x = (np.linspace(0.5, -0.5 + 1./float(ncol), ncol)) * dxy * ncol
    y = (np.linspace(0.5, -0.5 + 1./float(nrow), nrow)) * dxy * nrow * v_origin

    # we shrink the x axis, since PA is the angle East of North of the
    # the plane of the disk (orthogonal to the angular momentum axis)
    # PA=0 is a disk with vertical orbital node (aligned along North-South)

    ##rotate x,y

    xxx, yyy = np.meshgrid((x - Dx) / inc_cos, (y - Dy))
    #xxx, yyy = np.meshgrid(((x - Dx) * cos_pa - (y - Dy) * sin_pa) * inc_cos, ((x - Dx) * sin_pa + (y - Dy) * cos_pa))

    x_meshgrid = np.sqrt(xxx ** 2. + yyy ** 2.)

    f = interp1d(gridrad, I, kind='linear', fill_value=0.,
                 bounds_error=False, assume_sorted=True)
    intensmap = f(x_meshgrid)

    # central pixel: compute the average brightness
    intensmap[int(nrow / 2 + Dy / dxy * v_origin), int(ncol / 2 - Dx / dxy)] = central_pixel(I, Rmin, dR, dxy)

    # convert to Jansky
    intensmap *= dxy**2.

    return intensmap.astype(dtype_image)

def show_image(image, ax, nwidth=None, **kwargs):
    """

    Parameters
    ----------
    image: ndarray, float
        2D image
    nwidth: int, optional
        Portion of the image to show: will crop the image to a size 2*nwidth x 2*nwidth.
        Units: number of pixels
    """
    nx, ny = image.shape

    if 'cmap' not in kwargs.keys():
        kwargs['cmap'] = 'plasma'
    if 'norm' not in kwargs.keys():
        import matplotlib.colors as colors
        kwargs['norm'] = colors.PowerNorm(gamma=0.4)


    if not nwidth:
        # by default, show the whole image
        nwidth = min(nx // 2, ny // 2)
    else:
        nwidth = int(nwidth)
        if nwidth > nx//2 or nwidth > ny//2:
            raise ValueError("Expect nwidth to be smaller than half the number of pixels on each direction, "
                             "got {} for image shape {}".format(nwidth, image.shape))

    ax.matshow(image[(nx//2-nwidth):(nx//2+nwidth), (nx//2-nwidth):(nx//2+nwidth)], **kwargs)


def create_image(f, nxy, dxy, inc=0., pa=0., Dx=0, Dy=0, Rmin=1e-6, dR=0.0001*4.84813681109536e-06, nR=1.7*1e4):
    """
    f:
    nxy: int
        Number of pixels on each dimension (assumes square image);
    dxy: float
        Pixel size. Units: arcsec
    Rmin: float, optional
        Innermost radius of the radial grid. Units: arcsec
    dR: float, optional
        Radial grid cell size. Units: arcsec
    inc: float, optional
        Inclination. Units: deg
    nR: int, optional
        Number of cells in the radial grid.

    Returns
    -------
    image: ndarray, float
        2D image produced by sweeping the 1D profile

    """
    dxy *= arcsec
    inc *= deg
    Rmin *= arcsec
    dR *= arcsec
    nR = int(nR)

    if callable(f):
        # radial grid
        gridrad = np.linspace(Rmin, Rmin + dR * (nR - 1), nR)
        f_arr = f(gridrad)
    else:
        f_arr = f

    ##image = sweep(f_arr, Rmin, dR, nxy, dxy, inc)
    image = sweep_ref(I=f_arr, Rmin=Rmin, dR=dR, nrow=nxy, ncol=nxy, dxy=dxy, inc=inc, pa=pa, Dx=Dx, Dy=Dy)

    print("Emission peak is {:e} mJy/sr at ({})".format(np.max(image)/(dxy**2)*1e3, np.unravel_index(np.argmax(image), shape=image.shape)))
    print("Total flux: {} mJy".format(np.nansum(image)*1e3))

    return image

def central_pixel(I, Rmin, dR, dxy):
    """
    Compute brightness in the central pixel as the average flux in the pixel.
    """
    # with quadrature method: tends to over-estimate it
    # area = np.pi*((dxy/2.)**2-Rmin**2)
    # flux, _ = quadrature(lambda z: f(z)*z, Rmin, dxy/2., tol=1.49e-25, maxiter=200)
    # flux *= 2.*np.pi
    # intensmap[int(nrow/2+Dy/dxy), int(ncol/2-Dx/dxy)] = flux/area

    # with trapezoidal rule: it's the same implementation as in galario.cpp
    iIN = int(np.floor((dxy / 2 - Rmin) // dR))
    print('iIN',iIN)
    flux = 0.
    for i in range(1, iIN - 1): # TODO: remove - 1
        flux += (Rmin + dR * i) * I[i]

    flux *= 2.
    flux += Rmin * I[0] + (Rmin + iIN * dR) * I[iIN]
    flux *= dR

    # add flux between Rmin+iIN*dR and dxy/2
    I_interp = (I[iIN + 1] - I[iIN]) / (dR) * (dxy / 2. - (Rmin + dR * (iIN))) + \
               I[iIN]  # brightness at R=dxy/2
    flux += ((Rmin + iIN * dR) * I[iIN] + dxy / 2. * I_interp) * (
                dxy / 2. - (Rmin + iIN * dR))

    # flux *= 2 * np.pi / 2.  # to complete trapezoidal rule (***)
    area = (dxy / 2.) ** 2 - Rmin ** 2
    # area *= np.pi  # elides (***)

    return flux / area
