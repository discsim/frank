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
"""This module contains methods for fitting a radial brightness profile to a set
  of deprojected visibities. Routines in this file assume that the emission is
  optically thin with a Gaussian vertical structure.
"""
import abc
from collections import defaultdict
import logging
import numpy as np

from frank.radial_fitters import FourierBesselFitter, FrankFitter

class FourierBesselDebrisFitter(FourierBesselFitter):
    """
    Fourier-Bessel series optically-thin model for fitting visibilities.

    The brightness model is :math:`I(R, z) = I(R) exp(-z^2/2H(R)^2)`, where
    :math:`H(R)` is the (known) scale-height.

    Parameters
    ----------
    Rmax : float, unit = arcsec
        Radius of support for the functions to transform, i.e.,
            f(r) = 0 for R >= Rmax
    N : int
        Number of collocation points
    geometry : SourceGeometry object
        Geometry used to deproject the visibilities before fitting
    scale_height : function R --> H
        Specifies the thickness of disc as a function of radius. Both
        units should be in arcsec.
    nu : int, default = 0
        Order of the discrete Hankel transform (DHT)
    block_data : bool, default = True
        Large temporary matrices are needed to set up the data. If block_data
        is True, we avoid this, limiting the memory requirement to block_size
        elements.
    block_size : int, default = 10**5
        Size of the matrices if blocking is used
    verbose : bool, default = False
        Whether to print notification messages
    """

    def __init__(self, Rmax, N, geometry, scale_height, nu=0, block_data=True,
                 block_size=10 ** 5, verbose=True):

        # All functionality is provided by the base class. 
        # FourierBesselDebrisFitter is just a sub-set of FourierBesselFitter
        super(FourierBesselDebrisFitter, self).__init__(
            Rmax, N, geometry, nu=nu, block_data=block_data,
            assume_optically_thick=False, scale_height=scale_height,
            block_size=block_size, verbose=verbose
        )

class FrankDebrisFitter(FrankFitter):
    """
    Fit a Gaussian process model using the Discrete Hankel Transform of
    Baddour & Chouinard (2015).

    The brightness model is :math:`I(R, z) = I(R) exp(-z^2/2H(R)^2)`, where
    :math:`H(R)` is the (known) scale-height.

    The GP model is based upon Oppermann et al. (2013), which use a maximum
    a posteriori estimate for the power spectrum as the GP prior for the
    real-space coefficients.

    Parameters
    ----------
    Rmax : float, unit = arcsec
        Radius of support for the functions to transform, i.e., f(r) = 0 for
        R >= Rmax.
    N : int
        Number of collaction points
    geometry : SourceGeometry object
        Geometry used to deproject the visibilities before fitting
    scale_height : function R --> H
        Specifies the thickness of disc as a function of radius. Both
        units should be in arcsec.
    nu : int, default = 0
        Order of the discrete Hankel transform, given by J_nu(r)
    block_data : bool, default = True
        Large temporary matrices are needed to set up the data. If block_data
        is True, we avoid this, limiting the memory requirement to block_size
        elements
    block_size : int, default = 10**5
        Size of the matrices if blocking is used
    alpha : float >= 1, default = 1.05
        Order parameter of the inverse gamma prior for the power spectrum
        coefficients
    p_0 : float >= 0, default = None, unit=Jy^2
        Scale parameter of the inverse gamma prior for the power spectrum
        coefficients. If not provided p_0 = 1e-15 (method="Normal") or 
        1e-35 (method="LogNormal") will be used.
    weights_smooth : float >= 0, default = 1e-4
        Spectral smoothness prior parameter. Zero is no smoothness prior
    tol : float > 0, default = 1e-3
        Tolerence for convergence of the power spectrum iteration
    method : string, default="Normal"
        Model used for the brightness reconstrution. This must be one of
        "Normal" of "LogNormal".
    I_scale : float, default = 1e5, unit= Jy/Sr
        Brightness scale. Only used in the LogNormal model. Notet the 
        LogNormal model produces I(Rmax) =  I_scale.
    max_iter: int, default = 2000
        Maximum number of fit iterations
    check_qbounds: bool, default = True
        Whether to check if the first (last) collocation point is smaller
        (larger) than the shortest (longest) deprojected baseline in the dataset
    store_iteration_diagnostics: bool, default = False
        Whether to store the power spectrum parameters and brightness profile
        for each fit iteration
    verbose:
        Whether to print notification messages

    References
    ----------
        Baddour & Chouinard (2015)
            DOI: https://doi.org/10.1364/JOSAA.32.000611
        Oppermann et al. (2013)
            DOI:  https://doi.org/10.1103/PhysRevE.87.032136
    """

    def __init__(self, Rmax, N, geometry, scale_height, nu=0, block_data=True,
                 block_size=10 ** 5, alpha=1.05, p_0=None, weights_smooth=1e-4,
                 tol=1e-3, method='Normal', I_scale=1e5, max_iter=2000, check_qbounds=True,
                 store_iteration_diagnostics=False, verbose=True):

        # All functionality is provided by the base class. FrankDebrisFitter is just a 
        # sub-set of FrankFitter
        super(FrankDebrisFitter, self).__init__(
            Rmax, N, geometry, nu=nu, block_data=block_data,
            block_size=block_size, alpha=alpha, p_0=p_0, weights_smooth=weights_smooth,
            tol=tol, method=method, I_scale=I_scale, max_iter=max_iter, 
            check_qbounds=check_qbounds, store_iteration_diagnostics=store_iteration_diagnostics, 
            assume_optically_thick=False, scale_height=scale_height, verbose=verbose)
