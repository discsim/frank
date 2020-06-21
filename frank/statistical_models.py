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


import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.optimize
from collections import defaultdict
import logging

from frank.hankel import DiscreteHankelTransform
from frank.constants import rad_to_arcsec, deg_to_rad

from frank.minimizer import LineSearch, MinimizeNewton


class GaussianModel:
    r"""
    Solves the linear regression problem to compute the posterior,

    .. math::
       P(I|q,V,p) \propto G(I-\mu, D),

    where :math:`I` is the intensity to be predicted, :math:`q` are the
    baselines and :math:`V` the visibility data. :math:`\mu` and :math:`D` are
    the mean and covariance of the posterior distribution.

    If :math:`p` is provided, the covariance matrix of the prior is included,
    with

    .. math::
        P(I|p) \propto G(I, S(p)),

    and the Bayesian Linear Regression problem is solved. :math:`S` is computed
    from the power spectrum, :math:`p`, if provided. Otherwise the traditional
    (frequentist) linear regression is used.

    The problem is framed in terms of the design matrix :math:`M` and
    information source :math:`j`.

    :math:`H(q)` is the matrix that projects the intensity :math:`I` to
    visibility space. :math:`M` is defined by

    .. math::
        M = H(q)^T w H(q),

    where :math:`w` is the weights matrix and

    .. math::
        j = H(q)^T w V.

    The mean and covariance of the posterior are then given by

    .. math::
        \mu = D j

    and

    .. math::
        D = [ M + S(p)^{-1}]^{-1},

    if the prior is provided, otherwise

    .. math::
        D = M^{-1}.


    Parameters
    ----------
    DHT : DiscreteHankelTransform
        A DHT object with N bins that defines H(p). The DHT is used to compute
        :math:`S(p)`
    M : 2D array, size = (N, N)
        The design matrix, see above
    j : 1D array, size = N
        Information source, see above
    p : 1D array, size = N, optional
        Power spectrum used to generate the covarience matrix :math:`S(p)`
    geometry: SourceGeometry object, optional
        If provided, this geometry will be used to deproject the visibilities
        in self.predict
    noise_likelihood : float, optional
        An optional parameter needed to compute the full likelihood, which
        should be equal to

        .. math::
            -\frac{1}{2} V^T w V + \frac{1}{2} \sum \log[w/(2 \pi)].

        If not  provided, the likelihood can still be computed up to this
        missing constant
    """

    def __init__(self, DHT, M, j, p=None, guess=None, noise_likelihood=0):

        self._DHT = DHT
        self._M = M
        self._j = j

        self._p = p
        if p is not None:
            if np.any(p <= 0) or np.any(np.isnan(p)):
                raise ValueError("Bad value in power spectrum. The power"
                                 " spectrum must be postive and not contain"
                                 " any NaN values")

            Ykm = self._DHT.coefficients()
            self._Sinv = np.einsum('ji,j,jk->ik', Ykm, 1/p, Ykm)
        else:
            self._Sinv = None

        self._like_noise = noise_likelihood

        self._fit()

    def _fit(self):
        """Compute the mean and variance"""
        # Compute the inverse prior covariance, S(p)^-1
        Sinv = self._Sinv
        if Sinv is None:
            Sinv = 0

        Dinv = self._M + Sinv

        try:
            self._Dchol = scipy.linalg.cho_factor(Dinv)
            self._Dsvd = None

            self._mu = scipy.linalg.cho_solve(self._Dchol, self._j)

        except np.linalg.LinAlgError:
            U, s, V = scipy.linalg.svd(Dinv, full_matrices=False)

            s1 = np.where(s > 0, 1. / s, 0)

            self._Dchol = None
            self._Dsvd = U, s1, V

            self._mu = np.dot(V.T, np.multiply(np.dot(U.T, self._j), s1))

        # Reset the covariance matrix - we will compute it when needed
        self._cov = None

    def Dsolve(self, b):
        r"""
        Compute :math:`D \cdot b` by solving :math:`D^{-1} x = b`.

        Parameters
        ----------
        b : array, size = (N,...)
            Right-hand side to solve for

        Returns
        -------
        x : array, shape = np.shape(b)
            Solution to the equation D x = b

        """
        if self._Dchol is not None:
            return scipy.linalg.cho_solve(self._Dchol, b)
        else:
            U, s1, V = self._Dsvd
            return np.dot(V.T, np.multiply(np.dot(U.T, b), s1))

    def draw(self, N):
        """Compute N draws from the posterior"""
        return np.random.multivariate_normal(self.mean, self.covariance, N)

    def log_likelihood(self, I=None):
        r"""
        Compute one of two types of likelihood.

        If :math:`I` is provided, this computes

        .. math:
            \log[P(I,V|S)].

        Otherwise the marginalized likelihood is computed,

        .. math:
            \log[P(V|S)].


        Parameters
        ----------
        I : array, size = N, optional, unit = Jy / sr
            Intensity :math:`I(r)` to compute the likelihood of

        Returns
        -------
        log_P : float
            Log likelihood, :math:`\log[P(I,V|p)]` or :math:`\log[P(V|p)]`

        Notes
        -----
        1. The prior probability P(S) is not included.
        2. The likelihoods take the form:

        .. math::
              \log[P(I,V|p)] = \frac{1}{2} j^T I - \frac{1}{2} I^T D^{-1} I
                 - \frac{1}{2} \log[\det(2 \pi S)] + H_0

        and

        .. math::
              \log[P(V|p)] = \frac{1}{2} j^T D j
                 + \frac{1}{2} \log[\det(D)/\det(S)] + H_0

        where

        .. math::
            H_0 = -\frac{1}{2} V^T w V + \frac{1}{2} \sum \log(w /2 \pi)

        is the noise likelihood.
        """

        if I is None:
            like = 0.5 * np.sum(self._j * self._mu)

            if self._Sinv is not None:
                Q = self.Dsolve(self._Sinv)
                like += 0.5 * np.linalg.slogdet(Q)[1]
        else:
            Sinv = self._Sinv
            if Sinv is None:
                Sinv = 0

            Dinv = self._M + Sinv

            like = 0.5 * np.sum(self._j * I) - 0.5 * np.dot(I, np.dot(Dinv, I))

            if self._Sinv is not None:
                like += 0.5 * np.linalg.slogdet(2 * np.pi * Sinv)[1]

        return like + self._like_noise

    def solve_non_negative(self):
        """Compute the best fit solution with non-negative intensities"""
        Sinv = self._Sinv
        if Sinv is None:
            Sinv = 0

        Dinv = self._M + Sinv
        return scipy.optimize.nnls(Dinv, self._j,
                                   maxiter=100*len(self._j))[0]

    @property
    def mean(self):
        """Posterior mean, unit = Jy / sr"""
        return self._mu

    @property
    def MAP(self):
        """Posterior maximum, unit = Jy / sr"""
        return self.mean

    @property
    def covariance(self):
        """Posterior covariance, unit = (Jy / sr)**2"""
        if self._cov is None:
            self._cov = self.Dsolve(np.eye(self.size))
        return self._cov

    @property
    def power_spectrum(self):
        """Power spectrum coefficients"""
        return self._p

    @property
    def r(self):
        """Radius points, unit = arcsec"""
        return self._DHT.r * rad_to_arcsec

    @property
    def Rmax(self):
        """Maximum radius, unit = arcsec"""
        return self._DHT.Rmax * rad_to_arcsec

    @property
    def q(self):
        r"""Frequency points, unit = :math:`\lambda`"""
        return self._DHT.q

    @property
    def Qmax(self):
        r"""Maximum frequency, unit = :math:`\lambda`"""
        return self._DHT.Qmax

    @property
    def size(self):
        """Number of points in reconstruction"""
        return self._DHT.size



class LogNormalMAPModel:
 

    def __init__(self, DHT, M, j, p=None, scale=1.0, guess=None, noise_likelihood=0):

        self._DHT = DHT
        self._M = M
        self._j = j

        self._scale = 1.0

        self._p = p
        if p is not None:
            if np.any(p <= 0) or np.any(np.isnan(p)):
                raise ValueError("Bad value in power spectrum. The power"
                                 " spectrum must be postive and not contain"
                                 " any NaN values")

            Ykm = self._DHT.coefficients()
            self._Sinv = np.einsum('ji,j,jk->ik', Ykm, 1/p, Ykm)
        else:
            self._Sinv = None

        self._like_noise = noise_likelihood

        self._fit(guess)

    def _fit(self, guess):
        """Find the maximum likelihood solution and variance"""
        Sinv = self._Sinv
        if Sinv is None:
            Sinv = 0 * self._M

        scale = self._scale

        def H(s):
            """Log-likelihood function"""
            I = np.exp(scale * s)

            f = 0.5*np.dot(s, np.dot(Sinv, s))
            
            f += 0.5*np.dot(I, np.dot(self._M, I))
            f -= np.dot(I, self._j)
            
            return f
        
        def jac(s):  
            """1st Derivative of log-likelihood"""
            I = np.exp(scale * s)
                
            S1_s = np.dot(Sinv, s)
            
            MI = I * np.dot(self._M, I)
            jI = I * self._j
            
            return S1_s + scale*(MI - jI)
        
        def hess(s):
            """2nd derivative of log-likelihood"""
            I = np.exp(scale * s)
                
            Mij = np.einsum('i,ij,j->ij',I, self._M, I)
            MI = I * np.dot(self._M, I)
            jI = I * self._j
            
            term = scale**2 * (Mij + np.diag(MI - jI))
            return Sinv + term
        
        if guess is None:
            U, s_, V = scipy.linalg.svd(self._M + Sinv, full_matrices=False)
            s1 = np.where(s_ > 0, 1./s_, 0)
            I = np.dot(V.T, np.multiply(np.dot(U.T, self._j), s1))
            I = np.maximum(I, 1e-3*I.max())
            x = np.log(I) / scale 
        else:
            x = guess

        def limit_step(dx, x):
            alpha = 1.1*np.min(np.abs(x/dx))
                                    
            alpha = min(alpha, 1)
            return alpha*dx
           
        # Ignore convergence because it will often fail due to round off when
        # we're super close to the minimum
        search = LineSearch(reduce_step=limit_step)
        s, _ = MinimizeNewton(H, jac, hess, x, search, tol=1e-6)
 
        s  = self._s  = s
        I = np.exp(scale * s)
        
        # Now compute the inverse information propogator  
        Dinv = Sinv   
        Dinv += scale**2 * np.einsum('i,ij,j->ij', I, self._M, I)
        Dinv += scale**2 * np.diag(I * np.dot(self._M, I)) 
        Dinv -= scale**2 * np.diag(self._j * I)
        
        try:
            self._Dchol = scipy.linalg.cho_factor(Dinv)
            self._Dsvd  = None
        except np.linalg.LinAlgError:
            U, s_svd, V = scipy.linalg.svd(Dinv, full_matrices=False)
            
            s1 = np.where(s_svd > 0, 1./s_svd, 0)
            
            self._Dchol = None
            self._Dsvd  = U, s1, V

        self._cov = None
                
        return self._s

    def Dsolve(self, b):
        r"""
        Compute :math:`D \cdot b` by solving :math:`D^{-1} x = b`.

        Parameters
        ----------
        b : array, size = (N,...)
            Right-hand side to solve for

        Returns
        -------
        x : array, shape = np.shape(b)
            Solution to the equation D x = b

        """
        if self._Dchol is not None:
            return scipy.linalg.cho_solve(self._Dchol, b)
        else:
            U, s1, V = self._Dsvd
            return np.dot(V.T, np.multiply(np.dot(U.T, b), s1))

    @property
    def MAP(self):
        """Posterior maximum, unit = Jy / sr"""
        return self._s

    @property
    def covariance(self):
        """Posterior covariance at MAP, unit = (Jy / sr)**2"""
        if self._cov is None:
            self._cov = self.Dsolve(np.eye(self.size))
        return self._cov

    @property
    def power_spectrum(self):
        """Power spectrum coefficients"""
        return self._p

    @property
    def r(self):
        """Radius points, unit = arcsec"""
        return self._DHT.r * rad_to_arcsec

    @property
    def Rmax(self):
        """Maximum radius, unit = arcsec"""
        return self._DHT.Rmax * rad_to_arcsec

    @property
    def q(self):
        r"""Frequency points, unit = :math:`\lambda`"""
        return self._DHT.q

    @property
    def Qmax(self):
        r"""Maximum frequency, unit = :math:`\lambda`"""
        return self._DHT.Qmax

    @property
    def size(self):
        """Number of points in reconstruction"""
        return self._DHT.size

    @property 
    def scale(self):
        return self._scale
