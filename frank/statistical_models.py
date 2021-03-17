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
                                 " spectrum must be positive and not contain"
                                 " any NaN values. This is likely due to"
                                 " your UVtable (incorrect units or weights), "
                                 " or the deprojection being applied (incorrect"
                                 " geometry and/or phase center). Else you may"
                                 " want to increase `rout` by 10-20% or `n` so"
                                 " that it is large, >~300.")

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
              \log[P(I,V|p)] = j^T I - \frac{1}{2} I^T D^{-1} I
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

            like = np.sum(self._j * I) - 0.5 * np.dot(I, np.dot(Dinv, I))

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
    def size(self):
        """Number of points in reconstruction"""
        return self._DHT.size


class LogNormalMAPModel:
    r"""
    Finds the maximum a posteriori field for log-normal regression problems,

    .. math::
       P(s|q,V,p,s0) \propto G(H exp(s*s0) - V, M) P(s|p)

    where :math:`s` is the log-intensity to be predicted, :math:`q` are the
    baselines and :math:`V` the visibility data. :math:`\mu` and :math:`H` is
    the design matrix of the transform, e.g. the coefficient matrix of
    the forward Hankel transform.

    If :math:`p` is provided, the covariance matrix of the prior is included,
    with

    .. math::
        P(s|p) \propto G(s, S(p)),

    The problem is framed in terms of the design matrix :math:`M` and
    information source :math:`j`.

    :math:`H(q)` is the matrix that projects the intensity :math:`exp(s*s0)` to
    visibility space. :math:`M` is defined by

    .. math::
        M = H(q)^T w H(q),

    where :math:`w` is the weights matrix and

    .. math::
        j = H(q)^T w V.


    The maximum a posteori field, s_MAP, is found by maximizing 
    :math:`\log P(s|q,V,p,s0)` and the posterior covariance at s_MAP is

    .. math::
        D = [ M + S(p)^{-1}]^{-1}.

    If the prior is not provided then

    .. math::
        D = M^{-1}.

    and the posterior for exp(s) is the same as the standard Gaussian model.

    Note: This class also supports :math:`M` and :math:`j` being split into
    multiple terms (i.e. :math:`M_i, j_i`) such that different scale 
    factors, :math:`s0_i` can be applied to each system. This allows fitting
    multi-frequency data.


    Parameters
    ----------
    DHT : DiscreteHankelTransform
        A DHT object with N bins that defines H(p). The DHT is used to compute
        :math:`S(p)`
    M : 2D array, size = (N, N), or list of
        The design matrix, see above
    j : 1D array, size = N, or list of
        Information source, see above
    p : 1D array, size = N, optional
        Power spectrum used to generate the covarience matrix :math:`S(p)`
    scale : float, 1D array (size=N), or list of
        Scale factors s0 (see above). These factors can be a constant, one
        per brightness point or per band (optionally per collocation point)
        to enable multi-frequency fitting.
    noise_likelihood : float, optional
        An optional parameter needed to compute the full likelihood, which
        should be equal to

        .. math::
            -\frac{1}{2} V^T w V + \frac{1}{2} \sum \log[w/(2 \pi)].

        If not  provided, the likelihood can still be computed up to this
        missing constant
    """ 

    def __init__(self, DHT, M, j, p=None, scale=1.0, guess=None, noise_likelihood=0):

        self._DHT = DHT

        # Correct shape of design matrix etc.
        scale = np.atleast_1d(scale)
        if len(scale) == 1:
            if len(M.shape) == 2:
                M = [M,]
            if len(j.shape) == 1:
                j = [j,]

        self._M = M
        self._j = j
        self._scale = scale


        self._p = p
        if p is not None:
            if np.any(p <= 0) or np.any(np.isnan(p)):
                raise ValueError("Bad value in power spectrum. The power"
                                 " spectrum must be positive and not contain"
                                 " any NaN values. This is likely due to"
                                 " your UVtable (incorrect units or weights), "
                                 " or the deprojection being applied (incorrect"
                                 " geometry and/or phase center). Else you may"
                                 " want to increase `rout` by 10-20% or `n` so"
                                 " that it is large, >~300.")

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
            I = np.exp(np.einsum('i,j->ij', scale,s))

            f = 0.5*np.dot(s, np.dot(Sinv, s))
            
            f += 0.5*np.einsum('ij,ijk,ik',I,self._M,I)
            f -= np.sum(I*self._j)
            
            return f
        
        def jac(s):  
            """1st Derivative of log-likelihood"""
            I = np.exp(np.einsum('i,j->ij', scale, s))
            sI = (I.T*scale).T

            S1_s = np.dot(Sinv, s) 
            MI = np.einsum('ij,ijk,ik->j', sI, self._M, I)
            jI = np.einsum('ij,ij->j', sI, self._j)
            
            return S1_s + (MI - jI)
        
        def hess(s):
            """2nd derivative of log-likelihood"""
            I = np.exp(np.einsum('i,j->ij', scale, s))
            s2I = (I.T*scale**2).T
            
            Mij = np.einsum('ki,kij,kj->ij',s2I, self._M, I)
            MI = np.einsum('ij,ijk,ik->j', s2I, self._M, I)
            jI = np.einsum('ij,ij->j', s2I, self._j)
            
            term = (Mij + np.diag(MI - jI))
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
 
        s  = self._s_MAP  = s
        I = np.exp(np.einsum('i,j->ij', scale, s))
        s2I = (I.T*scale**2).T

        # Now compute the inverse information propogator  
        Dinv = Sinv.copy()
        Dinv += np.einsum('ki,kij,kj->ij', s2I, self._M, I)
        Dinv += np.diag(np.einsum('ij,ijk,ik->j', s2I, self._M, I))
        Dinv -= np.diag(np.einsum('ij,ij->j', s2I, self._j))
        
        try:
            self._Dchol = scipy.linalg.cho_factor(Dinv)
            self._Dsvd  = None
        except np.linalg.LinAlgError:
            U, s_svd, V = scipy.linalg.svd(Dinv, full_matrices=False)
            
            s1 = np.where(s_svd > 0, 1./s_svd, 0)
            
            self._Dchol = None
            self._Dsvd  = U, s1, V

        self._cov = None
                
        return self._s_MAP

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
    

    def log_likelihood(self, s=None):
        r"""
        Compute the likelihood,

        .. math:
            \log[P(I,V|S)].

        Parameters
        ----------
        s : array, size = N, optional
            Log-intensity :math:`I(r)=exp(s0*s)` to compute the likelihood of

        Returns
        -------
        log_P : float
            Log likelihood, :math:`\log[P(I,V|p)]`

        Notes
        -----
        1. The prior probability P(S) is not included.
        2. The likelihood takes the form:

        .. math::
              \log[P(I,V|p)] = j^T I - \frac{1}{2} I^T D^{-1} I
                 - \frac{1}{2} \log[\det(2 \pi S)] + H_0

        where

        .. math::
            H_0 = -\frac{1}{2} V^T w V + \frac{1}{2} \sum \log(w /2 \pi)

        is the noise likelihood.
        """

        if s is None:
            s = self._s_MAP 
        
        Sinv = self._Sinv
        if Sinv is None:
            Sinv = 0


        I = np.exp(np.einsum('i,j->ij', self._scale,s))

        like = - 0.5*np.dot(s, np.dot(Sinv, s))
            
        like -= 0.5*np.einsum('ij,ijk,ik',I,self._M,I)
        like += np.sum(I*self._j)
            

        if self._Sinv is not None:
            like += 0.5 * np.linalg.slogdet(2 * np.pi * Sinv)[1]

        return like + self._like_noise

    @property
    def MAP(self):
        """Posterior maximum, unit = Jy / sr"""
        return self._s_MAP

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
    def scale(self):
        return self._scale

    @property
    def size(self):
        """Number of points in reconstruction"""
        return self._DHT.size