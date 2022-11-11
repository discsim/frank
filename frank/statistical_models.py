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

class VisibilityMapping:
    r"""Builds the mapping between the visibility and image planes.

    VisibilityMapping generates the transform matrices :math:`H(q)` such that
    :math:`V_\nu(q) = H(q) I_\nu`. It also uses these to construct the design
    matrices :math:`M` and :math:`j` used in the fitting. 
    
    VisibilityMapping supports following models:
      1. An optically thick and geometrically thin disc
      2. An optically thin and geometrically thin disc
      3. An opticall thin disc with a known Gaussian verictal structure.
    All models are axisymmetric.

    Parameters
    ----------
    DHT : DiscreteHankelTransform
        A DHT object with N bins that defines H(p). The DHT is used to compute
        :math:`S(p)`
    geometry: SourceGeometry object, optional
        Geometry used to correct the visibilities for the source inclination. 
    vis_model : string,
        One of ['opt_thick', 'opt_thin', 'debris'], corresponding to models
        1-3 described above, respectively. 
    scale_height : function H(R), units = arcsec
        The vertical thickness of the disc in terms of its Guassian scale-height.
        Only used if vis_model="debris".
    block_data : bool, default = True
        Large temporary matrices are needed to set up the data. If block_data
        is True, we avoid this, limiting the memory requirement to block_size
        elements.
    block_size : int, default = 10**5
        Size of the matrices if blocking is used
    check_qbounds: bool, default = True
        Whether to check if the first (last) collocation point is smaller
        (larger) than the shortest (longest) deprojected baseline in the dataset
    verbose : bool, default = False
        Whether to print notification messages
    """
    def __init__(self, DHT, geometry,  
                 vis_model='opt_thick', scale_height=None, block_data=True,
                 block_size=10 ** 5, check_qbounds=True, verbose=True):
        
        _vis_models = ['opt_thick', 'opt_thin', 'debris']
        if vis_model not in _vis_models:
            raise ValueError(f"vis_model must be one of {_vis_models}")

        # Store flags
        self._vis_model = vis_model
        self.check_qbounds = check_qbounds
        self._verbose = verbose 

        self._chunking = block_data
        self._chunk_size = block_size

        self._DHT = DHT
        self._geometry = geometry

        # Check for consistency and report the model choice.
        self._scale_height = None
        if self._vis_model == 'opt_thick':
            if self._verbose:
                logging.info('  Assuming an optically thick model (the default): '
                             'Scaling the total flux to account for the source '
                             'inclination')
        elif self._vis_model == 'opt_thin':
            if self._verbose:
                logging.info('  Assuming an optically thin model: *Not* scaling the '
                             'total flux to account for the source inclination')
        elif self._vis_model == 'debris':
            if scale_height is None:
                raise ValueError('You requested a model with a non-zero scale height'
                                 ' but did not specify H(R) (scale_height=None)')
            self._scale_height = scale_height
            self._H2 = 0.5*(2*np.pi*scale_height(self.r) / rad_to_arcsec)**2
            
            if self._verbose:
                logging.info('  Assuming an optically thin model but geometrically: '
                             'thick model: *Not* scaling the total flux to account for '
                             'the source inclination')
   
    def map_visibilities(self, u, v, V, weights, frequencies=None, geometry=None):
        r"""
        Compute the matrices :math:`M` abd :math:`j` from the visibility data.

        Also compute the null likelihood,
        .. math:
            `H0 = 0.5*\log[det(weights/(2*np.pi))]
             - 0.5*np.sum(V * weights * V):math:`
             
        Parameters
        ----------
        u,v : 1D array, unit = :math:`\lambda`
            uv-points of the visibilies
        V : 1D array, unit = Jy
            Visibility amplitudes at q
        weights : 1D array, optional, unit = J^-2
            Weights of the visibilities, weight = 1 / sigma^2, where sigma is
            the standard deviation
        frequencies : 1D array, optional, unit = Hz
            Channel frequencies of each data point. If not provided, a single 
            channel is assumed.
        geometry : SourceGeometry object
            Geometry used to deproject the visibilities before fitting. If not
            provided the geometry passed in during construction will be used.
        
        Returns
        -------
        mapped visibilities : dict. 
            The format will be depend whether a single channel was assumed 
            (frequencies=None). The following data maybe present:
               'multi_freq' : bool,
                    Specifies if mult-frequency analysis was done
               'channels' : array, 1D, mult-frequency analysis only
                    The frequency of each channel.
                'M' : array, 2D or 3D.
                    The matrix :math:`M` of the mapping. This will have shape
                    (num_channels, size, size) for multi_freq analysis and
                    (size, size) for single channel analysis.
                'j' : array, 1D or 2D.
                    The matrix :math:`j` of the mapping. This will have shape
                    (num_channels, size) for multi_freq analysis and (size,)
                    for single channel analysis.
                'null_likelihood' : float,
                    The likelihood of a model with I=0. See above.
                'hash' : list,
                    Identifying data, used to ensure compatability between the
                    mapped visibilies and fitting objects.
        """

        if geometry is None:
            geometry = self._geometry

        if self._verbose:
            logging.info('    Building visibility matrices M and j')
        
        # Deproject the visibilities
        u, v, k, V = self._geometry.apply_correction(u, v, V, use3D=True)
        q = np.hypot(u, v)

        # Check consistency of the uv points with the model
        self._check_uv_range(q)

        # Use only the real part of V. 
        V = V.real
        w = np.ones_like(V) * weights

        multi_freq = True
        if frequencies is None:
            multi_freq = False
            frequencies = np.ones_like(V)

        channels = np.unique(frequencies)
        Ms = np.zeros([len(channels), self.size, self.size], dtype='f8')
        js = np.zeros([len(channels), self.size], dtype='f8')
        for i, f in enumerate(channels):
            idx = frequencies == f

            qi = q[idx]
            ki = k[idx]
            wi = w[idx]
            Vi = V[idx]
        
            # If chunking is used, we will build up M and j chunk-by-chunk
            if self._chunking:
                Nstep = int(self._chunk_size / self.size + 1)
            else:
                Nstep = len(Vi)

            start = 0
            end = Nstep
            Ndata = len(Vi)
            M = Ms[i]
            j = js[i]
            while start < Ndata:
                qs = qi[start:end]
                ks = ki[start:end]
                ws = wi[start:end]
                Vs = Vi[start:end]

                X = self._get_mapping_coefficients(qs, ks)

                wXT = np.array(X.T * ws, order='C')

                M += np.dot(wXT, X)
                j += np.dot(wXT, Vs)

                start = end
                end += Nstep

        # Compute likelihood normalization H_0, i.e., the
        # log-likelihood of a source with I=0.
        H0 = 0.5 * np.sum(np.log(w / (2 * np.pi)) - V * w * V)

        if multi_freq:
            return {
                'mult_freq' : True,
                'channels' : channels,
                'M' : Ms,
                'j' : js,
                'null_likelihood' : H0,
                'hash' : [True, self._DHT, geometry, self._vis_model, self._scale_height],
            }
        else: 
            return {
                'mult_freq' : False,
                'M' : Ms[0],
                'j' : js[0],
                'null_likelihood' : H0,
                'hash' : [False, self._DHT, geometry, self._vis_model, self._scale_height],
            }

    def check_hash(self, hash, multi_freq=False, geometry=None):
        """Checks whether the hash of some mapped visibilities are compatible
        with this VisibilityMapping.
        
        Parameters
        ----------
        hash : list
            Hash to compare
        multi_freq : bool
            Whether we are expected a multi-frequncy fit
        geometry : SourceGeometry object, optional
            Geometry to use in the comparison.
        """
        if geometry is None:
            geometry = self._geometry
        
        passed = (
            multi_freq == hash[0] and
            self._DHT.Rmax  == hash[1].Rmax and
            self._DHT.size  == hash[1].size and
            self._DHT.order == hash[1].order and
            geometry.inc  == hash[2].inc and
            geometry.PA   == hash[2].PA and
            geometry.dRA  == hash[2].dRA and
            geometry.dDec == hash[2].dDec and
            self._vis_model == hash[3]
        )

        if not passed:
            return False

        if self._scale_height is None:
            return hash[4] is None
        else:
            if hash[4] is None:
                return False
            else:
                return np.alltrue(self._scale_height(self.r) == hash[4](self.r))


    def predict_visibilities(self, I, q, k=None, geometry=None):
        r"""Compute the predicted visibilities given the brightness profile, I
        
        Parameters
        ----------
        I : array, unit = Jy
            Brightness at the collocation points.
        q : array, unit = :math:`\lambda`
            Radial uv-distance to predict the visibility at
        k : array, optional,  unit = :math:`\lambda`
            Vertical uv-distance to predict the visibility. Only needed for a
            geometrically thick model.
        geometry : SourceGeometry object, optional
            Geometry used to correct the visibilities for the source
            inclination. Only needed for the optically thick model. If not 
            provided, the geometry passed in during construction will be used. 
        
        Returns
        -------
        V(q, k) : array, unit = Jy
            Predicted visibilties of a source with a radial flux distribution
            given by :math:`I` and the position angle, inclination determined
            by the geometry.
        
        Notes
        -----
        For an optically thick model the visibility amplitudes are reduced due
        to the projection but phase centre corrections are not added.
        """
        # Chunk the visibility calulation for speed
        if self._chunking:
            Ni = int(self._chunk_size / self.size + 1)
        else:
            Ni = len(q)

        end = 0
        start = 0
        V = []
        while end < len(q):
            start = end
            end = start + Ni
            qi = q[start:end]
            
            ki = None
            if k is not None:
                ki = k[start:end]

            H = self._get_mapping_coefficients(qi, ki, geometry)

            V.append(np.dot(H, I))
        return np.concatenate(V)

    def invert_visibilities(self, V, R, geometry=None):
        r"""Compute the brightness, I, from the visibilities. 
        
        Note this method does not work for an arbitrary distribution of 
        baselines and therefore cannot be used to determine the brightness
        given a generic set of data. Instead it needs the visibilities at 
        collocation points of the DiscrteHankelTransform, q.

        For geometrically thick models the visibilities used must be those for
        which kz = 0.

        Given the above constraints, this method computes the inverse of
        predict_visibilites.
        
        Parameters
        ----------
        V : array, unit = Jy
            Visibility at the collocation points. This must be the deprojected
            and phase-center corrected visibilities.
        R : array, unit = arcsec
            Radial distance to compute the brightness at
        geometry : SourceGeometry object, optional
            Geometry used to correct the visibilities for the source
            inclination. Only needed for the optically thick model. If not 
            provided, the geometry passed in during construction will be used. 
        
        Returns
        -------
        I(R) : array, unit = Jy / Sr
            Brightness at the radial locations, R. 
        
        Notes
        -----
        The brightness is corrected under the optically thin
        """
        # Chunk the visibility calulation for speed
        R = np.atleast_1d(R)
        if self._chunking:
            Ni = int(self._chunk_size / self.size + 1)
        else:
            Ni = len(R)

        end = 0
        start = 0
        I = []
        while end < len(R):
            start = end
            end = start + Ni
            Ri = R[start:end]

            H = self._get_mapping_coefficients(Ri, 0, geometry, inverse=True)

            I.append(np.dot(H, V))
        return np.concatenate(I)[R < self.Rmax]

    def transform(self, f, q=None, direction='forward'):
        """Apply a DHT directly to data provided
        
        Parameters
        ----------
        f : array, size = N
            Function to Hankel transform, evaluated at the collocation points:
                f[k] = f(r_k) or f[k] = f(q_k)
        q : array or None
            The frequency points at which to evaluate the Hankel
            transform. If not specified, the conjugate points of the
            DHT will be used. For the backwards transform, q should be
            the radius points in arcsec
        direction : { 'forward', 'backward' }, optional
            Direction of the transform. If not supplied, the forward
            transform is used

        Returns
        -------
        H[f] : array, size = N or len(q) if supplied
            The Hankel transform of the array f

        """
        if direction == 'backward' and q is not None:
            q = q / rad_to_arcsec
            
        return self._DHT.transform(f, q, direction)


    def _get_mapping_coefficients(self, qs, ks, geometry=None, inverse=False):
        """Get :math:`H(q)`, such that :math:`V(q) = H(q) I_\nu`"""
        
        if self._vis_model == 'opt_thick':
            # Optically thick & geometrically thin
            if geometry is None:
                geometry = self._geometry
            scale = np.cos(geometry.inc * deg_to_rad)
        elif self._vis_model == 'opt_thin':
            # Optically thin & geometrically thin
            scale = 1
        elif self._vis_model == 'debris':
            # Optically thin & geometrically thick
            scale = np.exp(-np.outer(ks*ks, self._H2))
        else:
            raise ValueError("model not supported. Should never occur.")

        if inverse:
            scale = np.atleast_1d(1/scale).reshape(1,-1)
            qs = qs / rad_to_arcsec
            direction='backward'
        else:
            direction='forward'

        H = self._DHT.coefficients(qs, direction=direction) * scale

        return H


    def _check_uv_range(self, uv):
        """Check that the uv domain is properly covered"""

        # Check whether the first (last) collocation point is smaller (larger)
        # than the shortest (longest) deprojected baseline in the dataset
        if self.check_qbounds:
            if self.q[0] < uv.min():
                logging.warning(r"WARNING: First collocation point, q[0] = {:.3e} \lambda,"
                                " is at a baseline shorter than the"
                                " shortest deprojected baseline in the dataset,"
                                r" min(uv) = {:.3e} \lambda. For q[0] << min(uv),"
                                " the fit's total flux may be biased"
                                " low.".format(self.q[0], uv.min()))

            if self.q[-1] < uv.max():
                raise ValueError(r"ERROR: Last collocation point, {:.3e} \lambda, is at"
                                 " a shorter baseline than the longest deprojected"
                                 r" baseline in the dataset, {:.3e} \lambda. Please"
                                 " increase N in FrankMultFrequencyFitter (this is"
                                 " `hyperparameters: n` if you're using a parameter"
                                 " file). Or if you'd like to fit to shorter maximum baseline,"
                                 " cut the (u, v) distribution before fitting"
                                 " (`modify_data: baseline_range` in the"
                                 " parameter file).".format(self.q[-1], uv.max()))

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
    def scale_height(self):
        "Vertial thickness of the disc, unit = arcsec"
        if self._scale_height is not None:
            return self._scale_height(self.r)
        else:
            return None

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
    M : 2/3D array, size = (N, N) or (Nf, N, N)
        The design matrix, see above. Nf is the number of channels (frequencies).
    j : 1/2D array, size = N or (Nf, N)
        Information source, see above
    p : 1D array, size = N, optional
        Power spectrum used to generate the covarience matrix :math:`S(p)`
    guess: array, optional
        Initial guess used in computing the brightness.
    Nfields : int, optional
        Number of fields used to fit the data. Typically 1, but could be more
        for multi-wavelength data. If not provided it will be determined from
        the shape of guess.
    noise_likelihood : float, optional
        An optional parameter needed to compute the full likelihood, which
        should be equal to

        .. math::
            -\frac{1}{2} V^T w V + \frac{1}{2} \sum \log[w/(2 \pi)].

        If not  provided, the likelihood can still be computed up to this
        missing constant
    """

    def __init__(self, DHT, M, j, p=None, scale=None, guess=None,
                 Nfields=None, noise_likelihood=0):

        self._DHT = DHT

        # Correct shape of design matrix etc.        
        if len(M.shape) == 2:
            M = M.reshape(1, *M.shape)
        if len(j.shape) == 1:
            j = j.reshape(1, *j.shape)

        # Number of frequencies / radial points
        Nf, Nr = j.shape
        
        # Get the number of fields
        if Nfields is None:
            if guess is None:
                Nfields = 1
            else:
                guess = guess.reshape(-1, Nr)
                Nfields = guess.shape[0]
        elif guess is not None:
            guess = guess.reshape(Nfields, Nr)
        self._Nfields = Nfields
        
        # Create the correct shape for the power spectrum and scale factors
        if p is not None:
            p = p.reshape(-1, Nr)
            if p.shape[0] == 1 and Nfields > 1:
                p = np.repeat(p, Nfields, axis=0)
        self._p = p
                
        if scale is None:
            self._scale = np.ones([Nf, Nfields], dtype='f8')
        else:
            self._scale = np.empty([Nf, Nfields], dtype='f8')
            self._scale[:] = scale.reshape(Nf, -1)       

        if p is not None:
            if np.any(p <= 0) or np.any(np.isnan(p)):
                print(p)
                raise ValueError("Bad value in power spectrum. The power"
                                 " spectrum must be positive and not contain"
                                 " any NaN values. This is likely due to"
                                 " your UVtable (incorrect units or weights), "
                                 " or the deprojection being applied (incorrect"
                                 " geometry and/or phase center). Else you may"
                                 " want to increase `rout` by 10-20% or `n` so"
                                 " that it is large, >~300.")

            Ykm = self._DHT.coefficients()
            Sj = np.einsum('ji,lj,jk->lik', Ykm, 1/p, Ykm)

            self._Sinv = np.zeros([Nr*Nfields, Nr*Nfields], dtype='f8')
            for n in range(0, Nfields):
                sn = n*Nr
                en = (n+1)*Nr 

                self._Sinv[sn:en, sn:en] += Sj[n]
        else:
            self._Sinv =  None

        # Compute the design matrix
        self._M = np.zeros([Nr*Nfields, Nr*Nfields], dtype='f8')
        self._j = np.zeros(Nr*Nfields, dtype='f8')
        for si, Mi, ji in zip(self._scale, M, j):
            
            for n in range(0, Nfields):
                sn = n*Nr
                en = (n+1)*Nr 

                self._j[sn:en] += si[n] * ji
                for m in range(0, Nfields):
                    sm = m*Nr
                    em = (m+1)*Nr 

                    self._M[sn:en, sm:em] += si[n]*si[m] * Mi

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
        if self._Nfields > 1:
            self._mu = self._mu.reshape(self._Nfields, self.size)
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
        draws = np.random.multivariate_normal(self.mean.reshape(-1), self.covariance, N)
        if self.num_fields > 1:
            draws = draws.reshape(N, self.num_fields, self.size)
        return draws

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
            self._cov = self.Dsolve(np.eye(self.size*self.num_fields))
        return self._cov

    @property
    def s_0(self):
        return 0

    @property
    def power_spectrum(self):
        """Power spectrum coefficients"""
        if self.num_fields == 1 and self._p is not None:
            return self._p.reshape(self.size)
        return self._p
    
    @property
    def num_fields(self):
        """Number of fields fit for"""
        return self._Nfields

    @property
    def size(self):
        """Number of points in reconstruction"""
        return self._DHT.size


class LogNormalMAPModel:
    r"""
    Finds the maximum a posteriori field for log-normal regression problems,

    .. math::
       P(s|q,V,p,s0) \propto G(H exp(scale*(s+s0)) - V, M) P(s|p)

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
    M : 2/3D array, size = (N, N) or (Nf, N, N)
        The design matrix, see above. Nf is the number of channels (frequencies).
    j : 1/2D array, size = N or (Nf, N)
        Information source, see above
    p : 1D array, size = N, optional
        Power spectrum used to generate the covarience matrix :math:`S(p)`
    scale : float, 1D array (size=N), or list of
        Scale factors s0 (see above). These factors can be a constant, one
        per brightness point or per band (optionally per collocation point)
        to enable multi-frequency fitting.
    s0 : float or 1D array, size Nfields. Optional
        Zero point for the brightness function, see above.
    guess: array, optional
        Initial guess used in computing the brightness.
    Nfields : int, optional
        Number of fields used to fit the data. Typically 1, but could be more
        for multi-wavelength data. If not provided it will be determined from
        the shape of guess.
    full_hessian: float, range [0, 1]
        If 1 then use the full Hessian in evaluating the matrix :math:`D`. When
        0 a term is omitted that can cause the Hessian not be a positive-
        definite matrix when the solution is a poor fit to the data. A value
        between 0 and 1 will scale this term by that factor.
    noise_likelihood : float, optional
        An optional parameter needed to compute the full likelihood, which
        should be equal to

        .. math::
            -\frac{1}{2} V^T w V + \frac{1}{2} \sum \log[w/(2 \pi)].

        If not  provided, the likelihood can still be computed up to this
        missing constant
    """ 

    def __init__(self, DHT, M, j, p=None, scale=None, s0=None, guess=None, 
                 Nfields=None, full_hessian=1, noise_likelihood=0):

        self._DHT = DHT
        self._full_hess = full_hessian

        # Correct shape of design matrix etc.        
        if len(M.shape) == 2:
            M = M.reshape(1, *M.shape)
        if len(j.shape) == 1:
            j = j.reshape(1, *j.shape)

        self._M = M
        self._j = j

        # Number of frequencies / radial points
        Nf, Nr = j.shape
        # Number of signal fields:
        # Get the number of fields
        if Nfields is None:
            if guess is None:
                Nfields = 1
            else:
                guess = guess.reshape(-1, Nr)
                Nfields = guess.shape[0]
        elif guess is not None:
            guess = guess.reshape(Nfields, Nr)
        self._Nfields = Nfields

        
        # Create the correct shape for the power spectrum and scale factors
        if p is not None:
            p = p.reshape(-1, Nr)
            if p.shape[0] == 1 and Nfields > 1:
                p = np.repeat(p, Nfields, axis=0)
        self._p = p

                
        if scale is None:
            self._scale = np.ones([Nf, Nfields], dtype='f8')
        else:
            self._scale = np.empty([Nf, Nfields], dtype='f8')
            self._scale[:] = scale.reshape(Nf, -1)        

        if s0 is None:
            self._s0 = np.ones(Nfields, dtype='f8')
        else:
            s0 = np.atleast_1d(s0)
            if len(s0) == 1:
                s0 = np.repeat(s0, Nfields)
            elif len(s0) != Nfields:
                raise ValueError("Signal zero-point (s0) must have the same "
                                 "length as the number of fields or length 1")
        self._s0 = s0.reshape(Nfields, 1)

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
            self._Sinv = np.einsum('ji,lj,jk->lik', Ykm, 1/p, Ykm)
        else:
            self._Sinv = np.zeros([Nfields, Nr, Nr], dtype='f8')

        self._like_noise = noise_likelihood

        self._fit(guess)

    def _fit(self, guess):
        """Find the maximum likelihood solution and variance"""
        Sinv = self._Sinv
        Ns, Nr = Sinv.shape[:2]
        
        Sflat = np.zeros([Ns*Nr, Ns*Nr])
        for i in range(Ns):
            s = i*Nr
            e = (i+1)*Nr
            Sflat[s:e,s:e] = Sinv[i]


        scale = self._scale
        s0 = self._s0

        def H(s):
            """Log-likelihood function"""
            s = s.reshape(Ns, Nr)
            I = np.exp(np.dot(scale,s+s0)) # shape Nf, Nr

            f  = 0.5*np.einsum('ij,ijk,ik', s, Sinv, s)

            f += 0.5*np.einsum('ij,ijk,ik',I,self._M,I)
            f -= np.sum(I*self._j)
            
            return f
        
        def jac(s):  
            """1st Derivative of log-likelihood"""
            s = s.reshape(Ns, Nr)
            I = np.exp(np.dot(scale,s+s0)) # shape Nf, Nr

            sI = np.einsum('is,ij->isj',scale, I) # shape Nf, Ns, Nr

            S1_s = np.einsum('sjk,sk->sj', Sinv, s)  # shape Ns, Nr
            MI = np.einsum('isj,ijk,ik->sj', sI, self._M, I)
            jI = np.einsum('isj,ij->sj', sI, self._j)
            
            return (S1_s + (MI - jI)).reshape(Ns*Nr)
        
        def hess(s):
            """2nd derivative of log-likelihood"""
            s = s.reshape(Ns, Nr)
            I = np.exp(np.dot(scale,s+s0)) # shape Nf, Nr
            
            sI = np.einsum('is,ij->isj',scale, I) # shape Nf, Ns, Nr
            s2I = np.einsum('is,ij->isj',scale**2, I) # shape Nf, Ns, Nr
            
            Mjk = np.einsum('isj,ijk,itk->sjtk',sI, self._M, sI)

            resid = 0
            if self._full_hess > 0:
                MI = Mjk.sum(3) 
                jI = np.einsum('is,itj,ij->sjt', scale, sI, self._j)

                resid = np.einsum('sjt,jk->sjtk', MI-jI, np.eye(Nr)).reshape(Ns*Nr, Ns*Nr)
                if self._full_hess < 1:
                    resid *= self._full_hess    
            
            return Mjk.reshape(Ns*Nr, Ns*Nr) + resid + Sflat
        
        x = guess.reshape(Ns*Nr)

        def limit_step(dx, x):
            alpha = 1.1*np.min(np.abs(x/dx))
                                    
            alpha = min(alpha, 1)
            return alpha*dx
           
        # Ignore convergence because it will often fail due to round off when
        # we're super close to the minimum
        search = LineSearch(reduce_step=limit_step)
        s, _ = MinimizeNewton(H, jac, hess, x, search, tol=1e-7)
        self._s_MAP = s.reshape(Ns, Nr)

        Dinv = hess(s)
        try:
            self._Dchol = scipy.linalg.cho_factor(Dinv)
            self._Dsvd  = None
        except np.linalg.LinAlgError:
            U, s_svd, V = scipy.linalg.svd(Dinv, full_matrices=False)
            
            s1 = np.where(s_svd > 0, 1./s_svd, 0)
            
            self._Dchol = None
            self._Dsvd  = U, s1, V

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
        """Compute N draws from the (approximate) posterior"""
        draws = np.random.multivariate_normal(self.MAP.reshape(-1), self.covariance, N)
        if self.num_fields > 1:
            draws = draws.reshape(N, self.num_fields, self.size)
        return draws    

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

        like = - 0.5*np.dot(s-self._s0, np.dot(Sinv, s-self._s0))
            
        like -= 0.5*np.einsum('ij,ijk,ik',I,self._M,I)
        like += np.sum(I*self._j)
            

        if self._Sinv is not None:
            like += 0.5 * np.linalg.slogdet(2 * np.pi * Sinv)[1]

        return like + self._like_noise

    def solve_non_negative(self):
        """Compute the best fit solution with non-negative intensities"""
        # The solution is alway non-negative. Provided for convenience.
        return self.MAP
        
    @property
    def MAP(self):
        """Posterior maximum, unit = Jy / sr"""
        MAP = self._s_MAP
        if MAP.shape[0] == 1:
            return MAP.reshape(self.size)
        return MAP

    @property
    def covariance(self):
        """Posterior covariance at MAP, unit = (Jy / sr)**2"""
        if self._cov is None:
            self._cov = self.Dsolve(np.eye(self.size*self.num_fields))
        return self._cov

    @property
    def power_spectrum(self):
        """Power spectrum coefficients"""
        p = self._p
        if p.shape[0] == 1:
            return p.reshape(self.size)
        return p

    @property 
    def scale(self):
        scale = self._scale
        if scale.shape[1] == 1:
            return scale[:,0]
        return self._scale
    
    @property 
    def s_0(self):
        s0 = self._s0
        if s0.shape[0] == 1:
            return s0[0]
        return s0
    
    @property
    def num_fields(self):
        """Number of fields fit for"""
        return self._Nfields
        
    @property
    def size(self):
        """Number of points in reconstruction"""
        return self._DHT.size
