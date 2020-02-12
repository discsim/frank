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
"""This module contains plotting routines for visualizing and analyzing
Frankenstein fits.
"""

import matplotlib.pyplot as plt

def plot_brightness_profile(fit_r, fit_i, ax, yscale='linear', comparison_profile=None):
    """ # TODO: add docstring
    """

    ax.plot(fit_r, fit_i / 1e10, 'r', label='Frank')

    if comparison_profile:
        ax.plot(comparison_profile[0], comparison_profile[1] / 1e10, 'c', label='Comparison profile')

    ax.set_xlabel('r ["]')
    ax.set_ylabel(r'Brightness [$10^{10}$ Jy sr$^{-1}$]')
    ax.set_yscale(yscale)


def plot_vis_fit(baselines, vis_fit, ax, xscale='log', yscale='linear',
                            comparison_profile=None):
    """ # TODO: add docstring
    """
    ax.plot(baselines, vis_fit, 'r')

    if comparison_profile:
        ax.plot(comparison_profile[0], comparison_profile[1], 'c', label='DHT of comparison profile')

    ax.set_xlabel(r'Baseline [$\lambda$]')
    ax.set_ylabel('Re(V) [Jy]')
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)


def plot_binned_vis(baselines, vis, vis_err, ax, xscale='log', yscale='linear', plot_CIs=False):
    """ # TODO: add docstring
    """
    if plot_CIs:
        ax.errorbar(baselines, vis, yerr=vis_err, fmt='r.', ecolor='#A4A4A4', label=r'Obs., %s k$\lambda$ bins'%binwidth)
    else:
        ax.plot(baselines, vis)

    ax.set_xlabel(r'Baseline [$\lambda$]')
    ax.set_ylabel('Re(V) [Jy]')
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

def plot_vis_resid(baselines, vis, vis_err, ax, xscale='log', yscale='linear', plot_CIs=False):
    """ # TODO: add docstring
    """

=============
    ax.plot(np.hypot(u,v), vis.real, 'k.')
    ax.plot(np.hypot(u,v), sol.predict(u,v).real,'g.')
    ax.plot(sol.q, sol.predict_deprojected(sol.q).real, 'r.')
    ax.set_yscale(yscale)


=================
        from constants import rad_to_arcsec
        import matplotlib as mpl
        mpl.rcParams['xtick.labelsize'] = 24
        mpl.rcParams['ytick.labelsize'] = 24

        grid = np.linspace(0, Rmax, 10000)

        ki = 2*np.pi*np.logspace(np.log10(min(baselines.min(), GPHF.uv_pts[0])),
                                 np.log10(max(baselines.max(), GPHF.uv_pts[-1])),
                                 10**4)

        '''
        resid = model(GPHF.Rc[:cut]) - mu[:cut]
        rmse = (np.mean(resid ** 2)) ** .5

        clean_r, clean_i, _ = np.genfromtxt(uvtable_dir + disc + '/' + disc + '_cleaned_image_radial_profile.txt').T
        interp_func = interp1d(clean_r, clean_i, kind='nearest', bounds_error=False)
        regrid_clean_i = interp_func(GPHF.Rc * rad_to_arcsec)[:-1]

        resid_clean = model(GPHF.Rc[:cut]) - regrid_clean_i
        rmse_clean = (np.nanmean(resid_clean ** 2)) ** .5 # NaNs arise
        # when this profile doesn't extend to as large radii as the fit
        '''

            integral = trapz(y=profile_guzman * j0(0) * grid2, x=grid2)

        ax1.plot(GPHF.Rc[:cut] * rad_to_arcsec, mu[:cut] / 1e10 / np.cos(inc * np.pi / 180), c=cs[counter_gp], label=line_labels[counter_gp], zorder=11)#label=r'Fit to Fedele et al. 2018 data') #AD13FF

        err = np.diag(GPHF.cov)**.5

        if plot_ci:
            ax1.fill_between(GPHF.Rc[:cut] * rad_to_arcsec, (mu[:cut] - err[:cut]) / 1e10 / np.cos(inc * np.pi / 180), (mu[:cut] + err[:cut]) / 1e10 / np.cos(inc * np.pi / 180), color=cs[counter_gp], alpha=.3, zorder=10)


        from scipy.interpolate import interp1d
        from scipy.ndimage import gaussian_filter

        def apply_guassian_smoothing(r, I, obs, n_per_sigma=5,axis=0):
            '''Apply Gaussian smoothing to radial profile'''

            # Setup the geometry for the smoothing mesh
            #   We align the beam with the grid (major axis aligned) and rotate the
            #   image accordingly
            beam = obs['beam']
            bmaj, bmin = beam['maj'], beam['min'] # FWHM in arcsec

            # Convert beam FWHM to sigma (and units to AU)
            bmaj = bmaj / np.sqrt(8*np.log(2))
            bmin = bmin / np.sqrt(8*np.log(2))

            PA = (obs['PA'] - beam['PA'])*np.pi/180.

            cos_i  = np.cos(obs['inclination']*np.pi/180.)
            cos_PA = np.cos(PA)
            sin_PA = np.sin(PA)

            # Pixel size in terms of bmin
            rmax = r.max()
            dx = bmin / n_per_sigma

            xmax = rmax*(np.abs(cos_i*cos_PA) + np.abs(sin_PA))
            ymax = rmax*(np.abs(cos_i*sin_PA) + np.abs(cos_PA))

            xmax = int(xmax/dx + 1)*dx
            ymax = int(ymax/dx + 1)*dx

            x = np.arange(-xmax, xmax+dx/2, dx)
            y = np.arange(-ymax, ymax+dx/2, dx)

            xi, yi = np.meshgrid(x,y)

            xp =  xi*cos_PA + yi*sin_PA
            yp = -xi*sin_PA + yi*cos_PA
            xp /= cos_i

            r1D = np.hypot(xi, yi)

            im_shape = r1D.shape + I.shape[1:]

            # Interpolate to grid and apply smoothing
            interp = interp1d(r, I, bounds_error=False, fill_value=0., axis=axis)

            I2D = interp(r1D.ravel()).reshape(*im_shape)
            sigma = [float(n_per_sigma), (bmaj/bmin)*n_per_sigma]
            print('sigma',sigma)
            #for i in range(I2D.shape[-1]):
            #    I2D[...,i] = gaussian_filter(I2D[...,i], sigma,
            #                                 mode='nearest', cval=0.)
            I2D = gaussian_filter(I2D, sigma, mode='nearest', cval=0.)

            # Now convert back to a 1D profile
            edges = np.concatenate(([r[0]*r[0]/r[1]], r, [r[-1]*r[-1]/r[-2]]))
            edges = 0.5*(edges[:-1] + edges[1:])

            I_smooth = np.empty_like(I)
            #for i in range(I2D.shape[-1]):
            #    I_smooth[...,i] = np.histogram(r1D.ravel(), weights=I2D[...,i].ravel(),
            #                                   bins=edges)[0]
            I_smooth = np.histogram(r1D.ravel(), weights=I2D.ravel(), bins=edges)[0]
            print('shape I_smooth',np.shape(I_smooth))
            counts = np.maximum(np.histogram(r1D.ravel(), bins=edges)[0],1)
            print('shape counts',np.shape(counts.reshape(-1,1)))
            print('counts',counts)
            #I_smooth /= counts.reshape(-1,1) # TODO: CHECK
            I_smooth /= counts
            return I_smooth

        if disc == 'as209':
            disc_dir = '/Users/morgan/gdrive_jmjenn/recipes/Frankenstein/results/dsharp/as209/'
            r, I = np.genfromtxt(disc_dir + 'fit.txt').T
            obs = {
                "inclination" : 0., # deg
                "PA" : 0.,
                "beam" : {
                  "maj" : 0.038216292858108, # arcsec
                  "min" : 0.036224924027904,
                  "PA" : 62.98761749268
                }
            }
        if disc == 'dr_tau':
            disc_dir = '/Users/morgan/gdrive_jmjenn/recipes/Frankenstein/results/taurus/dr_tau/'
            r, I = np.genfromtxt(disc_dir + 'fit.txt').T
            obs = {
                "inclination" : 0., # deg
                "PA" : 0.,
                "beam" : {
                  "maj" : 0.12737759947776, # arcsec
                  "min" : 0.091601334512244,
                  "PA" : 44.98328018188
                }
            }

        if disc == 'hd163296':
            disc_dir = '/Users/morgan/gdrive_jmjenn/recipes/Frankenstein/results/dsharp/hd163296/'
            r, I = np.genfromtxt(disc_dir + 'fit.txt').T
            obs = {
                "inclination" : 0., # deg
                "PA" : 0.,
                "beam" : {
                  "maj" : 0.038216292858108, # TODO: WRONG! JUST A PLACEHOLDER FROM AS209
                  "min" : 0.036224924027904,
                  "PA" : 62.98761749268
                }
            }
        #"""
        print('convolving profile with beam')
        convolved_profile = apply_guassian_smoothing(r, I, obs, n_per_sigma=5, axis=0)
        nonzero_idxs = np.nonzero(convolved_profile)
        r = r[nonzero_idxs]
        convolved_profile = convolved_profile[nonzero_idxs]
        print('done convolving')

        ax1.plot(r, convolved_profile / 1e10, c='#0691C4', ls=':', lw=3, label='Fit convolved with beam')#label=r'Fit to Fedele et al. 2018 data') #AD13FF
        print('max(convolved_profile / 1e10)',max(convolved_profile / 1e10))
        #"""
        #r2, I2 = np.genfromtxt(disc_dir + 'fit_convolved.txt').T
        #ax1.plot(r2, I2 / 1e10, 'm:')#label=r'Fit to Fedele et al. 2018 data') #AD13FF
        #"""
        if disc == 'as209':
            ax1.plot(grid * rad_to_arcsec, model(grid) / 1e10, c='#a4a4a4', ls='-.', lw=3, label='CLEAN, Andrews+18')
            #ax1.plot(GPHF.Rc[:cut] * rad_to_arcsec, regrid_clean_i / 1e10, c='#a4a4a4', label=r'CLEAN, Briggs 0.5')
        #"""
        if image_extracted_profile:
            if disc == 'dr_tau':
                clean_r, clean_i, _  = np.genfromtxt('/Users/morgan/gdrive_jmjenn/recipes/Frankenstein/results/taurus/dr_tau/dr_tau_cleaned_image_radial_profile.txt').T
                #clean_r, clean_i, _  = np.genfromtxt('/Users/morgan/gdrive_jmjenn/recipes/Frankenstein/results/taurus/dr_tau/DR_Tau_selfcal_cont_bin60s_1chan_nopnt_nofl_deeper_clean_clean_profile.txt').T
            if disc == 'as209':
                clean_r, clean_i, _  = np.genfromtxt('/Users/morgan/gdrive_jmjenn/recipes/Frankenstein/results/dsharp/as209/as209_cleaned_image_radial_profile.txt').T
            #clean_r, clean_i, _  = np.genfromtxt(pwd + '/anna_miotello/' + disc + '_cleaned_image_radial_profile.txt').T
            #clean_r, clean_i, _  = np.genfromtxt(pwd + '/anna2/' + disc + '_cleaned_image_radial_profile.txt').T
            interp_func = interp1d(clean_r, clean_i, kind='nearest', bounds_error=False)
            regrid_clean_i = interp_func(GPHF.Rc * rad_to_arcsec)[:-1]
            #if disc == 'dr_tau':
                #ax1.plot(GPHF.Rc[:cut] * rad_to_arcsec, regrid_clean_i / 1e10, c='#a4a4a4', ls='-.', label=r'CLEAN, Briggs 0.5')

        #'''
        #"""
        """
        ax1_5 = ax1.twiny()
        """
        ax1_5.spines['top'].set_color('#1A9E46')
        ax1_5.tick_params(axis='x', which='both', colors='#1A9E46')
        print('inc in plotting',inc)
        print('inc in plotting [rad]',inc * np.pi / 180)
        ax1_5.plot(GPHF.Rc[:cut] * rad_to_arcsec * dist, mu[:cut] / 1e10 / np.cos(inc * np.pi / 180), c=cs[counter_gp])
        ax1_5.set_xlabel('r [AU]', color='#1A9E46')
        #"""
        if disc == 'as209':
            ax1.set_xlim(-.05, 1.75)
            ax1_5.set_xlim(-.05 * dist, 1.75 * dist)
            ax6.set_xlim(-.05, 1.75)
        if disc == 'dr_tau':
            ax1.set_xlim(-.01, .5)
            ax1_5.set_xlim(-.01 * dist, .5 * dist)
            ax6.set_xlim(-.01, .5)
        if disc == 'ds_tau':
            ax1.set_xlim(-.01, .75)
            ax1_5.set_xlim(-.01 * dist, .75 * dist)
            ax6.set_xlim(-.01, .75)
            ax6.set_ylim(1e-3,2)

        #'''
        if disc == 'dr_tau':
            ax1.plot(GPHF.Rc[:cut] * rad_to_arcsec, regrid_clean_i / 1e10, c='#a4a4a4', ls='-.', lw=3, label=r'CLEAN, Briggs 0.5')
            ax6.plot(GPHF.Rc[:cut] * rad_to_arcsec, regrid_clean_i / 1e10, c='#a4a4a4', ls='-.', lw=3)#, label=r'CLEAN, Briggs 0.5')
        ax6.plot(GPHF.Rc[:cut] * rad_to_arcsec, mu[:cut] / 1e10 / np.cos(inc * np.pi / 180), c=cs[counter_gp], zorder=11)#, label='Fit')#label=r'Fit to Fedele et al. 2018 data') #AD13FF
        if plot_ci:
            ax6.fill_between(GPHF.Rc[:cut] * rad_to_arcsec, (mu[:cut] - err[:cut]) / 1e10 / np.cos(inc * np.pi / 180),  (mu[:cut] + err[:cut]) / 1e10 / np.cos(inc * np.pi / 180), color=cs[counter_gp], alpha=.3, zorder=10)


        #"""
        #ax6.plot(r, convolved_profile / 1e10 / np.cos(inc * np.pi / 180), c='#0691C4', ls=':', zorder=9)#label=r'Fit to Fedele et al. 2018 data') #AD13FF
        ax6.plot(r, convolved_profile / 1e10, c='#0691C4', ls=':', lw=3, zorder=9)#label=r'Fit to Fedele et al. 2018 data') #AD13FF
        #ax6.plot(r2, I2 / 1e10, 'm:')#label=r'Fit to Fedele et al. 2018 data') #AD13FF

        if disc == 'as209':
            ax6.plot(grid * rad_to_arcsec, model(grid) / 1e10, c='#a4a4a4', ls='-.')#, label='CLEAN')#label='Andrews et al. 2018 (for reference)')
            #ax6.plot(grid2, profile_guzman / 1e10, c='#3498DB', ls='-')#, label=r'Visibility fit, Guzm$\'{a}$n+18')#label='Andrews et al. 2018 (for reference)')
        #"""
        #plt.errorbar(GPHF.Rc * rad_to_arcsec, mu, yerr=np.diag(cov_renorm)**0.5, fmt='+', label='bins', zorder=-1)
        #plt.fill_between(GPHF.Rc * rad_to_arcsec, mu - np.diag(cov_renorm)**0.5, mu + np.diag(cov_renorm)**0.5, where=  mu - np.diag(cov_renorm)**0.5 >=  mu + np.diag(cov_renorm)**0.5, facecolor='r', alpha=.1)

        xlo1, xhi1 = ax1.get_xlim()

        #ax3 = fig1.add_subplot(gs1_double[2])
        #ax3 = fig1.add_subplot(gs1_double[4])
        """
        ax5 = fig1.add_subplot(gs1[1])
        ax2 = fig1.add_subplot(gs1[3])
        """
        if counter_gp == 0:
            uvbins, binned_re, binned_im, binned_re_err, binned_im_err = \
                uv.plot(linestyle='+', color='#A4A4A4',
                    label=r'Obs, %.0f k$\lambda$ bins' % (uvbin_size),
                    uvbin_size=uvbin_size, fontsize=24, return_binned_values=True)

            uvbins2, binned_re2, binned_im2, binned_re_err2, binned_im_err2 = \
                uv.plot(linestyle='x', linestyle2='+', color='#A4A4A4',
                    label=r'obs, %.0f k$\lambda$ bins' % (uvbin_size2),
                    uvbin_size=uvbin_size2, fontsize=24, return_binned_values=True)

            fn = savedir + 'uvbins.txt'
            fn2 = savedir + 'uvbins2.txt'
            np.savetxt(fn, np.array([uvbins, binned_re]).T, header='Baseline [lambda]\n Re(V) [Jy]')
            np.savetxt(fn2, np.array([uvbins2, binned_re2]).T, header='Baseline [lambda]\n Re(V) [Jy]')
            print('saved binned vis')
        #"""
        def fourier_beam(u, v, bmaj, bmin, pa):
            smaj = 2 * bmaj**2 / (8*np.log(2)) * np.pi**2
            smin = 2 * bmin**2 / (8*np.log(2)) * np.pi**2
            print('width', 2 * np.sqrt(np.log(2) / smaj))
            c = np.cos(pa)
            s = np.sin(pa)
            a = smaj*c*c + smin*s*s
            b = smaj*c*s - smin*c*s
            c = smaj*s*s + smin*c*c
            return np.exp( - (a*u*u + 2*b*u*v + c*v*v))
        ft_beam = fourier_beam(uv.u, uv.v, 1.852778161890806443e-7, 1.756233876588962206e-7, 62.98761749268 * np.pi / 180)
        conv_vis = ft_beam * uv.re
        import copy
        uv_conv = copy.deepcopy(uv)
        print('pre-convolved vis',uv_conv.re)
        uv_conv.re *= ft_beam
        print('post-convolved vis',uv_conv.re)

        uvbins_conv, binned_re_conv, binned_im_conv, binned_re_err_conv, binned_im_err_conv = \
            uv_conv.plot(linestyle='x', linestyle2='+', color='#A4A4A4',
                label=r'obs, %.0f k$\lambda$ bins' % (uvbin_size),
                uvbin_size=uvbin_size, fontsize=24, return_binned_values=True)
        uvbins_conv2, binned_re_conv2, binned_im_conv2, binned_re_err_conv2, binned_im_err_conv2 = \
            uv_conv.plot(linestyle='x', linestyle2='+', color='#A4A4A4',
                label=r'obs, %.0f k$\lambda$ bins' % (uvbin_size2),
                uvbin_size=uvbin_size2, fontsize=24, return_binned_values=True)
        #"""

        from scipy.special import j0, j1, jn_zeros, jv

        class DiscreteHankelTransform(object):
            """Utilities for computing the discrete Hankel Transform.

            This class provides the necessary interface to compute the
            a discrete version of the Hankel transform (DHT):

                H[f](q) = \int_0^R_{max} f(r) J_nu(2*pi*q*r) * 2*pi*r dr,

            The DHT is based on [1].

            Additionally, this class provides coef
            DHT [1] and provide Fourier-Bessel series representations.

            Parameters
            ----------
            Rmax : float
                Maximum radius beyond which f(r) is zero.
            N : integer
                Number of terms to use in the series
            nu : integer, default = 0
                Order of the Bessel function, J_nu(r)

            References
            ----------
            [1] Baddour & Chouinard (2015)
                DOI: https://doi.org/10.1364/JOSAA.32.000611
                Note: the definition of the DHT used here differs by factors
                of 2*pi.
            """
            def __init__(self, Rmax, N, nu=0):

                # Select the fast Bessel functions, if available.
                if nu == 0:
                    self._jnu0 = j0
                    self._jnup = j1
                elif nu == 1:
                    self._jnu0 = j1
                    self._jnup = lambda x: jv(2, x)
                else:
                    self._jnu0 = lambda x: jv(nu,   x)
                    self._jnup = lambda x: jv(nu+1, x)

                self._N = N
                self._nu = nu

                # Compute the collocation points
                j_nk = jn_zeros(nu, N+1)
                j_nk, j_nN = j_nk[:-1], j_nk[-1]

                Qmax = j_nN / (2*np.pi*Rmax)
                self._Rnk = Rmax * (j_nk / j_nN)
                self._Qnk = Qmax * (j_nk / j_nN)

                self._Rmax = Rmax
                self._Qmax = Qmax

                # Compute the weights Matrix
                Jnk = np.outer(np.ones_like(j_nk), self._jnup(j_nk))

                self._Ykm = (2 / (j_nN * Jnk * Jnk)) * \
                    self._jnu0(np.prod(np.meshgrid(j_nk, j_nk/j_nN), axis=0))

                self._scale_factor = 1 / self._jnup(j_nk)**2

                # Store the extra data needed
                self._j_nk = j_nk
                self._j_nN = j_nN

                self._J_nk = Jnk

            def transform(self, f, q=None, direction='forward'):
                """Computes the Hankel transform of an array.

                Parameters
                ----------
                f : array, size=N
                    Array to Hankel Transform
                q : array or None.
                    The frequency points at which to evaluate the Hankel
                    transform. If not specified the conjugate points of the
                    DHT will be used. For the backwards transform q should be
                    the radius points
                direction : { 'forward', 'backward' }, optional
                    Direction of the transform. If not supplied, the forward
                    transform is used.

                Returns
                -------
                H[f] : array, size=N or len(q) if supplied
                    The Hankel transform of the array f.
                """
                if q is None:
                    Y = self._Ykm

                    if direction == 'forward':
                        norm = (2*np.pi*self._Rmax**2)/self._j_nN
                    elif direction == 'backward':
                        norm = (2*np.pi*self._Qmax**2)/self._j_nN
                    else:
                        raise AttributeError("direction must be one of {}"
                                             "".format(['forward', 'backward']))
                else:
                    Y = self.coefficients(q, direction=direction)
                    norm = 1.0

                return norm * np.dot(Y, f)

            def coefficients(self, q=None, direction='forward'):
                """Coefficients of the transform matrix, defined by
                    H[f](q) = np.dot(Y, f).

                Parameters
                ----------
                q : array, or None
                    Frequency points to evaluate the transform at. If q=None, the
                    points of the DHT are used. If direction='backward' then these
                    points should be the radius points instead.
                direction : { 'forward', 'backward' }, optional
                    Direction of the transform. If not supplied, the forward
                    transform is used.

                Returns
                -------
                Y : array, size=(len(q),N)
                    The transformation matrix
                """
                if direction == 'forward':
                    norm = 1/(np.pi*self._Qmax**2)
                    k = 1./self._Qmax
                elif direction == 'backward':
                    norm = 1/(np.pi*self._Rmax**2)
                    k = 1./self._Rmax
                else:
                    raise AttributeError("direction must be one of {}"
                                         "".format(['forward', 'backward']))

                # For the DHT points we can use the cached Ykm points
                if q is None:
                    return 0.5*self._j_nN * norm * self._Ykm

                H = (norm * self._scale_factor) * \
                    self._jnu0(np.outer(k*q, self._j_nk))

                return H

            @property
            def roots(self):
                """Bessel function roots"""
                return self._j_nk

            @property
            def bessel_funcs(self):
                """Bessel functions"""
                return self._J_nk

            @property
            def r(self):
                """Radius points"""
                return self._Rnk

            @property
            def Rmax(self):
                """Maximum Radius"""
                return self._Rmax

            @property
            def q(self):
                """Frequency points"""
                return self._Qnk
            @property
            def Qmax(self):
                """Maximum Frequency"""
                return self._Qmax

            @property
            def size(self):
                """Number of points used in the DHT"""
                return self._N
            @property
            def order(self):
                """Order of the Bessel function"""
                return self._nu

        print('doing DHT of clean profile')
        dht = DiscreteHankelTransform(Rmax, 1000, nu=0)
        if disc == 'as209' or disc == 'hd163296' or disc == 'hd142666':
            i_colloc = np.interp(dht.r, grid, model(grid) * np.cos(inc * np.pi / 180))
            dht_input_profile = dht.transform(i_colloc, q=ki)
        if disc == 'dr_tau':
            i_colloc = np.interp(dht.r, clean_r / rad_to_arcsec, clean_i * np.cos(inc * np.pi / 180))
            print('baselines.min()',baselines.min())
            print('baselines.max()',baselines.max())
            print('len(baselines)',len(baselines))
            print('GPHF.uv_pts[0]',GPHF.uv_pts[0])
            print('min(ki)',min(ki))
            ki_for_clean_profile = np.logspace(np.log10(min(baselines.min(), GPHF.uv_pts[0])),
                                     np.log10(max(baselines.max(), GPHF.uv_pts[-1])),
                                     10**4)
            dht_input_profile = dht.transform(i_colloc, q=ki_for_clean_profile)
        print('done with DHT of clean profile')

        def find_nearest(array,value):
        	idx = (np.abs(array - value)).argmin()
        	#print 'nearest value %s. array position %s'%(array[idx],idx)
        	return idx

        #ax3.plot(GPHF.Rc[:cut] * rad_to_arcsec, resid_clean / max(model(GPHF.Rc[:cut])), '.', c='#a4a4a4',
        #     label='CLEAN, RMSE %.3f'% round(rmse_clean / max(model(GPHF.Rc[:cut])),3))
        #ax3.plot(GPHF.Rc[:cut] * rad_to_arcsec, resid[:cut] / max(model(GPHF.Rc[:cut])), '.', c='r', label=r'Fit, RMSE %.3f' % round(rmse / max(model(GPHF.Rc[:cut])),3))
        if counter_gp == 0:
            ax5.plot(uvbins, binned_re, '+', c='#a4a4a4', ms=8, label=r'Obs., %.0f k$\lambda$ bins'%(uvbin_size/1e3))
            ax5.plot(uvbins2, binned_re2, 'x', c='k', ms=8, label=r'%.0f k$\lambda$'%(uvbin_size2 / 1e3))
            '''
            ax5.plot(uvbins_conv, binned_re_conv, 'c.', alpha=.7, zorder=10)
            ax5.plot(uvbins_conv2, binned_re_conv2, 'm.', alpha=.7, zorder=11)
            '''
            #ax5.plot(uvbins, binned_im, 'x', c='g', ms=8, label=r'Im: Obs., %.0f k$\lambda$ bins'%(uvbin_size/1e3))
            #ax2.plot(baselines, uv.re, marker='.', ls='none', c='k', label='Obs. > 0') #k
            #ax2.plot(baselines, -uv.re, marker='.', ls='none', c='#A4A4A4', label='< 0')
            # plt.errorbar(baselines, uv.re, yerr=uv.weights**-0.5,marker='+', ls='none', c='k', label='obs')
            # plt.errorbar(baselines, -uv.re, yerr=uv.weights**-0.5,marker='x', ls='none', c='c')
            ax2.plot(uvbins, binned_re, marker='+', ms=8, ls='none', color='#a4a4a4') #33D9FF
            ax2.plot(uvbins, -binned_re, marker='+', ms=8, ls='none', color='#a4a4a4')#, label=r'< 0, %.0f k$\lambda$'%(uvbin_size/1e3)) #9CE73C
            ax2.plot(uvbins2, binned_re2, marker='x', ms=8, ls='none', c='k')
            ax2.plot(uvbins2, -binned_re2, marker='x', ms=8, ls='none', c='k')#, label=r'%.0f k$\lambda$'%(uvbin_size2 / 1e3)) #880901
            '''
            ax2.plot(uvbins_conv, binned_re_conv, 'c.', alpha=.7, zorder=10)
            ax2.plot(uvbins_conv2, binned_re_conv2, 'm.', alpha=.7, zorder=11)
            ax2.plot(uvbins_conv, -binned_re_conv, 'c.', alpha=.7, zorder=10)
            ax2.plot(uvbins_conv2, -binned_re_conv2, 'm.', alpha=.7, zorder=11)
            '''
            #ax2.plot(uvbins, binned_im, marker='x', ms=8, ls='none', color='g', label=r'Imaginary: %.0f k$\lambda$ bins'%(uvbin_size/1e3)) #33D9FF
            #ax2.plot(uvbins, -binned_im, marker='x', ms=8, ls='none', color='g')#, label=r'< 0, %.0f k$\lambda$'%(uvbin_size/1e3)) #9CE

            #ax2.errorbar(uvbins, binned_re, yerr=binned_re_err, marker='+', color='r', label=r'Obs. > 0, %.0f k$\lambda$ bins'%(uvbin_size * 50 / 1e3))
            #ax2.errorbar(uvbins, -binned_re, yerr=binned_re_err, marker='x', color='m', label=r'Obs. < 0, %.0f k$\lambda$ bins'%(uvbin_size * 50 / 1e3))
            #plt.errorbar(ki / (2 * np.pi), GPHF.HankelTransform(ki), yerr=np.diag(cov_renorm)**0.5, fmt='+', label='bins', zorder=-1)
        """
        fitb,fitv = np.genfromtxt(savedir + 'fit_visspace.txt').T
        ax5.plot(fitb,fitv,'c',label='Fit, all baselines')
        ax2.plot(fitb,fitv,'c')
        ax2.plot(fitb,-fitv,'c--')
        """
        #ax5.plot(ki / (2 * np.pi), GPHF.HankelTransform(ki), c='r', label=r'Fit, cut %.1f M$\lambda$'%(maxuv/1e6), zorder=10)#c='#AD13FF')#, label=r'%s bins, $\alpha$ =%s, ss=%.0e' % (nbins, alpha, smooth_strength))
        ax5.plot(ki / (2 * np.pi), GPHF.HankelTransform(ki), c=cs[counter_gp], label=r'Fit, all baselines', zorder=10)#c='#AD13FF')#, label=r'%s bins, $\alpha$ =%s, ss=%.0e' % (nbins, alpha, smooth_strength))
        #"""
        if disc == 'as209':
            ax5.plot(ki, dht_input_profile, '--', c='#2719FF', label='DHT of CLEAN profile')
        #"""
        if disc == 'dr_tau':
            ax5.plot(ki_for_clean_profile, dht_input_profile, '--', c='#2719FF', label='DHT of\nCLEAN profile')

        plot_log_abs(ki/(2*np.pi), GPHF.HankelTransform(ki), ax2, c=cs[counter_gp], zorder=10)#c='#AD13FF')#, alpha=.5)#, label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins,alpha,smooth_strength))
        #"""
        if disc == 'as209':
            ax2.plot(ki, dht_input_profile, '--', c='#2719FF')
            ax2.plot(ki, -dht_input_profile, '--', c='#2719FF')
        #"""
        if disc == 'dr_tau':
            ax2.plot(ki_for_clean_profile, dht_input_profile, '--', c='#2719FF')
            ax2.plot(ki_for_clean_profile, -dht_input_profile, '--', c='#2719FF')

        if disc == 'ds_tau' or disc == 'dl_tau' or disc == 'go_tau':
            #'''
            #ax4 = fig1.add_subplot(gssmall[9])
            ax4.set_ylabel('Power spectrum\n[arb. units]', fontsize=24)

            #gs_cbar = GridSpec(2, 8, right=.99)
            """
            plt.loglog(uv_pts_toplot[0], pi_toplot[0] * 2 * np.pi, c=cmaps[counter_gp](0), label=r'%s bins, $\alpha$ =%s, ss=%.0e, %s iterations'%(nbins, alpha, smooth_strength, smooth_iterations))

            print('uv_pts_toplot[0]',uv_pts_toplot[0])
            print('pi_toplot[0] * 2 * np.pi',(pi_toplot[0] * 2 * np.pi))

            for l in range(len(uv_pts_toplot)):
                plt.loglog(uv_pts_toplot[l], pi_toplot[l] * 2 * np.pi, c=cmaps[counter_gp](l / len(uv_pts_toplot)))
            """
            #ax4.loglog(uv_pts_toplot[-1], pi_toplot[-1] * 2 * np.pi, c=cs[counter_gp])
            pi_globalmax = np.amax(pi_toplot)
            ax4.loglog(GPHF.uv_pts, pi_toplot[-1], c=cs[counter_gp]) # TODO: is pi_toplot right (no missing multiplicative prefactor?)

            #plt.xlim(xlo1, xhi1)
            #plt.legend(fontsize=10)
            #plt.setp(ax4.get_xticklabels(), visible=False)
            ax4.set_xscale('log')

            #ax8 = fig1.add_subplot(gssmall[11])
            ax8.set_ylabel(r'SNR', fontsize=24)# = |Re(V)| $\cdot$ $\sqrt{\rm w}$', fontsize=24)
            #plt.xlim(xlo, xhi)
            #'''

        sortidx = np.argsort(baselines)
        bls = baselines[sortidx] / 1e3
        snr = abs(uv.re[sortidx]) / (1 / uv.weights[sortidx] ** .5)
        #binwidth = (max(bls) - min(bls)) / 128
        print('max(bls)',max(bls),'min(bls)',min(bls))
        binwidth = 50 # [klambda]

        bins = np.arange(0, max(bls) + binwidth, binwidth)
        mids = bins[:-1] + binwidth / 2
        bins_snr = np.vstack((np.digitize(bls, bins), snr)).T
        mean_snr = [np.mean(bins_snr[bins_snr[:, 0] == i, 1]) for i in
                    range(len(bins))]  # Runtimewarning here; how to ignore?
        std_snr = [np.std(bins_snr[bins_snr[:, 0] == i, 1]) for i in range(len(bins))]
        #print('mids',mids)
        #print('mean_snr',mean_snr,'std_snr',std_snr)
        print('len(mids)',len(mids),'len(mean_snr)',len(mean_snr))
        np.savetxt(savedir + 'binned_snr.txt', np.array([mids, mean_snr[1:]]).T)

        if disc == 'ds_tau' or disc == 'dl_tau' or disc == 'go_tau':
            #'''
            if counter_gp == 0:
                ax8.errorbar(mids * 1e3, mean_snr[1:], yerr=std_snr[1:], color='r',fmt='.', ecolor='#a4a4a4',
                             label='Obs., %.0f k$\lambda$ bins' % binwidth, zorder=10)

                print('mids * 1e3',(mids * 1e3))
                print('mean_snr[1:]',(mean_snr[1:]))

                ax8.axhline(1, c='k', ls='--', label='SNR = 1')
                ax8.legend(fontsize=24)

            #ax3.axhline(0, c='k', ls='--', zorder=10)
            #ax8.axhline(1, c='k', ls='--', label='SNR = 1', zorder=10)
            #eff_res = 5e6 #1.93e6 #2.45e6 #2.45e6
            #ax5.axvline(eff_res, zorder=20, c='#FF8B00', ls='--')
            #ax2.axvline(eff_res, zorder=20, c='#FF8B00', ls='--')
            #ax8.axvline(eff_res, zorder=20, c='#FF8B00', ls='--')
            #ax5.text(.7, .05, '  Effective\nresolution', transform=ax5.transAxes, fontsize=20, color='#FF8B00')
            if disc == 'dr_tau':
                ax8.set_ylim(5e-2,30)
            if disc == 'dr_tau':
                ax4.set_ylim(5e-12,5e-2)

            ax4.set_xscale('log')
            ax4.set_yscale('log')
            ax8.set_xscale('log')
            ax8.set_yscale('log')

            ax8.set_xlabel(r'Baseline [$\lambda$]', fontsize=24)
            ax8.text(.9, .9, 'f)', transform=ax8.transAxes, fontsize=24)
            ax4.text(.1, .1, 'e)', transform=ax4.transAxes, fontsize=24)
            #'''

        ax5.axhline(0, zorder=-10, c='#2ECC71', ls='--', label='Re(V) = 0')

        ax1.legend(fontsize=24)
        #ax3.legend(fontsize=24, ncol=1)
        #ax6.legend(fontsize=24, ncol=1)
        ax2.legend(fontsize=24, loc='lower left')
        #ax8.legend(fontsize=24)
        if disc == 'dr_tau':
            ax5.legend(fontsize=20, loc='upper right')
        if disc == 'as209':
            ax5.legend(fontsize=20, loc='lower left')

        #ax3.set_xlim(xlo1, xhi1)
        #ax6.set_xlim(xlo1, xhi1)
        if disc == 'go_tau':
            ax5.set_xlim(8e3, 4e6)
            ax2.set_xlim(8e3, 4e6)
            ax2.set_ylim(1e-5, .2) # band6, band3
            #ax5.set_ylim(-.025, .025) # band3
            ax4.set_xlim(8e3, 4e6)
            ax8.set_xlim(8e3, 4e6)
            #ax4.set_ylim(5e-12,5e-2)
            ax8.set_ylim(1e-1,30)
        if disc == 'ds_tau':
            ax5.set_xlim(5e3, 3e6)
            ax2.set_xlim(5e3, 3e6)
            ax2.set_ylim(1e-5, 5e-2) # band6, band3
            ax5.set_ylim(-.025, .025) # band6
            ax4.set_xlim(5e3, 3e6)
            ax8.set_xlim(5e3, 3e6)
            ax8.set_ylim(1e-1,30)
            #ax1.set_xlim(0,.75)
            #ax6.set_xlim(0,.75)
        if disc == 'dl_tau':
            ax5.set_xlim(5e3, 3e6)
            ax2.set_xlim(5e3, 3e6)
            ax2.set_ylim(1e-5, 2e-1)
            #ax5.set_ylim(-.025, .025)
            ax4.set_xlim(5e3, 3e6)
            ax8.set_xlim(5e3, 3e6)
            ax8.set_ylim(1e-1,30)
        if disc == 'dr_tau':
            ax5.set_xlim(1e4, 5e6)
            ax2.set_xlim(1e4, 5e6)
            ax2.set_ylim(1e-5, .2)
            #ax5.set_ylim(-.03,.16)
            #ax2.set_ylim(1e-5,.2)
            #ax1.set_ylim(-.5, 13)
            ax6.set_ylim(1e-3, 15)
        #ax3.set_ylim(-.52,.52)
        if disc == 'as209':
            ax5.set_xlim(1e4, 1e7)
            ax2.set_xlim(1e4, 1e7)
            """
            ax5.set_ylim(-.025,.025)
            """
            ax5.set_ylim(-.02,.02)
            ax2.set_ylim(1e-5,.5)
            ax1.set_ylim(-.2,7)
            ax6.set_ylim(1e-4,7)
        #ax8.set_ylim(1e-2,5)
        if disc == 'dr_tau':
            xlo, xhi = [1e4, 5e6] #ax5.get_xlim()
        else:
            xlo, xhi = ax5.get_xlim() #[9e3, 1e7] #ax5.get_xlim()
            print('xlo',xlo,' xhi',xhi)

        #ax1.set_ylim(-1,12)
        if disc == 'J16011549-4152351.cont_v2':
            ax1.set_ylim(-.05,1.2)
            ax2.set_ylim(1e-5,.1)
            ax5.set_ylim(-.02,.02)
            ax2.set_xlim(1e3,2e6)
            ax5.set_xlim(1e3,2e6)

        #xlo, xhi = ax2.get_xlim()
        #ax5.set_xlim(1e4, xhi)
        #ax2.set_xlim(1e4, xhi)
        ax2.set_yscale('log')
        ax6.set_yscale('log')
        ax5.set_xscale('log')
        ax2.set_xscale('log')

        #ax3.set_yticks([-.5,0,.5])
        #ax6.set_yticks([1e-4,1e-2,1e0])

        ax1.set_ylabel('Brightness [$10^{10}$ Jy sr$^{-1}$]', fontsize=24)
        ax6.set_ylabel('Brightness [$10^{10}$ Jy sr$^{-1}$]', fontsize=24)
        ax6.set_xlabel('r ["]', fontsize=24)
        #ax3.set_ylabel('Normalized\nresidual', fontsize=24)
        ax5.set_ylabel('Re(V) [Jy]', fontsize=24)
        ax2.set_ylabel('Re(V) [Jy]', fontsize=24)
        #ax8.set_ylabel('SNR', fontsize=24)
        ax2.set_xlabel(r'Baseline [$\lambda$]', fontsize=24)

        ax1.text(.9, .5, 'a)', transform=ax1.transAxes, fontsize=24)
        ax6.text(.9, .8, 'b)', transform=ax6.transAxes, fontsize=24)
        #ax3.text(.9, .82, 'c)', transform=ax3.transAxes, fontsize=24)
        if disc == 'dr_tau':
            ax5.text(.05, .5, 'c)', transform=ax5.transAxes, fontsize=24)
        if disc == 'as209':
            ax5.text(.05, .8, 'c)', transform=ax5.transAxes, fontsize=24)
        else: ax5.text(.05, .5, 'c)', transform=ax5.transAxes, fontsize=24)
        if disc == 'go_tau' or disc == 'ds_tau' or disc == 'dl_tau':
            ax2.text(.05, .8, 'd)', transform=ax2.transAxes, fontsize=24)
        else:
            ax2.text(.05, .65, 'd)', transform=ax2.transAxes, fontsize=24)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax5.get_xticklabels(), visible=False)
        #plt.setp(ax2.get_xticklabels(), visible=False)
        #plt.setp(ax4.get_xticklabels(), visible=False)
        #plt.setp(ax6.get_xticklabels(), visible=False)

        """


        #ax8.text(.1, .1, 'f)', transform=ax8.transAxes, fontsize=24)

        #ax9.set_ylim(1e-3, 1e1)

        #ax10.semilogx(uvbins, resid_vis, '.', c='#33D9FF')


        """
        '''
        ax4 = fig1.add_subplot(gs1_single[5])
        resid = binned_re - GPHF.HankelTransform(uvbins)
        rmse = (np.mean(resid ** 2)) ** .5
        ax4.semilogx(uvbins, (binned_re - GPHF.HankelTransform(uvbins)) / max(binned_re), label='RMSE %.2d'% (rmse / max(binned_re)))
        ax4.set_ylabel('Normalized residual', fontsize=24)
        xlo, xhi = ax4.get_xlim()
        ax4.set_xlim(xlo, xhi)
        #plt.legend(fontsize=10)
        plt.setp(ax4.get_xticklabels(), visible=False)
        '''
        '''
        ax4 = fig1.add_subplot(gs1_double[9])
        plt.ylabel('Power', fontsize=24)

        #gs_cbar = GridSpec(2, 8, right=.99)
        plt.loglog(uv_pts_toplot[0], pi_toplot[0] * 2 * np.pi, c=cmaps[counter_gp](0))#, label=r'%s bins, $\alpha$ =%s, ss=%.0e, %s iterations'%(nbins, alpha, smooth_strength, smooth_iterations))
        for l in range(len(uv_pts_toplot)):
            plt.loglog(uv_pts_toplot[l], pi_toplot[l] * 2 * np.pi, c=cmaps[counter_gp](l / len(uv_pts_toplot)))
        plt.xlim(xlo, xhi)

        plt.xlim(xlo, 1e7)

        #plt.legend(fontsize=10)
        plt.setp(ax4.get_xticklabels(), visible=False)
        '''
        '''
        ax6 = fig1.add_subplot(gs1_small[12])
        plt.ylabel(r'$\chi^2$', fontsize=24)

        #plt.plot(np.array(chi2s_alt) / len(uv.u), '.', label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins, alpha, smooth_strength))
        plt.plot(np.array(chi2s) / len(uv.u), label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins, alpha, smooth_strength))
        #plt.legend(fontsize=10)

        plt.setp(ax6.get_xticklabels(), visible=False)


        ax9 = fig1.add_subplot(gs1_small[14])
        plt.xlabel('Fit iteration #', fontsize=24)
        plt.ylabel('Con.#(i$_{cov}$)', fontsize=24)

        plt.semilogy(condition_numbers, label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins, alpha, smooth_strength))
        plt.yticks(ax9.get_ylim(), ('%.1e'%ax9.get_ylim()[0], '%.1e'%ax9.get_ylim()[1]))
        '''
        #'''
        """
        #ax8 = fig1.add_subplot(gs1_single[7])
        #ax8 = fig1.add_subplot(gs1_single[5])
        #ax8 = fig1.add_subplot(gs1_double[9])
        ax8 = fig1.add_subplot(gs1_single2[3])
        #ax8.text(.95, .8, 'f)', transform=ax8.transAxes, fontsize=24)
        #ax8.text(.1, .1, 'f)', transform=ax8.transAxes, fontsize=24)
        #plt.ylabel(r'SNR = |Re(V)| $\cdot$ $\sqrt{\rm w}$', fontsize=24)
        plt.ylabel(r'SNR', fontsize=24)
        plt.xlabel(r'Baseline [$\lambda$]', fontsize=24)
        xlo, xhi = ax1.get_xlim()
        ax8.set_xlim(xlo, xhi)

        #ax8.set_xlim(xlo, 4e6)
        #ax8.set_xlim(xlo, 1e7)

        #ax8.set_ylim(1e-1, 1e1)
        #ax8.set_ylim(1e-3, 1e1)
        ax8.set_ylim(0, 5)
        print('start snr')
        if counter_gp == 0:
            #'''

            #plt.legend(fontsize=24, loc=[.5, .05])
        #'''
        """
        '''
        plt.axhline(0, c='#A4A4A4', ls='--')
        plt.xlabel('Baseline [$\lambda$]', fontsize=24)
        plt.ylabel('Im(V) [Jy]', fontsize=24)
        if counter_gp == 0:
            plt.errorbar(uvbins, binned_im, yerr=binned_im_err, marker='.', ls='none', c='k', label='obs, %.0f k$\lambda$ bins'%(uvbin_size/1e3))
        ax8.set_xscale('log')
        plt.xlim(xlo, xhi)
        plt.legend(fontsize=10)
        '''

        '''
        corr = cov_renorm.copy()
        for i in range(len(mu)):
            corr[i] /= np.sqrt(cov_renorm[i, i])
            corr[:, i] /= np.sqrt(cov_renorm[i, i])
        '''
        #"""
        #ax10.axvline(eff_res, zorder=20, c='r', ls='--', label='RMSE[:eff. res.] %.3f'%round(rmse_vis,3))
        #ax8.axvline(eff_res, zorder=20, c='b', ls='--')
        #ax5.legend(loc=[.4,.3])
        #ax10.legend(fontsize=24, ncol=1)
        #"""
        fig1.savefig(plot_msfigs_savefile)#, dpi=600)
        #plt.show()
