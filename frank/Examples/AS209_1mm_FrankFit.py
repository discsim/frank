import frank
import numpy as np 
import matplotlib.pyplot as plt

from frank.geometry import SourceGeometry
from frank.radial_fitters import FrankFitter, FourierBesselFitter

# Huang 2018 
inc = 34.97
pa = 85.76
dra = 1.9e-3
ddec = -2.5e-3
r_out = 1.9

# Frank Parameters
n_pts = 300
alpha = 1.3
w_smooth = 1e-1

# UVtable AS209 at 1mm with removed visibilities between .
dir = "./"
data_file = dir + 'AS209_continuum_prom_1chan_30s_keepflagsFalse_removed1.txt'

# Loading data
u, v, Re, Im, Weights = np.loadtxt(data_file, unpack = True, skiprows = 1)
vis = Re + Im*1j

geom = SourceGeometry(inc= inc, PA= pa, dRA= dra, dDec= ddec)

" Fitting with frank "
#FF = FrankFitter(r_out, n_pts, geom, alpha = alpha, weights_smooth = w_smooth)
FF = FourierBesselFitter(r_out, n_pts, geom)
sol = FF.fit(u, v, vis, Weights)
#setattr(sol, 'positive', sol.solve_non_negative())

fig = plt.figure(num = 1, figsize = (10, 5))
plt.plot(sol.r, sol.mean / 1e10)
plt.xlabel('Radius ["]', size = 15)
plt.ylabel(r'Brightness profile [$10^{10}$ Jy sr$^{-1}$]', size = 15)
plt.title('Frank Fit AS209 1mm', size = 15)
plt.xlim(0, 1.3)
#plt.savefig('FrankFit_AS209_1mm.jpg', bbox_inches='tight')
plt.show()
plt.close()