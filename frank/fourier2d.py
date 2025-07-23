
import numpy as np
import time

class DiscreteFourierTransform2D(object):
    def __init__(self, Rmax, N, nu=0):
        # Remember that now N is to create N**2 points in image plane.
        self.Xmax = Rmax #Rad
        self.Ymax = Rmax
        self.Nx = N 
        self.Ny = N
        self.N = self.Nx*self.Ny # Number of points we want to use in the 2D-DFT.

        # Real space collocation points
        x1n = np.linspace(-self.Xmax, self.Xmax, self.Nx, endpoint=False) # rad
        x2n = np.linspace(-self.Ymax, self.Ymax, self.Ny, endpoint=False) # rad
        x1n, x2n = np.meshgrid(x1n, x2n, indexing='ij')
        # x1n.shape = N**2 X 1, so now, we have N**2 collocation points in the image plane.
        x1n, x2n = x1n.reshape(-1), x2n.reshape(-1) # x1n = x_array and x2n = y_array
        self.dx = 2*self.Xmax/self.Nx
        self.dy = 2*self.Ymax/self.Ny

        # Frequency space collocation points.
        q1n = np.fft.fftfreq(self.Nx, d = self.dx)
        q2n = np.fft.fftfreq(self.Ny, d = self.dy)
        q1n, q2n = np.meshgrid(q1n, q2n, indexing='ij') 
        # q1n.shape = N**2 X 1, so now, we have N**2 collocation points.
        q1n, q2n = q1n.reshape(-1), q2n.reshape(-1) # q1n = u_array and q2n = v_array

        self.Xn = x1n
        self.Yn = x2n
        self.Un = q1n
        self.Vn = q2n

    def get_collocation_points(self):        
        return np.array([self.Xn, self.Yn]), np.array([self.Un, self.Vn])

    def coefficients(self, u = None, v = None, x = None, y = None, direction="forward"):
        #start_time = time.time()
        if direction == 'forward':
            ## Normalization is dx*dy since we the DFT to be an approximation
            ## of the integral (which depends on the area)
            norm = 4*self.Xmax*self.Ymax/self.N
            factor = -2j*np.pi
            
            X, Y = self.Xn, self.Yn
            if u is None:
                u = self.Un
                v = self.Vn
        elif direction == 'backward':
            ## Correcting for the normalization above 1/N is replaced by this:
            norm = 1 / (4*self.Xmax*self.Ymax)
            factor = 2j*np.pi

            X, Y = self.Un, self.Vn
            if u is None:
                u = self.Xn
                v = self.Yn
        else:
            raise AttributeError("direction must be one of {}"
                                 "".format(['forward', 'backward']))
        H = norm * np.exp(factor*(np.outer(u, X) + np.outer(v, Y)))
        #print("--- %s minutes to calculate 2D-DFT coefficients---" % (time.time()/60 - start_time/60))
        return H


    def transform(self, f, u=None, v=None, direction="forward"):
        Y = self.coefficients(u, v, direction=direction)
        return np.dot(Y, f)
          
          
    @property
    def size(self):
        """Number of points used in the 2D-DFT"""
        return self.N
    
    @property
    def uv_points(self):
        """u and v  collocation points"""
        return self.Un, self.Vn

    @property
    def q(self):
        """Frequency points"""
        return np.hypot(self.Un, self.Vn)
    
    @property
    def Rmax(self):
        """ Maximum value of the x coordinate in rad"""
        return self.Xmax
    
    @property
    def resolution(self):
        """ Resolution of the grid in the x coordinate in rad"""
        return self.dx
    
    @property
    def xy_points(self):
        """ Collocation points in the image plane"""
        return self.Xn, self.Yn