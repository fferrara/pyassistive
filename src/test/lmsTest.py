""" lms.py: a simple python class for Least mean squares adaptive filter """

from __future__ import division
import numpy as np
import pickle

__version__ = "2013-08-29 aug denis"

#...............................................................................
class LMS:
    """ lms = LMS( Wt, damp=.5 )  Least mean squares adaptive filter
    in:
        Wt: initial weights, e.g. np.zeros( 33 )
        damp: a damping factor for swings in Wt

    # for t in range(1000):

    yest = lms.est( X, y [verbose=] )
    in: X: a vector of the same length as Wt
        y: signal + noise, a scalar
        optional verbose > 0: prints a line like "LMS: yest y c"
    out: yest = Wt.dot( X )
        lms.Wt updated

    How it works:
    on each call of est( X, y ) / each timestep,
    increment Wt with a multiple of this X:
        Wt += c X
    What c would give error 0 for *this* X, y ?

        y = (Wt + c X) . X
        =>
        c = (y  -  Wt . X)
            --------------
               X . X

    Swings in Wt are damped a bit with a damping factor a.k.a. mu in 0 .. 1:
        Wt += damp * c * X

    Notes:
        X s are often cut from a long sequence of scalars, but can be anything:
        samples at different time scales, seconds minutes hours,
        or for images, cones in 2d or 3d x time.

"""
#...............................................................................
    def __init__( self, Wt, mu):
        self.Wt = np.squeeze( getattr( Wt, "A", Wt ))  # matrix -> array
        self.mu = mu

    def est( self, X, y, verbose=0 ):
        X = np.squeeze( getattr( X, "A", X ))
        yest = self.Wt.dot(X)
        c = (y - yest) / X.dot(X)

        self.Wt += 2 * self.mu * c * X
        if verbose:
            print "LMS: yest %-6.3g   y %-6.3g   err %-5.2g   c %.2g" % (
                yest, y, yest - y, c )
        return yest

#...............................................................................
if __name__ == "__main__":
    import sys

    modelorder = 2
    damp = .01
    nx = 600 * 1
    freq = 6.4  # chirp
    freq2 = 5.6
    noise = .05 * 2  # * swing
    plot = 1
    seed = 0

    t = np.arange( nx + 0. ) / 600.
    # d = np.sin(2 * np.pi * 5.6 * t)
    # d += np.random.normal( scale=noise, size=nx ) 
    # d *= 10
    fp = open("../shared5.6.pkl")
    POz20sec = pickle.load(fp)
    d = POz20sec[3000:3000+nx,0]

    x = np.empty((2 * modelorder, t.shape[0]))
    x[0,:] = np.sin(2 * np.pi * freq * t)
    x[1,:] = np.cos(2 * np.pi * freq * t)
    x[2,:] = np.sin(4 * np.pi * freq * t)
    x[3,:] = np.cos(4 * np.pi * freq * t)

    title = "LMS  chirp  filterlen %d  nx %d  noise %.2g  mu %.2g " % (
        modelorder, nx, noise, damp )
    print title
    ys = []
    yests = []

#...............................................................................
    lms = LMS( np.zeros(2 * modelorder), mu=damp )
    for t in xrange( nx - 1):
        X = x[:,t:t+1]
        y = d[t+1]  # predict
        yest = lms.est( X, y, verbose = (t % 10 == 0) )
        ys += [y]
        yests += [yest]

    y = np.array(ys)
    yest = np.array(yests)
    err = (yest - y)**2
    averr = "av %.2g += %.2g" % (err.mean(), err.std())
    print "LMS yest - y:", averr
    print "LMS weights:", lms.Wt
    if plot:
        import matplotlib.pyplot as pl
        fig, ax = pl.subplots( nrows=2 )
        fig.set_size_inches( 12, 8 )
        fig.suptitle( title, fontsize=12 )
        ax[0].plot( y, color="orangered", label="y" )
        ax[0].plot( yest, label="yest" )
        ax[0].legend()
        ax[1].plot( err, label=averr )
        ax[1].legend()
        if plot >= 2:
            pl.savefig( "tmp.png" )
        pl.show()

