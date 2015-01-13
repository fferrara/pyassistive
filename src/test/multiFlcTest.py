""" lms.py: a simple python class for Least mean squares adaptive filter """

from __future__ import division
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import os
from ..util import config,preprocessing, featex
import matplotlib.pyplot as pl

class MultiFLC(object):
    def __init__(self, W, mu):
        self.W = np.squeeze( getattr( W, "A", W ))  # matrix -> array
        # Sufficient for stability: 0 < mu < 1 / tr(R)
        # R = E[x * x.T]  = np.corrcoef(x.T)
        self.mu = mu

    def est(self, X, y):
        X = np.squeeze( getattr( X, "A", X ))
        yest = self.W.dot(X)
        c = (y - yest) / X.dot(X)

        self.W += 2 * self.mu * c * X
        return yest
#...............................................................................
if __name__ == "__main__":
    import sys

    modelorder = 2 * 4 # four w0
    damp = .01
    nx = 600 * 3
    noise = .05 * 2  # * swing
    plot = 1
    seed = 0

    DATA_FILE = "protocolo 7/andres_prot7_config1.mat"

    # read data
    data = sio.loadmat(os.path.join(config.DATA_PATH, DATA_FILE))
    X = data['data']

    # CAR FILTER & PASSBAND FILTER
    X -= X.mean(axis=0)
    X = np.dot(X, preprocessing.CAR(X.shape[1]))

    Fs = 600
    freqs = [5.6, 6.4, 6.9, 8]

    Wcritic = np.array([0., 4., 5., 49., 50., 300.])
    b, a = preprocessing._get_fir_filter(Wcritic, Fs, 251)
    X = signal.lfilter(b, (a,), X, axis=0)

    channel = 11 # Oz
    y1 = X[1*Fs:40*Fs, channel:channel+1] # protocol first part, 20s x 2

    f,psd = preprocessing.get_psd(y1[:,0], 3, Fs, config.NFFT)
    pl.plot(f,psd)
    pl.show()

    t = np.arange( nx + 0. ) / 600.
    x = np.empty((2 * modelorder, t.shape[0]))
    for i in range(len(freqs)):
        x[i * 4:(i+1) * 4,:] = featex.generate_references(t.shape[0], freqs[i], Fs, 2).T

    title = "LMS  chirp  filterlen %d  nx %d  noise %.2g  mu %.2g " % (
        modelorder, nx, noise, damp )
    print title
    ys = []
    yests = []

#...............................................................................
    lms = MultiFLC( np.zeros(2 * modelorder), mu=damp )
    for t in xrange( nx - 1):
        X = x[:,t:t+1]
        y = y1[t+1,0]  # predict
        yest = lms.est( X, y)
        ys += [y]
        yests += [yest]

    y = np.array(ys)
    yest = np.array(yests)
    print y.shape, yest.shape
    err = (yest - y)**2
    averr = "av %.2g += %.2g" % (err.mean(), err.std())
    print "LMS yest - y:", averr
    print "LMS weights:", lms.W
    if plot:
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

