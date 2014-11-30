""" lms.py: a simple python class for Least mean squares adaptive filter """

from __future__ import division
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import os
from util import config,preprocessing, processing
import matplotlib.pyplot as pl
from scipy import stats

class MultiFLC(object):
    def __init__(self, W, mu):
        self.W = np.squeeze( getattr( W, "A", W ))  # matrix -> array
        # Sufficient for stability: 0 < mu < 1 / tr(R)
        # R = E[x * x.T]  = np.corrcoef(x.T)
        self.mu = mu

    def est(self, X, y):
        X = np.squeeze( getattr( X, "A", X ))
        yest = self.W.dot(X)

        if X.ndim > 1:
            norm = np.diag(X.T.dot(X))
            c = (y - yest) / norm
            self.W += 2 * self.mu * c.dot(X.T)
        else:
            norm = X.dot(X)
            c = (y - yest) / norm
            self.W += 2 * self.mu * c * X

        return yest
#...............................................................................
if __name__ == "__main__":

    modelorder = 2 * 4 # four w0
    damp = .005
    nx = 600 * 2
    noise = .05 * 2  # * swing
    plot = 0
    seed = 0

    DATA_FILE = "protocolo 7/celso_prot7_config1.mat"

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

    channel = [11] # Oz

    t = np.arange( nx - 1 + 0. ) / 600.
    x = np.empty((2 * modelorder, t.shape[0]))
    for i in range(len(freqs)):
        x[i * 4:(i+1) * 4,:] = processing.generate_references(t.shape[0], freqs[i], Fs, 2).T

    title = "LMS  chirp  filterlen %d  nx %d  noise %.2g  mu %.2g " % (
        modelorder, nx, noise, damp )
    print title
    

#...............................................................................
    msi = processing.MSI(freqs, nx - 1, Fs)
    mflc = MultiFLC( np.zeros(2 * modelorder), mu=damp )
    flc = processing.FLC(freqs, nx - 1, Fs)

    ws = [list() for i in range(len(freqs))]
    flcs = [list() for i in range(len(freqs))]
    msis = [list() for i in range(len(freqs))]
    msiws = [list() for i in range(len(freqs))]
    errs = []
    stds = []
    for i in range(0,100):
        y1 = X[i*Fs:i*Fs + nx, channel].reshape(nx, len(channel)) # protocol first part, 20s x 2

        # f,psd = preprocessing.get_psd(y1[:,0], 3, Fs, config.NFFT)
        # pl.plot(f,psd)
        # pl.show()

        ys = []
        yests = []
        for t in xrange( nx - 1):
            XX = x[:,t:t+1]
            y = y1[t+1]  # predict
            yest = mflc.est( XX, y)
            ys += [y]
            yests += [yest]

        y = np.array(ys).reshape(nx - 1, 1)
        yest = np.array(yests).reshape(nx - 1, 1)

        for j in range(len(freqs)):
            msis[j].append( msi._compute_MSI(y, x[j * 4:(j+1) * 4,:].T) )
            flc._compute_error(y, x[j * 4:(j+1) * 4,:], flc.lms[j])
            flcs[j].append(np.abs(flc.lms[j].W).sum())
            ws[j].append( np.abs(mflc.W[j * 4:(j+1) * 4]).sum() )
            msiws[j].append( msi._compute_MSI(yest, x[j * 4:(j+1) * 4,:].T) )
        
        err = (yest - y)**2
        averr = "av %.2g += %.2g" % (err.mean(), err.std())
        errs += [err.mean()]
        stds += [err.std()]
        # print "LMS yest - y:", averr, "@%d" % i
        # print "LMS weights:", lms.W
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

    fig, ax = pl.subplots( nrows=5 )
    fig.set_size_inches( 12, 8 )
    for i in range(len(freqs)):
        ax[0].plot(ws[i], label=str(freqs[i]))
    ax[0].legend()
    ax[0].grid()
    for i in range(len(freqs)):
        ax[1].plot(flcs[i], label=str(freqs[i]))
    ax[1].legend()
    ax[1].grid()
    for i in range(len(freqs)):
        ax[2].plot(msiws[i], label=str(freqs[i]))
    ax[2].legend()
    ax[2].grid()
    ax[3].plot(errs, label='MSE')
    ax[3].legend()
    ax[3].grid()
    ax[4].plot(stds, label='STD')
    ax[4].legend()
    ax[4].grid()
    pl.show()

