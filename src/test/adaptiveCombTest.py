""" lmsTest.py: a simple adaptive notch filter """

import numpy as np
import scipy.io as sio
import scipy.signal as signal
import os
from ..util import config,preprocessing, processing
from ..util import adapt_filtering
import matplotlib.pyplot as pl

def blit(t):
    """
    Return a periodic band-limited impulse train (Dirac comb) with
    period 2*pi (same phase as a cos)
     
    Examples
    --------
    >>> t = linspace(0, 1, num = 1000, endpoint = False)
    >>> f = 5.4321 # Hz
    >>> plot(blit(2 * pi * f * t))
     
    References
    ----------
    http://www.music.mcgill.ca/~gary/307/week5/bandlimited.html
    """
    t = np.asarray(t)
     
    if np.abs((t[-1]-t[-2]) - (t[1]-t[0])) > .0000001:
        raise ValueError("Sampling frequency must be constant")
     
    if t.dtype.char in ['fFdD']:
        ytype = t.dtype.char
    else:
        ytype = 'd'
    y = np.zeros(t.shape, ytype)
     
    # Get sampling frequency from timebase
    fs = 1 / (t[1] - t[0])
     
    # Sum all multiple sine waves up to the Nyquist frequency
    N = int(fs * np.pi) + 1
    for h in range(1, N):
        y += np.cos(h * t)
    y /= N
     
    # h = arange(1, int(fs * pi) + 1)
    # phase = outer(t, h)
    # y = 2 / pi * cos(phase)
    # y = sum(y, axis=1)
     
    return y
#...............................................................................
if __name__ == "__main__":
    import sys

    damp = .005
    Fs = 600
    seconds = 3
    nx = Fs * seconds
    freq = 5  # chirp
    freq2 = 5.6*2
    components = np.array([i for i in [5, 7, 10, 12]])  # components
    plot = 1
    seed = 0

    t = np.arange(0., seconds + 0., 1. / Fs)

    # impulse train order L. L interest signal period
    period = Fs / freq
    # x = np.zeros((t.shape[0]))
    x = blit(2 * np.pi * freq * t)


    signal = np.sum([np.cos(2 * np.pi * i * t) for i in components],axis=0)
    noise = np.random.randn(nx)
    d = signal + noise


    title = "Adaptive notch  filterlen %d  nx %d  mu %.2g " % (
        period, nx, damp )
    print title

#...............................................................................
    ys = []
    yests = []
    lms = adapt_filtering.LMS( np.zeros((period)), mu=damp )
    for t in xrange( nx - period):
        X = x[t:t+period][::-1]
        y = d[t+period]  # predict
        yest = lms.est( X, y)
        ys += [y - yest]
        yests += [yest]

    y = np.array(ys)
    yest = np.array(yests)

    # MSI TEST
    freqs = [5, 7, 10, 12]
    for freq in freqs:
        ref = processing.generate_references(nx-period, freq, Fs)
        msi = processing.MSI([freq], nx - period, Fs)
        print 'filtered', freq, msi._compute_MSI(ref, yest.reshape(nx - period, 1))
        print 'original ', freq, msi._compute_MSI(ref, d[0:nx - period].reshape(nx - period, 1))

    err = (yest - y)**2
    averr = "av %.2g += %.2g" % (err.mean(), err.std())
    # print "LMS yest - y:", averr
    # print "LMS weights:", lms.W
    if plot == 2:
        fig, ax = pl.subplots( nrows=2 )
        fig.set_size_inches( 12, 8 )
        fig.suptitle( title, fontsize=12 )
        ax[0].plot( y, color="orangered", label="y" )
        ax[0].plot( yest, label="yest" )
        ax[0].legend()
        ax[1].plot( err, label=averr )
        ax[1].legend()
        pl.show()
    if plot == 1:
        Fxx = np.arange(50)

        Ps = np.fft.fft(signal, Fs) / 300 # fft computing and normalization
        Ps = Ps[range(50)]
        Pd = np.fft.fft(d, Fs) / 300
        Pd = Pd[range(50)]
        Py = np.fft.fft(yest, Fs) / 300
        Py = Py[range(50)]
        Pe = np.fft.fft(y, Fs) / 300
        Pe = Pe[range(50)]
        
        fig, ax = pl.subplots( nrows=4 )
        fig.set_size_inches( 12, 8 )
        fig.suptitle( title, fontsize=12 )
        ax[0].plot( Fxx, np.abs(Ps), label="original signal" )
        ax[0].legend()
        ax[1].plot( Fxx, np.abs(Pd), color="orangered", label="primary input" )
        ax[1].legend()
        ax[2].plot( Fxx, np.abs(Py), label="notch output" )
        ax[2].legend()
        ax[3].plot( Fxx, np.abs(Pe), label="bandpass output" )
        ax[3].legend()
        pl.show()

