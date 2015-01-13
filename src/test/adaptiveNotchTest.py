""" lmsTest.py: a simple adaptive notch filter """


import numpy as np
import scipy.io as sio
import scipy.signal as signal
import os
from ..util import config,preprocessing, featex
from ..util import adapt_filtering
import matplotlib.pyplot as pl

#...............................................................................
if __name__ == "__main__":
    import sys

    damp = .001
    Fs = 600
    nx = 600 *3
    freq = 5  # chirp
    freq2 = freq*2
    components = [i for i in [3, 4, 5,6,7,10,12]]  # components
    As = [1, 1, 0.5, 0.5, 0.5, 0.5, 0.5]
    plot = 2
    seed = 0

    t = np.arange( nx + 0.) / Fs

    ######## Choice: 2-weights per reference or TDL

    # 2-weights sin & cos
    modelorder = 4
    x = np.empty((t.shape[0],modelorder))
    C = 1
    x[:,0] = C*np.cos(2 * np.pi * freq * t)
    x[:,1] = C*np.sin(2 * np.pi * freq * t)
    x[:,2] = C*np.cos(2 * np.pi * freq2 * t)
    x[:,3] = C*np.sin(2 * np.pi * freq2 * t)

    # Multiple weights TDL
    # modelorder = 10
    # x = np.empty((t.shape[0]))
    # C = 0.2
    # x = C*np.cos(2 * np.pi * freq * t) + C*np.cos(2 * np.pi * freq2 * t)

    #####################

    signal = np.sum([As[i] * np.cos(2 * np.pi * components[i] * t) for i in range(len(components))],axis=0)
    noise = np.random.randn(nx)
    d = signal + noise
    d = d.reshape(nx ,1)


    title = "Adaptive notch  filterlen %d  nx %d  mu %.2g " % (
        modelorder, nx, damp )
    print title

#...............................................................................
    anf = adapt_filtering.ANF(modelorder, damp)
    yest, y = anf.filter_synch(x, d)
    

    # MSI TEST
    freqs = [4, 5]
    for freq in freqs:
        ref = featex.generate_references(nx, freq, Fs)
        msi = featex.MSI([freq], nx, Fs)
        print 'filtered ', freq, msi._compute_MSI(ref, yest.reshape(nx, 1))
        print 'original ', freq, msi._compute_MSI(ref, d.reshape(nx, 1))

    err = (yest - d)**2
    averr = "av %.2g += %.2g" % (err.mean(), err.std())
    # print "LMS yest - y:", averr
    # print "LMS weights:", lms.W
    if plot == 2:
        fig, ax = pl.subplots( nrows=2 )
        fig.set_size_inches( 12, 8 )
        fig.suptitle( title, fontsize=12 )
        ax[0].plot( x[:,0] + x[:,2], color="orangered", label="y" )
        ax[0].plot( yest, label="yest" )
        ax[0].legend()
        ax[1].plot( err, label=averr )
        ax[1].legend()
        pl.show()
    if plot == 1:
        Fxx = np.arange(20)

        Ps = np.fft.fft(signal, Fs) / 300 # fft computing and normalization
        Ps = Ps[range(20)]
        Pd = np.fft.fft(d[:,0], Fs) / 300
        Pd = Pd[range(20)]
        Py = np.fft.fft(yest[:,0], Fs) / 300
        Py = Py[range(20)]
        Pe = np.fft.fft(y[:,0], Fs) / 300
        Pe = Pe[range(20)]
        
        fig, ax = pl.subplots( nrows=4 )
        fig.set_size_inches( 12, 8 )
        fig.suptitle( title, fontsize=12 )
        ax[0].plot( Fxx, np.abs(Ps), label="original signal" )
        ax[1].plot( Fxx, np.abs(Pd), color="orangered", label="primary input" )
        ax[2].plot( Fxx, np.abs(Py), label="notch output" )
        ax[3].plot( Fxx, np.abs(Pe), label="bandpass output" )
        for i in range(len(ax)):
            ax[i].legend()
            ax[i].grid()
        pl.show()

