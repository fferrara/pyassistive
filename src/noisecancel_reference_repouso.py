""" lms.py: a simple python class for Least mean squares adaptive filter """

from __future__ import division
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import os
from util import config,preprocessing, processing, offline, performance
import matplotlib.pyplot as pl

class LMS(object):
    def __init__(self, W, mu):
        self.W = np.squeeze( getattr( W, "A", W ))  # matrix -> array
        # Sufficient for stability: 0 < mu < 1 / tr(R)
        # R = E[x * x.T]  = np.corrcoef(x.T)
        self.mu = mu

    def est(self, X, y):
        X = np.squeeze( getattr( X, "A", X ))
        yest = self.W.dot(X)
        c = (y - yest) 

        self.W += 2 * self.mu * c * X / X.dot(X)
        return y - yest
#...............................................................................
if __name__ == "__main__":

    DATA_FILE = "protocolo 6/bruno_prot6_8hz.mat"
    DATA_FILE_2 = "protocolo 6/bruno_prot6_repouso.mat"

    # read data
    data = sio.loadmat(os.path.join(config.DATA_PATH, DATA_FILE))
    X = data['data']
    data2 = sio.loadmat(os.path.join(config.DATA_PATH, DATA_FILE_2))
    Y = data2['data']

    Fs = 600
    WINLENGTH = 2
    freqs = [5.6, 6.4, 6.9, 8]
    freq = 8
    plot = 0
    fOrder = 10 # delay line
    damp = .005
    nx = Fs * WINLENGTH

    # CAR FILTER & PASSBAND FILTER
    X -= X.mean(axis=0)
    # X = np.dot(X, preprocessing.CAR(X.shape[1]))
    Y -= Y.mean(axis=0)

    b, a = preprocessing._get_fir_filter(config.WCRITIC, Fs, 251)
    X = signal.lfilter(b, (a,), X, axis=0)
    Y = signal.lfilter(b, (a,), Y, axis=0)

    channelS = 11 # Oz
    # channelN = 0 # POz
    s1 = X[0:150*Fs, channelS:channelS+1]# - X[0:160*Fs, channelN:channelN+1] # protocol first part, 20s x 2
    n1 = Y[0:150*Fs, channelS:channelS+1]
    # l1 = offline.make_label_matrix(s1.shape[0], period=20, window=WINLENGTH, fs=Fs, classes=4)

#...............................................................................
    # title = "LMS  noise cancellation  filterlen %d  nx %d  noise %.2g  mu %.2g " % (
        # fOrder, nx, channelN, damp )
    # print title

    signalWindows = preprocessing.sliding_window(s1, (nx, s1.shape[1]), (Fs, s1.shape[1]))
    noiseWindows = preprocessing.sliding_window(n1, (nx, s1.shape[1]), (Fs, s1.shape[1]))

    classifier = processing.MSI(freqs, nx - fOrder, Fs)

    outMonopolar = np.zeros(signalWindows.shape[0])
    outFiltered = np.zeros(signalWindows.shape[0])
    SNRsignal =  np.zeros(signalWindows.shape[0])
    SNRnoise =  np.zeros(signalWindows.shape[0])
    l1 = np.zeros(signalWindows.shape[0])
    l1.fill(3)
    for ii, (signalWin, noiseWin) in enumerate(zip(signalWindows, noiseWindows)):
        ys = []
        epss = []

        Fxx, Pxx = preprocessing.get_psd(signalWin[:,0], WINLENGTH, Fs, config.NFFT)
        Fyy, Pyy = preprocessing.get_psd(noiseWin[:,0], WINLENGTH, Fs, config.NFFT)
        if plot == 1:
            pl.plot(Fyy[Fyy < 50],Pyy[Fyy < 50])
            pl.axvline(x = freq, linewidth=2, color='r')
            pl.axvline(x = freq * 2,linewidth=2, color='r')
            pl.axvspan(freq - 3, freq + 3, facecolor='r', alpha=.3)
            pl.axvspan(freq * 2 - 3, freq * 2 + 3, facecolor='r', alpha=.3)
            pl.show()

        SNRsignal[ii] = processing.PSDA.compute_SNR(Fxx, Pxx, freq, Fs)
        SNRnoise[ii] = processing.PSDA.compute_SNR(Fyy, Pyy, freq, Fs)
        
        lms = LMS( np.zeros(fOrder), mu=damp )
        for t in xrange( nx - fOrder):
            XX = noiseWin[t:t+fOrder, :]
            y = signalWin[t+fOrder, 0]  # predict
            eps = lms.est( XX, y)
            ys += [y]
            epss += [eps]
            
        y = np.array(ys)
        sCap = np.array(epss)
        
        err = (epss - y)**2
        averr = "av %.2g += %.2g" % (err.mean(), err.std())
        # print "LMS MSE:", averr
        # print "LMS weights:", lms.W

        outMonopolar[ii] = freqs.index(classifier.perform(np.atleast_2d(y).T))
        outFiltered[ii] = freqs.index(classifier.perform(np.atleast_2d(sCap).T))
        # print outMonopolar[ii], outFiltered[ii], l1[ii]
        if plot == 2:
            fig, ax = pl.subplots( nrows=5 )
            fig.set_size_inches( 12, 8 )
            ax[0].plot( y, color="orangered", label="y" )
            ax[0].plot( epss, label="yest" )
            ax[0].legend()
            ax[1].plot( err, label=averr )
            ax[1].legend()
            
            f,psd = preprocessing.get_psd(signalWin[:,0], WINLENGTH, Fs, config.NFFT, ax[2])
            ax[2].legend()

            f,psd = preprocessing.get_psd(noiseWin[:,0], WINLENGTH, Fs, config.NFFT, ax[3])
            ax[3].legend()

            f,psd = preprocessing.get_psd(epss, WINLENGTH, Fs, config.NFFT, ax[4])
            ax[4].legend()
            pl.show()

    print 'ORDER %d' % (fOrder)
    print 'SNR signal: %f, SNR noise: %f, distorsion: %f, output SNR : %f' % (
        SNRsignal.mean(), SNRnoise.mean(), SNRnoise.mean() / SNRsignal.mean(), 1 / SNRnoise.mean())
    print performance.get_accuracy(l1, outMonopolar)
    print performance.get_accuracy(l1, outFiltered)
