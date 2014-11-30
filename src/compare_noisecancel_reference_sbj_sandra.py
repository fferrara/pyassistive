""" lms.py: a simple python class for Least mean squares adaptive filter """

from __future__ import division
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import os
from util import config,preprocessing, processing, offline, performance
import matplotlib.pyplot as pl
import scipy.linalg as LA

class LMS(object):
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
        return y - yest

def filter_and_accuracy(dataFile, signalChannel, noiseChannel):
    # read data
    data = sio.loadmat(os.path.join(config.DATA_PATH, dataFile))
    data = data['data']

    # CAR FILTER & PASSBAND FILTER
    data -= data.mean(axis=0)
    # X = np.dot(X, preprocessing.CAR(X.shape[1]))
    b, a = preprocessing._get_fir_filter(config.WCRITIC, Fs, 251)
    data = signal.lfilter(b, (a,), data, axis=0)

    s1 = data[0:160*Fs, np.in1d(SANDRA_SENSORS, signalChannel)]# - X[0:160*Fs, channelN:channelN+1] # protocol first part, 20s x 2
    n1 = data[0:160*Fs, np.in1d(SANDRA_SENSORS, noiseChannel)]

    l1 = offline.make_label_matrix(s1.shape[0], period=20, window=WINLENGTH, fs=Fs, classes=4)

    signalWindows = preprocessing.sliding_window(s1, (nx, s1.shape[1]), (Fs, s1.shape[1]))
    noiseWindows = preprocessing.sliding_window(n1, (nx, s1.shape[1]), (Fs, s1.shape[1]))

    # classifier = processing.PSDA(freqs, nx - fOrder, Fs)
    classifier = processing.MSI(freqs, nx - fOrder, Fs)

    outBipolar = np.zeros(signalWindows.shape[0])
    outFiltered = np.zeros(signalWindows.shape[0])
    SNRsignal =  np.zeros(signalWindows.shape[0])
    SNRnoise =  np.zeros(signalWindows.shape[0])
    for ii, (signalWin, noiseWin) in enumerate(zip(signalWindows, noiseWindows)):
        ys = []
        epss = []

        bipolar = signalWin[0:nx - fOrder] - noiseWin[0:nx - fOrder]

        Fxx, Pxx = preprocessing.get_psd(signalWin[:,0], WINLENGTH, Fs, config.NFFT)
        Fyy, Pyy = preprocessing.get_psd(noiseWin[:,0], WINLENGTH, Fs, config.NFFT)
        if plot == 1:
            pl.plot(Fyy[Fyy < 50],Pyy[Fyy < 50])
            pl.axvline(x = freqs[l1[ii]], linewidth=2, color='r')
            pl.axvline(x = freqs[l1[ii]] * 2,linewidth=2, color='r')
            pl.axvspan(freqs[l1[ii]] - 3, freqs[l1[ii]] + 3, facecolor='r', alpha=.3)
            pl.axvspan(freqs[l1[ii]] * 2 - 3, freqs[l1[ii]] * 2 + 3, facecolor='r', alpha=.3)
            pl.show()

        SNRsignal[ii] = processing.PSDA.compute_SNR(Fxx, Pxx, freqs[l1[ii]], Fs)
        SNRnoise[ii] = processing.PSDA.compute_SNR(Fyy, Pyy, freqs[l1[ii]], Fs)
        
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

        outBipolar[ii] = freqs.index(classifier.perform(bipolar))
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
            f,psd = preprocessing.get_psd(signalWin[:,0], 3, Fs, config.NFFT, ax[2])
            ax[2].legend()

            f,psd = preprocessing.get_psd(noiseWin[:,0], 3, Fs, config.NFFT, ax[3])
            ax[3].legend()
            ax[3].axvline(x = freqs[l1[ii]], linewidth=2, color='r')
            ax[3].axvline(x = freqs[l1[ii]] * 2,linewidth=2, color='r')
            ax[3].axvspan(freqs[l1[ii]] - 3, freqs[l1[ii]] + 3, facecolor='r', alpha=.3)
            ax[3].axvspan(freqs[l1[ii]] * 2 - 3, freqs[l1[ii]] * 2 + 3, facecolor='r', alpha=.3)
            f,psd = preprocessing.get_psd(epss, 3, Fs, config.NFFT, ax[4])
            ax[4].legend()
            ax[4].axvline(x = freqs[l1[ii]], linewidth=2, color='r')
            ax[4].axvline(x = freqs[l1[ii]] * 2,linewidth=2, color='r')
            ax[4].axvspan(freqs[l1[ii]] - 3, freqs[l1[ii]] + 3, facecolor='r', alpha=.3)
            ax[4].axvspan(freqs[l1[ii]] * 2 - 3, freqs[l1[ii]] * 2 + 3, facecolor='r', alpha=.3)

            pl.show()

    # print "LMS  noise cancellation  filterlen %d  nx %d  noise %s  mu %.2g " % (
        # fOrder, nx, noiseChannel, damp )
    # print 'SNR signal: %f, SNR noise: %f, distorsion: %f, output SNR : %f' % (
        # SNRsignal.mean(), SNRnoise.mean(), SNRnoise.mean() / SNRsignal.mean(), np.mean(1 / SNRnoise))
    # print performance.get_accuracy(l1, outBipolar), performance.get_accuracy(l1, outFiltered)
    return performance.get_accuracy(l1, outBipolar), performance.get_accuracy(l1, outFiltered), SNRnoise.mean() / SNRsignal.mean()

#...............................................................................
if __name__ == "__main__":
    SANDRA_SENSORS = ['P7', 'PO7', 'P5', 'PO3', 'POz', 'PO4', 'P6', 'PO8', 'P8', 'O1', 'O2', 'Oz']

    Fs = 600
    WINLENGTH = 3
    freqs = [5.6, 6.4, 6.9, 8]
    plot = 0
    fOrder = 120 # delay line
    damp = .005
    nx = Fs * WINLENGTH
    combs = ['Oz-POz', 'Oz-PO3', 'Oz-PO4', 'O1-POz', 'O2-POz', 'Oz-P7', 'O2-P7']
  
    results = np.zeros((len(config.SUBJECTS),len(combs),3))

    refs = [processing.generate_references(nx, f, Fs) for f in freqs]

    
    for ii, name in enumerate(config.SUBJECTS):
        DATA_FILE = "protocolo 7/%s_prot7_config1.mat" % name
        bipolars = np.zeros((len(combs), len(config.SUBJECTS)))

        for j, comb in enumerate(combs):
            channelS = comb.split('-')[0]
            channelN = comb.split('-')[1]

            results[ii, j, :] = np.array(filter_and_accuracy(DATA_FILE, channelS, channelN))
        print '#'

    filename = "aggregated_noise_cancellation_MSI_3s_len10_projected.txt"

    results = results.reshape(len(config.SUBJECTS)*len(combs), 3)
    np.savetxt(filename, results, fmt="%.2f", delimiter=',')

