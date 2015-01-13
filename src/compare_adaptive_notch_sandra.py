import numpy as np
import scipy.io as sio
import scipy.signal as signal
import os
from util import config,preprocessing, featex, adapt_filtering
import matplotlib.pyplot as pl

def qtest(obs):
        supp = np.array(obs)
        supp.sort()
        return (supp[-1] - supp[-2]) / (supp[-1] - supp[0])

def stdtest(obs):
    m = obs.mean()
    supp = np.array(obs)
    supp.sort()
    if supp[-1] > m + 5 * supp[0:-1].std():
        return True
    return False

def filter_classify(dataFile):
    # read data
    data = sio.loadmat(os.path.join(config.DATA_PATH, dataFile))
    EEG = data['data']

    Fs = 600
    freqs = [5.6, 6.4, 6.9, 8]
    # freqs = [8]

    # CAR FILTER & PASSBAND FILTER
    Wcritic = np.array([0., 4., 5., 49., 50., 300.])
    b, a = preprocessing._get_fir_filter(Wcritic, Fs, 851)
    EEG = signal.fftconvolve(EEG, b[:, np.newaxis], mode='valid')

    EEG -= EEG.mean(axis=0)
    EEG = np.dot(EEG, preprocessing.CAR(EEG.shape[1]))

    # Filter parameters
    damp = .005
    nx = 600 * 2
    modelorder = 2
    harmonics = 2
    singleFilter = 0

    channel = [9, 10, 11]
    EEG = EEG[0:110*Fs, channel].reshape(110*Fs, len(channel)) # First part of protocol
    protocolPart1 = preprocessing.sliding_window(EEG, (nx, EEG.shape[1]), (Fs, EEG.shape[1]))

    msi = featex.MSI(freqs, nx, Fs)
    # psda = processing.PSDA(freqs, nx, Fs)

    anfs = [adapt_filtering.ANF(harmonics * modelorder, damp) for f in freqs]
    anf = adapt_filtering.ANF(harmonics * modelorder, damp)

    befores = np.zeros( (len(protocolPart1), len(freqs)) )
    afters = np.zeros( (len(protocolPart1), len(freqs)) )
    for ii, win in enumerate(protocolPart1):
        ys = []

        #### ONE FILTER
        if singleFilter:
            x = np.zeros((nx, harmonics * modelorder))
            for i in range(len(freqs)):
                x[:,i * 4:(i+1) * 4] = featex.generate_references(nx, freqs[i], Fs, 2)

            y = np.zeros((nx, len(channel)))
            for c in range(len(channel)):
                y[:, c], _ = anf.filter_synch(x, win[:, c])

            # fig, ax = pl.subplots( nrows=2 )
            # fb, PSDb = preprocessing.get_psd(win[:,0], nx / Fs, Fs, config.NFFT, ax[0])
            # fa, PSDa = preprocessing.get_psd(y[:,0], nx / Fs, Fs, config.NFFT, ax[1])
            # pl.show()

            for i, freq in enumerate(freqs):
                x = featex.generate_references(nx, freq, Fs, harmonics)
                # Compute MSI / PSDA indicators
                befores[ii, i] = msi._compute_MSI(x, win)
                # f, PSD = preprocessing.get_psd(win[:,0], nx / Fs, Fs, config.NFFT)
                # befores[ii, i] = processing.PSDA.compute_SNR(f, PSD, freq, Fs)
                afters[ii, i] = msi._compute_MSI(x, y)
                # f, PSD = preprocessing.get_psd(y[:,0], nx / Fs, Fs, config.NFFT)
                # afters[ii, i] = processing.PSDA.compute_SNR(f, PSD, freq, Fs)
            np.round(befores, 3)
            np.round(afters, 3)
        else:
            #####################
            #### MULTIPLE FILTERS
            for i, freq in enumerate(freqs):
                x = featex.generate_references(nx, freq, Fs, harmonics)

                # Compute MSI / PSDA indicators
                befores[ii, i] = msi._compute_MSI(x, win)
                # f, PSD = preprocessing.get_psd(win[:,0], nx / Fs, Fs, config.NFFT)
                # befores[ii, i] = processing.PSDA.compute_SNR(f, PSD, freq, Fs)

                # Filter
                y = np.zeros((nx, len(channel)))
                for c in range(len(channel)):
                    y[:, c], _ = anfs[i].filter_synch(x, win[:, c])
                ys.append(y)

            y = np.sum(ys, axis=0)
            # fig, ax = pl.subplots( nrows=2 )
            # fb, PSDb = preprocessing.get_psd(win[:,0], nx / Fs, Fs, config.NFFT, ax[0])
            # fa, PSDa = preprocessing.get_psd(y[:,0], nx / Fs, Fs, config.NFFT, ax[1])
            # pl.show()

            for i, freq in enumerate(freqs):
                x = featex.generate_references(nx, freq, Fs, harmonics)
                # Compute MSI / PSDA indicators
                afters[ii, i] = msi._compute_MSI(x, y)
                # f, PSD = preprocessing.get_psd(y[:,0], nx / Fs, Fs, config.NFFT)
                # afters[ii, i] = processing.PSDA.compute_SNR(f, PSD, freq, Fs)
            np.round(befores, 3)
            np.round(afters, 3)
            ################

    c = 0
    for i in range(len(befores)):
        if stdtest(befores[i]): # 70%
            print np.argmax(befores[i])
            c += 1
    print 'count: %d' % c

    fig, ax = pl.subplots( nrows=2 )
    fig.set_size_inches( 12, 8 )
    plots = ax[0].plot(befores)
    ax[0].legend(plots, map(str, freqs), loc=1)
    start, end = ax[0].get_xlim()
    ax[0].xaxis.set_ticks(np.arange(start, end, 5))
    ax[0].grid()
    plots = ax[1].plot(afters)
    ax[1].legend(plots, map(str, freqs), loc=1)
    ax[1].xaxis.set_ticks(np.arange(start, end, 5))
    ax[1].grid()
    pl.show()


if __name__ == '__main__':

    DATA_FILE = "protocolo 6/gabriel_prot6_repouso.mat"
    filter_classify(DATA_FILE)

