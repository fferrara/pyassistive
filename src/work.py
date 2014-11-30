import numpy as np
import scipy.io as sio
import scipy.signal as signal
import os
from util import config,preprocessing, processing, adapt_filtering, offline, performance
import matplotlib.pyplot as pl

def filter_classify(dataFile):
    # read data
    data = sio.loadmat(os.path.join(config.DATA_PATH, dataFile))
    EEG = data['X'].astype('float32', copy=False)

    channels = np.array(['F3','P7','O1','O2','P8','F4'])
    EEG = EEG[:, np.in1d(np.array(config.SENSORS), channels)]

    # CAR FILTER
    EEG -= EEG.mean(axis=0)
    EEG = np.dot(EEG, preprocessing.CAR(EEG.shape[1]))

    Fs = config.FS
    freqs = [6.4, 6.9, 8]

    # HIGHPASS FILTER
    # Per Emotiv DEVE ESSERE DI ORDINE BASSO PER DARE UN POCHINO DI RITARDO
    Wcritic = np.array([0., 4., 5., 64.])
    b, a = preprocessing._get_fir_filter(Wcritic, config.FS, mask=[0, 1])
    EEG = signal.filtfilt(b, (a,), EEG, axis=0)


    # Filter parameters
    damp = .001
    window = 6
    nx = Fs * window
    modelorder = 50
    harmonics = 2


    channelSignal = np.in1d(channels, ['O2'])
    channelNoise = np.in1d(channels, ['P7'])
    s1 = EEG[:, channelSignal].reshape(len(EEG), 1) 
    n1 = EEG[:, channelNoise].reshape(len(EEG), 1) 
    
    signalWindows = preprocessing.sliding_window(s1, (nx, s1.shape[1]), (Fs, s1.shape[1]))
    noiseWindows = preprocessing.sliding_window(n1, (nx, s1.shape[1]), (Fs, s1.shape[1]))

    msi = processing.MSI(freqs, nx, Fs)
    psda = processing.PSDA(freqs, nx, Fs)

    anc = adapt_filtering.ANC(modelorder, damp)

    befores = np.empty( (len(signalWindows), len(freqs)) )
    afters = np.empty( (len(signalWindows), len(freqs)) )
    signalFiltered = np.empty((len(signalWindows), nx, len(channelSignal)))
    for ii, (signalWin, noiseWin) in enumerate(zip(signalWindows, noiseWindows)):
        ys = []

        #### ONE FILTER
        _, y = anc.filter(noiseWin, signalWin)

        signalFiltered[ii] = y
        # fig, ax = pl.subplots( nrows=3 )
        # ax[2].plot(y)
        # fb, PSDb = preprocessing.get_psd(signalWin[:,0], nx / Fs, Fs, config.NFFT, ax[0])
        # fa, PSDa = preprocessing.get_psd(y[:,0], nx / Fs, Fs, config.NFFT, ax[1])
        # pl.show()
        

        for i, freq in enumerate(freqs):
            x = processing.generate_references(nx, freq, Fs, harmonics)
            # Compute MSI / PSDA indicators
            befores[ii, i] = msi._compute_MSI(x, signalWin)
            # f, PSD = preprocessing.get_psd(win[:,0], nx / Fs, Fs, config.NFFT)
            # befores[ii, i] = processing.PSDA.compute_SNR(f, PSD, freq, Fs)
            afters[ii, i] = msi._compute_MSI(x, y)
            # f, PSD = preprocessing.get_psd(y[:,0], nx / Fs, Fs, config.NFFT)
            # afters[ii, i] = processing.PSDA.compute_SNR(f, PSD, freq, Fs)
        np.round(befores, 3)
        np.round(afters, 3)
        # print afters[ii, :], befores[ii, :]
        # pl.show()

    label = offline.make_label_matrix(EEG.shape[0], config.RECORDING_PERIOD, window, config.FS, len(freqs))
    # o1 = offline.offline_classify(signalWindows, freqs, msi)
    # o = offline.offline_classify(signalFiltered, freqs, msi)
    # print 100 * performance.get_accuracy(label, o1), 100 * performance.get_accuracy(label, o)

    fig, ax = pl.subplots( nrows=2 )
    fig.set_size_inches( 12, 8 )
    plots = ax[0].plot(befores)
    ax[0].plot(label * np.max(befores))
    ax[0].legend(plots, map(str, freqs), loc=1)
    start, end = ax[0].get_xlim()
    ax[0].xaxis.set_ticks(np.arange(start, end, 5))
    ax[0].grid()
    plots = ax[1].plot(afters)
    ax[1].plot(label * np.max(afters))
    ax[1].legend(plots, map(str, freqs), loc=1)
    ax[1].xaxis.set_ticks(np.arange(start, end, 5))
    ax[1].grid()
    pl.show()


if __name__ == '__main__':

    DATA_FILE = "emotiv_original_flavio2_low.mat"
    filter_classify(DATA_FILE)

