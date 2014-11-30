import numpy as np
import scipy.io as sio
import scipy.signal as signal
import os
from util import config,preprocessing, processing, adapt_filtering, offline, performance
import matplotlib.pyplot as pl

def qtest(obs):
    supp = np.array(obs)
    supp.sort()
    return (supp[2] - supp[1]) / (supp[2] - supp[0])

def filter_classify(dataFile):
    # read data
    data = sio.loadmat(os.path.join(config.DATA_PATH, dataFile))
    EEG = data['X'].astype('float32', copy=False)

    CARchannels = np.array(['F3','P7','O1','O2','P8','F4'])
    EEG = EEG[:, np.in1d(np.array(config.SENSORS), CARchannels)]

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
    damp = .005
    window = 10
    nx = Fs * window
    modelorder = 2
    harmonics = 2
    singleFilter = 0


    CCAchannels = np.array(['O1','O2'])
    EEG = EEG[:, np.in1d(CARchannels, CCAchannels)].reshape(len(EEG), len(CCAchannels)) 
    msi = processing.MSI(list(freqs), nx, Fs)
    psda = processing.PSDA(freqs, nx, Fs)

    anfs = [adapt_filtering.ANF(harmonics * modelorder, damp) for f in freqs]
    anf = adapt_filtering.ANF(harmonics * modelorder, damp)

    # windowsInPeriod = window*2 <= config.RECORDING_PERIOD and 2 or 1
    windowsInPeriod = config.RECORDING_PERIOD - window + 1
    befores = np.empty( (config.RECORDING_ITERATIONS * len(freqs) * windowsInPeriod, len(freqs)) )
    afters = np.empty( (config.RECORDING_ITERATIONS * len(freqs) * windowsInPeriod, len(freqs)) )
    protocolFiltered = np.empty((config.RECORDING_ITERATIONS * len(freqs) * windowsInPeriod, nx, len(CCAchannels)))

    # label = offline.make_label_segments(config.RECORDING_ITERATIONS, config.RECORDING_PERIOD, window, len(freqs))
    # W = offline.extract_segment_start(EEG, window, config.RECORDING_ITERATIONS, config.RECORDING_PERIOD, Fs)
    label = offline.make_label_windows(config.RECORDING_ITERATIONS, config.RECORDING_PERIOD, window, len(freqs))
    W = offline.extract_windowed_segment(EEG, window, config.RECORDING_ITERATIONS, config.RECORDING_PERIOD, Fs)

    for wi, win in enumerate(W):
        ys = []

        #### ONE FILTER
        if singleFilter:
            x = np.zeros((nx, harmonics * modelorder))
            for i in range(len(freqs)):
                x[:,i * 4:(i+1) * 4] = processing.generate_references(nx, freqs[i], Fs, 2)

            y = np.zeros((nx, len(CCAchannels)))
            for c in range(len(CCAchannels)):
                y[:, c], _ = anf.filter_synch(x, win[:, c])

            protocolFiltered[wi] = y
            # fig, ax = pl.subplots( nrows=2 )
            # fb, PSDb = preprocessing.get_psd(win[:,0], nx / Fs, Fs, config.NFFT, ax[0])
            # fa, PSDa = preprocessing.get_psd(y[:,0], nx / Fs, Fs, config.NFFT, ax[1])
            

            for i, freq in enumerate(freqs):
                x = processing.generate_references(nx, freq, Fs, harmonics)
                # Compute MSI / PSDA indicators
                befores[wi, i] = msi._compute_MSI(x, win)
                # f, PSD = preprocessing.get_psd(win[:,0], nx / Fs, Fs, config.NFFT)
                # befores[ii, i] = processing.PSDA.compute_SNR(f, PSD, freq, Fs)
                afters[wi, i] = msi._compute_MSI(x, y)
                # f, PSD = preprocessing.get_psd(y[:,0], nx / Fs, Fs, config.NFFT)
                # afters[ii, i] = processing.PSDA.compute_SNR(f, PSD, freq, Fs)
            np.round(befores, 3)
            np.round(afters, 3)
            # print afters[ii, :], befores[ii, :]
            # pl.show()
        else:
            #####################
            #### MULTIPLE FILTERS
            temp = []

            for i, freq in enumerate(freqs):
                x = processing.generate_references(nx, freq, Fs, harmonics)

                # Compute MSI / PSDA indicators
                befores[wi, i] = msi._compute_MSI(x, win)
                # befores[ii, i] = msi._compute_max_corr(x, win)
                # f, PSD = preprocessing.get_psd(win[:,1], nx / Fs, Fs, config.NFFT)
                # befores[wi, i] = processing.PSDA.compute_SNR(f, PSD, freq, Fs)

                # Filter
                y = np.zeros((nx, len(CCAchannels)))
                for c in range(len(CCAchannels)):
                    y[:, c], _ = anfs[i].filter_synch(x, win[:, c])
                ys.append(y)

                
            #     ics1 = processing.generate_references(nx, freqs[0], Fs, harmonics)
            #     ics2 = processing.generate_references(nx, freqs[1], Fs, harmonics)
            #     m1 = msi._compute_max_corr(ics1, y)
            #     m2 = msi._compute_max_corr(ics2, y)
            #     if freq == freqs[0]:
            #         temp.append(m1)
            #     else:
            #         temp.append(m2)


            # print befores[ii, :]
            # print temp
            y = np.sum(ys, axis=0)
            protocolFiltered[wi] = y
            # fig, ax = pl.subplots( nrows=2 )
            # fb, PSDb = preprocessing.get_psd(win[:,1], nx / Fs, Fs, config.NFFT, ax[0])
            # fa, PSDa = preprocessing.get_psd(y[:,0], nx / Fs, Fs, config.NFFT, ax[1])
            # pl.show()

            for i, freq in enumerate(freqs):
                x = processing.generate_references(nx, freq, Fs, harmonics)
                # Compute MSI / PSDA indicators
                afters[wi, i] = msi._compute_MSI(x, y)
                # f, PSD = preprocessing.get_psd(y[:,1], nx / Fs, Fs, config.NFFT)
                # afters[wi, i] = processing.PSDA.compute_SNR(f, PSD, freq, Fs)
            np.round(befores, 3)
            np.round(afters, 3)
            ################

    o1 = offline.offline_classify(W, freqs, msi)
    cm = performance.get_confusion_matrix(label, o1, len(freqs))
    print performance.get_accuracy(cm)

    for i in range(len(befores)):
        if qtest(befores[i]) > 0.9: # 70%
            print i, qtest(befores[i])

    fig, ax = pl.subplots( nrows=2 )
    fig.suptitle(dataFile)
    fig.set_size_inches( 12, 8 )
    plots = ax[0].plot(befores, '-o')
    ax[0].plot(label * np.max(befores))
    ax[0].legend(plots, [str(f) for f in freqs], loc=1)
    start, end = ax[0].get_xlim()
    ax[0].xaxis.set_ticks(np.arange(start, end, 1))
    ax[0].grid()
    plots = ax[1].plot(afters, '-o')
    # ax[1].plot(label * np.max(afters))
    ax[1].legend(plots,[str(f) for f in freqs], loc=1)
    ax[1].xaxis.set_ticks(np.arange(start, end, 1))
    ax[1].grid()
    pl.show()


if __name__ == '__main__':
    DATA_FILE = "emotiv_original_alan1_low.mat"
    filter_classify(DATA_FILE)

