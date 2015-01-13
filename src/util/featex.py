# -*- coding: utf-8 -*-
"""
Metodi vari per il processing
"""
import numpy as np
import scipy.linalg as LA
import config
from .preprocessing import get_psd


def generate_references(samples, frequency, fs, harmonics=2):
    vector = np.zeros([samples, 2 * harmonics])
    time = np.arange(samples, dtype=float) / fs
    for i in xrange(harmonics):
        vector[:, 2 * i] = np.sin((i + 1) * 2 * np.pi * frequency * time)
        vector[:, 2 * i + 1] = np.cos((i + 1) * 2 * np.pi * frequency * time)

    return vector


class FeatureExtractor(object):
    def __init__(self, frequencies, samples, fs):
        self.frequencies = frequencies
        self.window = samples / fs
        self.fs = fs

    def perform(self, data):
        """
        Almost every method compute features and classify according to max feature
        """
        raise NotImplementedError

    def compute_features(self, data):
        """
        Each method defines its own names
        """
        raise NotImplementedError


class PSDA(FeatureExtractor):
    """
    Normale PSDA. Calcola il SNR tramite la PSD e tira fuori la classe più probabile
    """

    def __init__(self, frequencies, samples, fs):
        FeatureExtractor.__init__(self, frequencies, samples, fs)

    def compute_features(self, data, resolution=20):
        if data.ndim > 1:
            data = data[:, 0]

        try:
            f, PSD = get_psd(data, self.window, self.fs, config.NFFT)
        except ValueError as v:
            print str(v)
            print type(self).__name__ + ": Error in Fourier trasform. Have you set proper Fs in config.py?"
            exit(1)

        ro = []
        for frequency in self.frequencies:
            ro.append(10 * np.log10(self.__class__.compute_SNR(f, PSD, frequency, self.fs, resolution)))
        return ro

    @staticmethod
    def compute_SNR(Fxx, Pxx, frequency, fs, resolution):
        """
        Compute SNR based on PSD. Inspired by 
            A Practical VEP-Based Brain–Computer Interface
            Yijun Wang

        Modified in order to consider power decrease for growing frequencies
        """
        harmonics = 2
        offset = resolution * fs / config.NFFT
        SNRs = np.zeros(harmonics)
        # neighb = np.zeros(4)
        for i in range(harmonics):
            f = frequency * (i + 1)

            # neighb[0] = (np.abs(Fxx-(f-offset))).argmin()
            # neighb[1] = (np.abs(Fxx-(f-2*offset))).argmin()
            # neighb[2] = (np.abs(Fxx-(f+offset))).argmin()
            # neighb[3] = (np.abs(Fxx-(f+2*offset))).argmin()
            # center = (np.abs(Fxx-f)).argmin()
            # neighb = neighb.astype(int)

            neighb = np.where((Fxx < f + offset) & (Fxx > f - offset))
            center = round(np.median(neighb))

            neighb = neighb[0]
            neighb = neighb[neighb != center]
            SNRs[i] = (len(neighb) * Pxx[center]) / np.sum(Pxx[neighb])
        return np.mean(SNRs)

    def perform(self, data):
        """ 
        Data deve avere un canale solo
        """
        # Nel dubbio prendo solo il primo canale
        if data.ndim > 1:
            data = data[:, 0]

        ro = self.compute_features(data)
        return self.frequencies[ro.index(max(ro))]


class CCA(FeatureExtractor):
    """
    Riceve il vettore delle frequenze usate
    Crea il vettore dei segnali di riferimento con le frequenze
    Il vettore deve contenere tanti samples quanti la finestra corrente
    """

    def __init__(self, frequencies, samples, fs):
        FeatureExtractor.__init__(self, frequencies, samples, fs)
        self.referencies = [generate_references(samples, f, fs) for f in frequencies]

    def perform(self, data):
        ro = self.compute_features(data)

        return self.frequencies[ro.index(max(ro))]

    def compute_features(self, data):
        return [np.max(self._compute_max_corr(X=data, Y=ref)) for ref in self.referencies]

    def _compute_max_corr(self, X, Y):
        """
        X e Y needs to be matrixes or vectors
        righe, colonne -> samples, channels
        """
        try:
            n, p = X.shape
            n, q = Y.shape
        except ValueError as v:
            print str(v)
            print type(self).__name__ + ": Data/references with wrong format. How do you generate them?"
            exit(1)

        # normalize variables
        X = X.astype('float32', copy=False)
        X -= X.mean(axis=0)
        X /= np.max(np.abs(X))
        Y = Y.astype('float32', copy=False)
        Y -= Y.mean(axis=0)
        Y /= np.max(np.abs(Y))

        # covariances
        try:
            S = np.cov(X.T, Y.T, bias=1)
        except ValueError as v:
            print str(v)
            print type(self).__name__ + ": Data/references shapes not aligned. Have you set proper Fs in config.py?"
            exit(1)

        SXX = S[:p, :p]
        SYY = S[p:, p:]
        SXY = S[:p, p:]

        # 
        sqx, _ = LA.sqrtm(LA.inv(SXX), False)  # SXX^(-1/2)
        sqy, _ = LA.sqrtm(LA.inv(SYY), False)  # SYY^(-1/2)
        M = np.dot(np.dot(sqx, SXY), sqy.T)  # SXX^(-1/2) * SXY * SYY^(-T/2)
        M = np.nan_to_num(M)
        a, s, b = LA.svd(M, full_matrices=False)

        return s


class MSI(FeatureExtractor):
    """
    Riceve il vettore delle frequenze usate
    Ha un metodo perform che mi da la frequenza (numero in Hz) corrispondente
    Crea il vettore dei segnali di riferimento con le frequenze
    Il vettore deve contenere tanti samples quanti la finestra corrente
    """

    def __init__(self, frequencies, samples, fs):
        FeatureExtractor.__init__(self, frequencies, samples, fs)
        self.referencies = [generate_references(samples, f, fs) for f in frequencies]

    def compute_features(self, data):
        """
        Named parameters: data, ref
        """
        return [np.max(self._compute_MSI(X=data, Y=ref)) for ref in self.referencies]

    def perform(self, data):
        """
        Calcola l'MSI
        """
        ro = self.compute_features(data)

        return self.frequencies[ro.index(max(ro))]

    def _compute_MSI(self, X, Y):
        try:
            n, p = X.shape
            n, q = Y.shape
        except ValueError as v:
            print str(v)
            print type(self).__name__ + ": Data/references with wrong format. How do you generate them?"
            exit(1)

        X = X.astype('float32', copy=False)
        X -= X.mean(axis=0)
        X /= np.max(np.abs(X))
        Y = Y.astype('float32', copy=False)
        Y -= Y.mean(axis=0)
        Y /= np.max(np.abs(Y))

        try:
            C = np.cov(X.T, Y.T, bias=1)
        except ValueError as v:
            print str(v)
            print type(self).__name__ + ": Data/references shapes not aligned. Have you set proper Fs in config.py?"
            exit(1)

        CXX = C[:p, :p]
        CYY = C[p:, p:]

        sqx, _ = LA.sqrtm(LA.inv(CXX), False)  # SXX^(-1/2)
        sqy, _ = LA.sqrtm(LA.inv(CYY), False)  # SYY^(-1/2)

        # build square matrix
        u1 = np.vstack((sqx, np.zeros((sqy.shape[0], sqx.shape[1]))))
        u2 = np.vstack((np.zeros((sqx.shape[0], sqy.shape[1])), sqy))
        U = np.hstack((u1, u2))

        R = np.dot(np.dot(U, C), U.T)

        eigvals = LA.eigh(R)[0]
        eigvals /= np.sum(eigvals)
        # Compute index
        return 1 + np.sum(eigvals * np.log(eigvals)) / np.log(eigvals.shape[0])


class FLC(object):
    """
    flc = FLC(frequencies, samples, fs, order, mu) Fourier Linear Combiner

    Fa filtraggio LMS e trova la frequenza con minor ???
    Provare:
    - MSE 
    - MSE STD
    - max(W)
    - sum(W)
    - sum(abs(W))
    """

    def __init__(self, frequencies, samples, fs, order=2, mu=0.01):
        self.referencies = [generate_references(samples, f, fs) for f in frequencies]
        self.frequencies = frequencies
        self.order = order
        self.mu = mu
        self.lms = [LMS(np.zeros(2 * self.order), mu=self.mu) for f in frequencies]

    def perform(self, data):
        # Quattro filtri paralleli
        ros = []
        for i in range(len(self.frequencies)):
            ref = self.referencies[i]
            lms = self.lms[i]
            try:
                error = self._compute_error(data, ref.T, lms)
            except ValueError as v:
                print str(v)
                print type(self).__name__ + ": Data/references shapes not aligned. Have you set proper Fs in config.py?"
                exit(1)

            ros.append(np.abs(lms.W).sum())
            # ros.append(error.mean())
            # ros.append(error.std())

        # return self.frequencies[ ros.index(min(ros)) ]
        return self.frequencies[ros.index(max(ros))]


    def _compute_error(self, d, x, lms):

        ys = []
        yests = []
        for t in xrange(d.shape[0] - 1):
            X = x[:, t:t + 1]
            y = d[t + 1]  # predict
            yest = lms.est(X, y)
            ys += [y]
            yests += [yest]

        y = np.array(ys)
        yest = np.array(yests)
        err = (yest - y) ** 2
        return err