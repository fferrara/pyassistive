 # -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.linalg as LA
import config
from preprocessing import get_psd

class Processing(object):
    """
    Metodi vari che servono al Processing
    """

    @staticmethod
    def generate_references(samples, frequency, harmonics=2):
        vector = np.zeros([samples, 2 * harmonics])
        time = np.arange(samples, dtype=float) / config.FS
        for i in xrange(0, harmonics):
            vector[:, 2 * i] = np.sin((2**i) * 2 * np.pi * frequency * time)
            vector[:, 2 * i + 1] = np.cos((2**i) * 2 * np.pi * frequency * time)

        return vector

class PSDA(object):
    """
    Normale PSDA. Calcola il SNR tramite la PSD e tira fuori la classe più probabile
    """
    def __init__(self, frequencies, window):
        self.frequencies = frequencies
        self.window = window

    def perform(self, data, resolution = 20):
        """ 
        Data deve avere un canale solo
        """
        # Nel dubbio prendo solo il primo canale
        if data.ndim > 1:
            data = data[:, 0]

        offset = resolution * config.FS / config.NFFT
        f, PSD = get_psd(data, self.window)

        ro = []
        for frequency in self.frequencies:
            neighb = np.where((f < frequency + offset) & (f > frequency - offset))
            center = round(np.median(neighb))
            ro.append(10 * np.log10((resolution * PSD[center]) / np.sum(PSD[neighb])))

        return self.frequencies[ ro.index(max(ro)) ]

class CCA(object):
    """
    Riceve il vettore delle frequenze usate
    Ha un metodo perform che mi da la frequenza (numero in Hz) corrispondente
    """
    def __init__(self, frequencies, window):
        self.referencies = [Processing.generate_references(window * config.FS, f) for f in frequencies]
        self.frequencies = frequencies

    def perform(self, data, window):
        """
        Crea il vettore dei segnali di riferimento con le frequenze
        Il vettore deve contenere tanti samples quanti la finestra corrente
        """
        ro = [np.max(self._compute_max_corr(data, ref)) for ref in self.referencies]

        return self.frequencies[ ro.index(max(ro)) ]

    def _compute_max_corr(self, X, Y):
        """
        X e Y needs to be matrixes or vectors
        righe, colonne -> samples, channels
        """
        try: 
            n, p = X.shape
            n, q = Y.shape
        except ValueError as v:
            return []

        # normalize variables
        X = X.astype('float32', copy=False)
        X -= X.mean(axis=0)
        X /= np.max(np.abs(X))
        Y = Y.astype('float32', copy=False)
        Y -= Y.mean(axis=0)
        Y /= np.max(np.abs(Y))

        # covariances
        S = np.cov(X.T, Y.T, bias=1)
        SXX = S[:p,:p]
        SYY = S[p:,p:]
        SXY = S[:p,p:]
        SYX = S[p:,:p]

        # 
        sqx,_ = LA.sqrtm(LA.inv(SXX),False) # SXX^(-1/2)
        sqy,_ = LA.sqrtm(LA.inv(SYY),False) # SYY^(-1/2)
        M = np.dot(np.dot(sqx, SXY), sqy.T) # SXX^(-1/2) * SXY * SYY^(-T/2)
        M = np.nan_to_num(M)
        a, s, b = LA.svd(M, full_matrices=False)

        return s

class MSI(object):
    """
    Riceve il vettore delle frequenze usate
    Ha un metodo perform che mi da la frequenza (numero in Hz) corrispondente
    Crea il vettore dei segnali di riferimento con le frequenze
    Il vettore deve contenere tanti samples quanti la finestra corrente
    """
    def __init__(self, frequencies, window):
        self.referencies = [Processing.generate_references(window * config.FS, f) for f in frequencies]
        self.frequencies = frequencies

    def perform(self, data, window):
        """
        Calcola l'MSI
        """
        ro = [np.max(self._compute_MSI(data, ref)) for ref in self.referencies]

        return self.frequencies[ ro.index(max(ro)) ]

    def _compute_MSI(self, X, Y):
        try: 
            n, p = X.shape
            n, q = Y.shape
        except ValueError as v:
            return []

        X = X.astype('float32', copy=False)
        X -= X.mean(axis=0)
        X /= np.max(np.abs(X))
        Y = Y.astype('float32', copy=False)
        Y -= Y.mean(axis=0)
        Y /= np.max(np.abs(Y))

        C = np.cov(X.T, Y.T, bias=1)
        CXX = C[:p,:p]
        CYY = C[p:,p:]

        sqx,_ = LA.sqrtm(LA.inv(CXX),False) # SXX^(-1/2)
        sqy,_ = LA.sqrtm(LA.inv(CYY),False) # SYY^(-1/2)

        # build square matrix
        u1 = np.vstack((sqx, np.zeros((sqy.shape[0], sqx.shape[1]))))
        u2 = np.vstack((np.zeros((sqx.shape[0], sqy.shape[1])), sqy))
        U = np.hstack((u1, u2))
        
        R = np.dot(np.dot(U, C), U.T)

        eigvals = LA.eigh(R)[0]
        eigvals /= np.sum(eigvals)
        # Compute index
        return 1 + np.sum(eigvals * np.log(eigvals)) / np.log(eigvals.shape[0])