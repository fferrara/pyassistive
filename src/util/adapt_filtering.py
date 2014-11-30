# -*- coding: utf-8 -*-
"""
Metodi vari per filtraggio adattativo
"""
import numpy as np
import scipy.linalg as LA
import config


class LMS(object):
    """ 
    lms = LMS( W, damp=.5 )  Least mean squares adaptive filters
    in:
        W: initial weights, e.g. np.zeros( 33 )
        mu: a damping factor for swings in W 

    Swings in W are damped a bit with a damping factor a.k.a. mu in 0 .. 1:
        W += damp * c * X

    Convergence and time constant are dependent from mu. Check the theory

"""
#...............................................................................
    def __init__(self, W, mu):
        self.W = np.squeeze( getattr( W, "A", W ))  # matrix -> array
        # Sufficient for stability: 0 < mu < 1 / tr(R)
        # R = E[x * x.T]  = np.corrcoef(x.T)
        self.mu = mu

    def est(self, X, y):
        X = np.squeeze( getattr( X, "A", X ))
        yest = self.W.dot(X)

        if X.ndim > 1:
            norm = np.diag(X.T.conj().dot(X))
            c = (y - yest) / norm
            self.W += 2 * self.mu * c.dot(X.T)
        else:  # one-channel X
            norm = X.dot(X)
            c = y - yest
            self.W += 2 * self.mu * c * X / norm

        return yest

class ANC(object):
    """
    Adaptive noise canceller filter:
        B. Widrow, et al., “Adaptive Noise Cancelling: Principles and Applications”

    An NCF produces an estimate of the noise by filtering the reference input and then subtracting this noise
    estimate from the primary input containing both signal and noise.
    The reference is obtained by placing one sensor in the noise field where the signal is absent or its strength is weak enough.

    NCF uses the NLMS algorithm for adjusting its impulse response and properly estimate the noise.
    """
    def __init__(self, order, mu, lms = None):
        if lms is None:
            lms = LMS( np.zeros(order), mu)
        self.lms = lms
        self.order = order

    def filter(self, x, d):
        """ Finds the NLMS adaptive filter w based on the inputs

        Inputs:
            x: reference data to the filter, NxM array
            d: primary input signal, NxC array
            p: filter order
        Outputs:
            y: estimated reference sequence y(n): NxC array
            e: error sequence e(n): NxC array
        """
        N = len(d)
        ys = np.zeros((N, d.shape[1]))
        epss = np.zeros((N, d.shape[1]))
        for t in xrange(N - self.order):
            X = x[t:t+self.order]
            y = d[t+self.order]
            yest = self.lms.est(X, y)
            ys[t] = yest
            epss[t] = y - yest

        return ys, epss

class AST(object):
    """
    Adaptive Self-tuning filter:
        B. Widrow, et al., “Adaptive Noise Cancelling: Principles and Applications”

    See class adapt_filtering.ANC
    An AST is obtained when the Adaptive Noise Canceller uses a delayed version of primary input as reference input.
    
    Filtrando il segnale, AST cercherà di ricostruire la parte periodica (correlata) e sottrarla dall'ingresso primario.
    Il delay deve essere sufficiente per far sì che la parte stocastica sia incorrelata.
    Quindi, se prendiamo l'uscita broadband, avremo il segnale pulito dalla parte periodica. Se prendiamo l'uscita narrowband, abbiamo
    la parte periodica.
    """
    def __init__(self, order, mu):
        self.lms = LMS( np.zeros(order), mu)
        self.order = order


    def filter(self, d):
        """ Finds the NLMS adaptive filter w based on the inputs

        Inputs:
            d: primary input signal, NxC array
            p: filter order
        Outputs:
            y: estimated reference sequence y(n): NxC array
            e: error sequence e(n): NxC array
        """
        # Con ANC
        anc = ANC(self.order, None, self.lms)
        return anc.filter(d, d)
        # Reimplementato
        N = len(d)
        ys = np.zeros((N - self.order, d.shape[1]))
        epss = np.zeros((N - self.order, d.shape[1]))
        for t in xrange(N - self.order):
            X = d[t:t+self.order]
            y = d[t+self.order]
            yest = self.lms.est(X, y)
            ys[t] = yest
            epss[t] = y - yest

        return ys, epss

class ANF(object):
    """
    Adaptive Notch filter:
        B. Widrow, et al., “Adaptive Noise Cancelling: Principles and Applications”
        J. Glover, "Adaptive Noise Canceling Applied to Sinusoidal Interferences"

    See class adapt_filtering.ANC
    An ANF is obtained when the Adaptive Noise Canceller uses a pure sinusoidal signal as reference input.

    Il filtro adattativo è equivalente a un notch di ordine 2 quando usiamo sin e cos. Ordine più alto ci permette di eliminare
    più frequenze.
    Filtrando la sinusoide, il filtro cerca di adattarla al picco più vicino che trova nel segnale primario.
    Quindi, se prendiamo l'uscita broadband, avremo il segnale pulito dalla componente di picco. Se prendiamo l'uscita narrowband, abbiamo
    la parte periodica.

    NCF uses the NLMS algorithm for adjusting its impulse response and properly estimate the noise.
    """
    def __init__(self, order, mu):
        self.lms = LMS( np.zeros(order), mu)
        self.order = order

    def filter_synch(self, x, d):
        """
        Synchronized sampling with sin/cos references
        ONLY SINGLE CHANNEL, BECAUSE MULTICHANNEL GENERATE 
            LINEAR DIPENDENT OUTPUT (ONE COMMON WEIGHT ALL CHANNELS)
        """
        N = len(d)
        ys = np.zeros(N)
        epss = np.zeros(N)
        for t in xrange( N ):
            X = x[t]
            y = d[t]
            yest = self.lms.est(X, y)
            ys[t] = yest
            epss[t] = y - yest

        return ys, epss

    def filter_delay(self, x, d):
        """
        Delayed references through TDL
        """
        anc = ANC(self.order, None, self.lms)
        return anc.filter(x, d)


class ACF(object):
    """
    Adaptive Comb filter:
        B. Widrow, et al., “Adaptive Noise Cancelling: Principles and Applications”
        P. Laguna, et al., "The adaptive linear combiner with a periodic-impulse reference
            input as a linear comb filter"

    See class adapt_filtering.ANC
    
    NCF uses the NLMS algorithm for adjusting its impulse response and properly estimate the noise.
    """
    def filter(self, d):
        pass