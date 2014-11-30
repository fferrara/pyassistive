 # -*- coding: utf-8 -*-
# Filtri servono a ridurre l'influenza del rumore
# Se abbiamo un segnale periodico da trovare, quando misuriamo il segnale principale avremo il nostro eroe sepolto da rumore bianco.
# Il rumore bianco ha uno spettro costante, cioè rompe il cazzo a tutte le frequenze
# Con un filtro, facciamo passare solo le frequenze che ci piacciono, cioè quella del nostro segnale originale più altre.
# Quindi avremo un segnale dove, oltre al nostro originale, c'è rumore bianco ma solo quelle componenti nelle frequenze non filtrate.
# In pratica sarà più vicino al segnale originale

import numpy as np
from scipy.signal import remez
from scipy.signal import ellip
from scipy.signal import welch
import matplotlib.pyplot as plt

from numpy.lib.stride_tricks import as_strided as ast
 
def sliding_window(a,ws,ss = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions
     
    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size 
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the 
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an 
                  extra dimension for each dimension of the input.
     
    Returns
        an array containing each n-dimensional window from a
    '''
     
    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)
     
    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every 
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)
     
     
    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))
     
    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))
     
    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return strided
     
    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    #dim = filter(lambda i : i != 1,dim)
    return strided.reshape(dim)

def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple, 
    even for one-dimensional shapes.
     
    Parameters
        shape - an int, or a tuple of ints
     
    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass
 
    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass
     
    raise TypeError('shape must be an int, or a tuple of ints')

def CAR(n):
    return np.eye(n) - 1. / float(n)

def get_passband_filter(passband, Fs, ellip = True):
    """
    passband needs to contain 6 elements, following style [0., stopband1, passband1, passband2, stopband2, nyquist]
    """
    try:
        if ellip:
            return _get_elliptic_filter(passband, Fs)
        else: return _get_fir_filter(passband, Fs)
    except ValueError as v:
        print str(v)
        print "Filter error. Have you set proper Fs in config.py?"
        exit(1)

def _get_elliptic_filter(passband, fs, order = 5, rp = 1, rs = 15):
    """
    Return an n-th order elliptic passband filter (default 5th)
    The resulting passband filter will be of order 2 * order
    Frequencies will be normalized with Fs / 2
    By default, the remaining parameters assume following values:
    rp = 1 (maximum passband ripple allowed in db)
    rs = 15 (minimum stopband attenuation required in db)
    """
    band = passband[2:4] / (fs / 2)
    return ellip(N = order, rp = rp, rs = rs, Wn = band, btype='pass')

def _get_fir_filter(passband, fs, order=183, weights=[5.75, 1., 5.75], mask=[0, 1, 0]):
    """
    Return a n-th order FIR filter
    """
    # return remez(order, passband, mask, weights, Hz=fs), 1.
    return remez(order, passband, mask, Hz=fs), 1.

def get_frequency_response(b, a, plot = False):
    w,h = signal.freqz(b,a)
    h_dB = 20 * log10 (abs(h))

    if plot:
        plt.subplot(211)
        plt.plot(w/max(w),h_dB)
        plt.ylim(-150, 5)
        plt.ylabel('Magnitude (db)')
        plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
        plt.title(r'Frequency response')
    
        plt.subplot(212)
        plt.l = len(b)
        plt.impulse = repeat(0.,l); impulse[0] =1.
        plt.x = arange(0,l)
        plt.response = signal.lfilter(b,a,impulse)
        plt.subplot(211)
        plt.stem(x, response)
        plt.ylabel('Amplitude')
        plt.xlabel(r'n (samples)')
        plt.title(r'Impulse response')

    return w / max(w), h_dB

def get_psd(data, window, fs, nfft, plot = None):
    samples = window * fs
    f, PSD = welch(data, fs, 'hann', scaling = 'density', nfft=nfft, nperseg=samples)

    if plot:
        plot.grid()
        plot.plot(f[np.where(f < 50)], PSD[np.where(f < 50)], label='PSD')
    return f, PSD