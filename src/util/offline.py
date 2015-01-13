# -*- coding: utf-8 -*-
"""
Metodi vari per analisi offline
"""

import numpy as np
from preprocessing import sliding_window


def extract_windowed_segment(data, ws, period, fs):
    """
    blablabla
    """
    segments = sliding_window(data, (fs * period, data.shape[1]), (fs * period, data.shape[1]))
    # Can't know the number of resulting windows. Instanciating with maximum possible number
    windows = np.zeros((len(data) / fs, ws * fs, data.shape[1]))

    length = 0
    for si, segment in enumerate(segments):
        wins = sliding_window(segment, (ws * fs, data.shape[1]), (fs, data.shape[1]))

        for wi, win in enumerate(wins):
            ii = len(wins) * si + wi  # global index
            windows[ii] = win
            length += 1

    return windows[0:length]


def extract_segment_start(data, ws, period, fs, n=2):
    """
    data : dataset to be segmented
    ws : window size
    iterations : N of complete iterations though the stimuli
    classes : # different label

    Extract first windows for each stimuli segment.
        Dataset must be a recording session, composed by several segment.
        blabla

    Es. >>>>>>>
    """
    windowsInPeriod = ws * 2 <= period and n or 1
    segments = sliding_window(data, (fs * period, data.shape[1]))
    # Can't know the number of resulting windows. Instanciating with maximum possible number
    windows = np.zeros((len(data) / (ws * fs), ws * fs, data.shape[1]))

    length = 0
    for si, segment in enumerate(segments):
        wins = sliding_window(segment, (ws * fs, data.shape[1]))
        if len(wins) > 1:
            wins = wins[0:windowsInPeriod]

        for wi, win in enumerate(wins):
            ii = len(wins) * si + wi  # global index
            windows[ii] = win
            length += 1

    return windows[0:length]


def make_label_windows(iterations, period, window, classes):
    """
    length : length of recording session, in samples
    period : segment with same label
    window : segment of analysis
    classes : # different label

    Build a label array for a dataset. 
        Assume a sliding windowing, with 1-s sliding.
        Assume a circular fashion (0, 1, 2, 3, 0, 1, 2, 3)

    Es. >>> make_label_windows(iterations = 2, period = 20, window=5, classes = 3)
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2])
    """
    # Each period -> (period - window + 1) windows
    length = iterations * classes * (period - window + 1)
    label = np.zeros(length, dtype=int)
    label.fill(-2)

    i = 0
    current = 0
    while current < length:
        # set label for each window range
        label[current:current + (period - window + 1)] = i
        # circular values in [0, classes]
        i = i != classes - 1 and i + 1 or 0

        current += period - window + 1

    return label


def make_label_segments(iterations, period, window, classes):
    """
    iterations : N of complete iterations though the stimuli
    period : segment with same label
    window : segment of analysis
    classes : # different label

    Build a label array for a dataset.
        Assume a simple (non-sliding) windowing, twice for each segment.
        If a segment is smaller than twice the window, just one label is created
        Assume a circular fashion (0, 1, 2, 3, 0, 1, 2, 3)

    Es. >>> make_label_matrix(iterations = 2, period = 20, window=5, classes = 3)
    array([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2])
    """
    windowsInPeriod = window * 2 <= period and 2 or 1
    length = iterations * classes * windowsInPeriod
    label = np.zeros(length, dtype=int)

    i = 0
    for it in range(0, length, windowsInPeriod):
        # set label for each window
        for w in range(windowsInPeriod):
            label[it + w] = i

        # circular values in [0, classes]
        i = i != classes - 1 and i + 1 or 0

    return label


def offline_classify(windows, frequencies, method):
    """
    Classify a recorded dataset data
    Windows must be an array representing windowed data. 
        Such that windows.shape = (#WINDOWS, #SAMPLES, #CHANNELS)
    Frequencies must be a list. Values in label must represent valid indexes in the list
    Method must offer a perform(data) method

    No reclassification, no pause

    Return an array with estimated label for each window
    """
    iterator = enumerate(windows)

    out = np.empty((windows.shape[0]))
    for ii, win in iterator:
        temp = method.perform(win)

        out[ii] = frequencies.index(temp)

    return out


def pseudo_online_classify(windows, frequencies, fs, method, period, pause=1, step=4):
    """
    Classify a recorded dataset data
    Windows must be an array representing windowed data. 
        Such that windows.shape = (#WINDOWS, #SAMPLES, #CHANNELS)
    Frequencies must be a list. Values in label must represent valid indexes in the list
    Method must offer a perform(data) method

    Output only if meet criteria
    Pause after each output

    Return a tuple (out, count, missed)
        out: an array with estimated label for each window
             -1 if no output for that window
        count: # of outputs provided
        missed: # of windows with no output
    """

    missed = 0  # n° of windows without output
    count = 0  # n° of output provided
    out = np.empty((windows.shape[0]))

    # Windowing
    iterator = enumerate(windows)
    windowLength = windows.shape[1] / fs
    winInSegment = period - windowLength + 1
    subSamples = (windowLength - 1) * fs

    for ii, win in iterator:
        # N° subwindows = step + 1
        subWindows = sliding_window(win, (subSamples, win.shape[1]), (np.floor(fs / step), win.shape[1]))

        temp = []
        for subWindow in subWindows:
            temp.append(method.perform(subWindow))

        # computing classification "confidence"
        for fIndex in range(len(frequencies)):
            if float(temp.count(frequencies[fIndex])) / len(temp) > 0.8:
                count += 1
                out[ii] = fIndex

                # simulate online waiting for gaze shift and full window
                for j in range(pause + windowLength):
                    try:
                        if ii + j + 1 >= ii + winInSegment:
                            break
                        iterator.next()
                        out[ii + j + 1] = -1
                    except StopIteration:
                        break  # game over
                break  # important to the else branch not be executed
        else:
            missed += 1
            out[ii] = -1

    return out, count, missed