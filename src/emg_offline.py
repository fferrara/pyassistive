__author__ = 'iena'

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal

from util import config

def detect_peak(timeline, data, threshold):
    """
    Data: 1-dimensional array

    Peak => several moments with value > 0, with at least one value > threshold
    """
    commands = []
    count = 2  # how many peaks must happen
    offset = 2  # time interval the peaks happen within
    peaks = signal.find_peaks_cwt(data, np.arange(1, 5))
    peaks = set(peaks)

    for p in peaks:
        if not set( range(p, p+count)).issubset(peaks):
            continue
        if not timeline[p + count] < timeline[p] + offset:
            continue
        if np.any(data[p:p+count] < threshold):
            continue
        commands.append( timeline[p] )

    return commands


if __name__ == '__main__':
    FILENAME = 'flavio_emg_2.csv'
    dataset = np.loadtxt(os.path.join(config.DATA_PATH, FILENAME), delimiter=',', skiprows=1)

    time = dataset[:, 0]
    lookLeft = dataset[:, 1]
    lookRight = dataset[:, 2]
    clench = dataset[:, 4]
    eyebrow = dataset[:, 3]

    fig, ax = plt.subplots( nrows=3 )
    ax[0].plot(time, lookLeft, label="left")
    ax[0].plot(time, lookRight, label="right")
    ax[0].legend()

    ax[1].plot(time, clench, label='clench')
    ax[1].plot(time, eyebrow, label='eyebrow')
    ax[1].legend()

    # print time[signal.find_peaks_cwt(lookRight, np.arange(1, 5))]
    # ax[0].vlines(time[signal.find_peaks_cwt(lookRight, np.arange(1, 5))], 0, 1, colors='r')
    # ax[1].vlines( time[signal.argrelmax(eyebrow, order=5)], 0, 1, colors='r')

    eyebrows = detect_peak(time, eyebrow, 0.2)
    print 'Eyebrow @ %s' % str(eyebrows)
    clenchs = detect_peak(time, clench, 0.2)
    print 'Clench @ %s' % str(clenchs)
    plt.show()




