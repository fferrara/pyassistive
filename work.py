from __future__ import print_function
from emokit.emotiv import Emotiv
import platform
if platform.system() == "Windows":
    import socket  # Needed to prevent gevent crashing on Windows. (surfly / gevent issue #459)
import gevent
import time
import signal
import matplotlib.pyplot as plt
from collections import deque
from gevent.queue import Queue
import numpy as np


SERIAL = "SN201308221848GM" # only necessary for OSX
PLOT_MIN_Y = 0
PLOT_MAX_Y = 1000

isRunning = False
currentValue = Queue()
prova = Queue()

def getSensorsInfo(emotiv, sensorName):
    """
        Greenlet to get a packet from Emotiv headset.
        Return a generator with the value of the specified sensor.
        If isQuality is True, value will represent the connection quality
    """
    global isRunning
    try:
        while isRunning:
            packet = emotiv.dequeue()
            values = [packet.sensors[name]['value'] for name in sensorName]
            #print values, tic()
            currentValue.put_nowait(values)
            prova.put_nowait(values)
            gevent.sleep(0)
    except KeyboardInterrupt:
        isRunning = False
    except GeneratorExit:
        isRunning = False
    finally:
        isRunning = False
        print (stderr, 'Generation over')
        emotiv.close()

def showTest():
    global isRunning
    startTime = time.time()
    tic = lambda: 'at %1.1f seconds' % (time.time() - startTime)
    print ('partito')
    try:
        while isRunning:
            while not prova.empty():
                values = prova.get()
                print (values, tic())

            gevent.sleep(0)
    except:
        print ('print test error')
        isRunning = False
    finally:
        print ('Print test over')
        isRunning = False


def plot(plotsNum, labels, bufferSize = 50):
    """
        Greenlet that plot y once per .1
        The y scale is specified through global config but is dynamically adjusted
    """
    global isRunning
    ax = plt.subplot(111)

    canvas = ax.figure.canvas
    plt.grid() # to ensure proper background restore
    background = None

    buffers = [deque([0]*bufferSize) for i in range(plotsNum)]
    lines = [plt.plot(buffers[i], lw=1, label=labels[i]).pop() for i in xrange(plotsNum)]
    plt.legend()
    
    plt.axis([0, bufferSize, PLOT_MIN_Y, PLOT_MAX_Y])

    try:
        while isRunning:
            while not currentValue.empty():
                if background is None:
                    background = canvas.copy_from_bbox(ax.bbox)
                canvas.restore_region(background)

                # Getting values from queue
                values = currentValue.get()
                # Adjusting Y scale
                minY = min(min(buffers[0:])) - 50
                maxY = max(max(buffers[0:])) + 50
                plt.ylim([minY,maxY])
                # Updating buffer
                [buffers[i].appendleft(values[i]) for i in xrange(plotsNum)]
                [buffers[i].pop() for i in xrange(plotsNum)]
                # Plot refreshes with new buffer
                [lines[i].set_ydata(buffers[i]) for i in xrange(plotsNum)]

                plt.draw()
                plt.pause(0.000001)
            gevent.sleep(1)
    except:
        isRunning = False
    finally:
        print ('Plot over')
        isRunning = False


if __name__ == "__main__":
    # Creating headset Object. No console output. No research headset
    headset = Emotiv(False, SERIAL, False)

    # Create a Greenlet for the setup...
    setupGLet = gevent.spawn(headset.setup)
    gevent.sleep(0) # ... and pass the control

    if not setupGLet.started:
        print ('Error: Is the device connected?')
        headset.close()
        quit()

    isRunning = True

    # Create the sensor data generator
    sensors = ["O1","O2", "AF3", "AF4", "T8"]
    g1 = gevent.spawn(getSensorsInfo, headset, sensors)
    # Create plot routine
    g2 = gevent.spawn(plot, len(sensors), sensors)
    g3 = gevent.spawn(showTest)
    # Kill both at Ctrl+C
    gevent.signal(signal.SIGINT, gevent.killall, [g1, g2, g3])
    # Run them until termination
    gevent.joinall([g1, g2, g3])
