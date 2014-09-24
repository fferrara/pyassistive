from emokit.emotiv import Emotiv
import platform
if platform.system() == "Windows":
    import socket  # Needed to prevent gevent crashing on Windows. (surfly / gevent issue #459)
import gevent
import time
import matplotlib.pyplot as plt
from collections import deque

SERIAL = "SN201308221848GM" # only necessary for OSX
PLOT_MIN_Y = -5
PLOT_MAX_Y = 5

def plot(y, yScale, bufferSize = 500):
    """
        Greenlet that plot y once per .1
        The y scale is specified through a tuple but is dynamically adjusted
    """
    buffer = deque([0]*bufferSize)
    _ = plt.axes(xlim=(0, bufferSize), ylim=(0, 1000))
    line, = plt.plot(buffer)
    plt.ion()
    plt.show()

    # Iterating the x axis, insert at head the new item and remove the oldest
    for i in xrange(0,bufferSize):
        # Adjusting Y scale
        minY = min(buffer)
        maxY = max(buffer)
        plt.ylim([minY,maxY])
        # Updating buffer
        buffer.appendleft(next(y))
        buffer.pop()
        # Plot refreshes with new buffer
        line.set_ydata(buffer)
        line.set_xdata(range(len(buffer)))

        plt.draw()
        print buffer[0]
        gevent.sleep(0)
        plt.pause(0.1)


def getSensorsInfo(emotiv, sensorName, isQuality=False):
    """
        Greenlet to get a packet from Emotiv headset.
        Return a generator with the value of the specified sensor.
        If isQuality is True, value will represent the connection quality
    """
    try:
        while True:
            packet = emotiv.dequeue()
            sensor = packet.sensors[sensorName]
            value = sensor['quality'] if isQuality else sensor['value']
            yield value
            gevent.sleep(0)
    except KeyboardInterrupt:
        emotiv.close()
    except GeneratorExit:
        emotiv.close()
    finally:
        emotiv.close()

if __name__ == "__main__":
    # Creating headset Object. No console output. No research headset
    headset = Emotiv(False, SERIAL, False)

    # Create a Greenlet for the setup...
    setupGLet = gevent.spawn(headset.setup)
    gevent.sleep(0) # ... and pass the control

    if not setupGLet.started:
        print 'Error: Is the device connected?'
        headset.close()
        quit()

    startTime = time.time()
    # Get the sensor data generator
    data = getSensorsInfo(headset, "O1", False)
    # Plot dynamically the value
    plot(data, (PLOT_MIN_Y, PLOT_MAX_Y ) )
