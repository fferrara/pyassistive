from emokit.emotiv import Emotiv
import platform
if platform.system() == "Windows":
    import socket  # Needed to prevent gevent crashing on Windows. (surfly / gevent issue #459)
import gevent
import time
import matplotlib.pyplot as plt
from collections import deque

SERIAL = "SN201308221848GM" # only necessary for OSX

def plot(y):
    """
        Greenlet that plot info once per millisecond
    """
    items = len(y)
    buffer = deque([0]*items)
    iterator = iter(y)
    axes = plt.axes(xlim=(0, items), ylim=(min(y), max(y)))
    line, = plt.plot(y)
    plt.ion()

    plt.show()
    # Foreach (x, y), insert at head the new item and remove the oldest
    for i in xrange(0,items):
        buffer.appendleft(next(iterator))
        buffer.pop()
        line.set_ydata(buffer)
        line.set_xdata(range(len(buffer)))
        plt.draw()
        print buffer[0]
        time.sleep(0.001)
        plt.pause(0.001)


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
    tics = []
    values = []
    try:
        while True:
            # Obtain packet
            tics.append(time.time() - startTime)
            packet = headset.dequeue()
            values.append(packet.gyro_x)
            gevent.sleep(0)
    except KeyboardInterrupt:
        headset.close()
    finally:
        headset.close()

    tics.pop(0)
    values.pop(0)
    #values = []


    for i in xrange(len(tics)):
        print "At {0:.10f}, value {1}".format(tics[i], values[i])

    plot(values)

