__author__ = 'iena'

import gevent
import signal
import sys
import os
from collections import deque
from gevent.queue import Queue
import numpy as np
import scipy.signal
from util import emotiv_engine

def detect_peak(timeline, data, threshold = 0.2):
    """
    Data: 1-dimensional array

    Peak => several moments with value > 0, with at least one value > threshold
    """
    commands = []
    count = 2  # how many peaks must happen
    offset = 1  # time interval the peaks happen within
    peaks = scipy.signal.find_peaks_cwt(data, np.arange(1, 5))
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

class Controller(object):
    def __init__(self, engine):
        self.isRunning = False
        self.isCollecting = False
        # EmoEngine supplies two samples each 0.5 seconds, leading to 4 each second
        self.SAMPLES = 10
        self.FIELDS = ['timestamp','look_left','look_right','eyebrow','clench']
        self.COMMANDS = commands = {'C' : 'Clench', 'B': 'Eyebrow', 'E' : 'Eye movement'}

        self.engine = engine
        self.dataQueue = Queue()

    def get_expressiv_packet(self):
        """
            Greenlet to get a packet from Emotiv headset.
            Append new data to queues where consumers will read from
        """
        try:
            while self.isRunning:
                if self.isCollecting:
                    print 'Reading packet...'
                    buf = np.zeros((self.SAMPLES, len(self.FIELDS)))

                    i = 0
                    while i < self.SAMPLES:
                        info = engine.get_expressiv_info()
                        if info is None:
                            continue
                        values = [info[field] for field in self.FIELDS]
                        buf[i] = np.array(values)
                        i += 1

                    if self.dataQueue is not None:
                        self.dataQueue.put_nowait(buf)
                else:
                    info = engine.get_expressiv_info()

                gevent.sleep(0)
        except KeyboardInterrupt:
            print ('Read stopped')
            self.isRunning = False
        except Exception as e:
            print ('Read Error: %s' % e)
            self.isRunning = False
        finally:
            print ('Read over')
            self.isRunning = False


    def produce_command(self, cmd):
        try:
            counter = 0
            self.errors = 0
            self.undefined = 0

            while self.isRunning:
                # Controller greenlets controls the command firing
                print 'Trying to produce command...'
                self.isCollecting = False
                buf = self.dataQueue.get()
                lookLeft = detect_peak(buf[:, 0], buf[:, 1], 1)
                lookRight = detect_peak(buf[:, 0], buf[:, 2], 1)
                eyebrows = detect_peak(buf[:, 0], buf[:, 3], 0.5)
                clenchs = detect_peak(buf[:, 0], buf[:, 4], 0.5)

                if len(lookLeft) > 0 or len(lookRight) > 0:
                    print self.COMMANDS['E']
                    if cmd != 'E':
                        errors += 1
                    counter = 0
                    gevent.sleep(2)
                elif len(clenchs) > 0:
                    print self.COMMANDS['C']
                    if cmd != 'C':
                        errors += 1
                    counter = 0
                    gevent.sleep(2)
                elif len(eyebrows) > 0:
                    print self.COMMANDS['B']
                    if cmd != 'B':
                        errors += 1
                    counter = 0
                    gevent.sleep(2)
                elif counter == 10:
                    print 'START OVER'
                    counter = 0
                    self.undefined += 1
                    gevent.sleep(0.1)
                else:
                    counter += 1
                    gevent.sleep(0.1)

                self.isCollecting = True
                gevent.sleep(0)
        except Exception as e:
            print ('Controller error: %s' % e)
            self.isRunning = False
        finally:
            print ('Controller over')
            self.isRunning = False

if __name__ == '__main__':
    engine = emotiv_engine.EmotivEngine()

    cmd = raw_input('Command to test: ')

    controller = Controller(engine)
    controller.isRunning = True
    controller.isCollecting = True

    # Create the sensor data generator
    g1 = gevent.spawn(controller.get_expressiv_packet)
    # Create controller routine
    g2 = gevent.spawn(controller.produce_command, cmd)
    # Kill both at Ctrl+C
    gevent.signal(signal.SIGINT, gevent.killall, [g1, g2])
    # Run them until termination
    try:
        gevent.joinall([g1, g2])
    except KeyboardInterrupt:
        pass
    finally:
        print 'Errors: %d, Undefined: %d' % (controller.errors, controller.undefined)
