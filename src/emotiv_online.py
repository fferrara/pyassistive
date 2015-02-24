import numpy as np
import platform
if platform.system() == "Windows":
    import socket  # Needed to prevent gevent crashing on Windows. (surfly / gevent issue #459)
import gevent
from gevent.queue import Queue
import signal
import sys
import os
from subprocess import Popen
from emokit.emotiv import Emotiv
from util import config, preprocessing, featex

class Recorder(object):
    """
    Recorder class. Read data from Emotiv and process the signal.
    Although not implemented, Recorder should be a Singleton.
    """
    def __init__(self, emotiv):
        self.isRunning = False
        self.isRecording = False

        self.sensors = config.SENSORS
        self.headset = emotiv
        self.dataQueue = Queue()

        # processing parameters
        self.freqs = [6.4, 6.9, 8]
        self.winSize = 6
        self.method = featex.MSI(list(self.freqs), self.winSize * config.FS, config.FS)

    def get_sensors_info(self):
        """
            Greenlet to get a packet from Emotiv headset.
            Append new data to queues where consumers will read from
        """
        try:
            while self.isRunning:
                buf = np.zeros((config.FS, len(self.sensors)))
                for i in range(len(buf)):
                    packet = self.headset.dequeue()
                    values = [packet.sensors[name]['value'] for name in self.sensors]
                    buf[i] = np.array(values)

                if self.dataQueue is not None and self.isRecording:
                    self.dataQueue.put_nowait(buf)

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
            self.headset.close()

    def process_data(self):
        try:
            while self.isRunning or not self.dataQueue.empty():
                # Controller greenlets controls the recording
                while self.isRecording or not self.dataQueue.empty():
                    # collecting packets
                    window = np.empty(self.winSize * config.FS, len(self.sensors))
                    for counter in range(self.winSize):
                        buf = self.recorderQueue.get()
                        window[counter] = buf

                    # processing time window
                    out = self._process_window(window)
                    while out is None:
                        # slide current window

                    print "OUPUT: " + str(out)
                    gevent.sleep(1)
                gevent.sleep(0)
        except Exception as e:
            print ('Recorder error: %s' % e)
            self.isRunning = False
        finally:
            print ('Recorder over')
            self.isRunning = False


