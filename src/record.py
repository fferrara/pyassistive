#!/usr/bin/env python
""" 
Utility for recording sessions. The concurrent execution flow is regulated by producer-consumer pattern, 
with one producer (get_sensors_info) and two consumer, with one data queue each.

Command 'show' plots the values each sensor read by Emotiv headset.

Command 'record' will additionally start a SSVEP recording session, with two stimuli.
Documentation for the stimuli can be found in corresponding C files.
The result of 'record' is to store recorded EEG values in a numpy array and save it in binary format.
"""

from emokit.emotiv import Emotiv
import platform
if platform.system() == "Windows":
    import socket  # Needed to prevent gevent crashing on Windows. (surfly / gevent issue #459)
import gevent
import time
import signal
import sys
import matplotlib.pyplot as plt
from collections import deque
from gevent.queue import Queue
from gevent.event import Event
from gevent.util import wrap_errors
import numpy as np
import argparse

SERIAL = "SN201308221848GM" # only necessary for OSX

class Recorder(object):
    """
    Recorder class. Producer, consumers and controller greenlets are methods of this class.
    Although not implemented, Recorder should be a Singleton.
    """
    def __init__(self, emotiv):
        self.isRunning = False
        self.isRecording = False

        self.sensors = ['F3','FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4'] 
        self.PLOT_MIN_Y = 0
        self.PLOT_MAX_Y = 1000
        self.ITERATIONS = 1
        self.RECORDING_PERIOD = 10
        self.PAUSE_INTER_INTERATIONS = 5
        self.FILENAME = "ciao.txt"

        self.headset = emotiv
        self.plotQueue = Queue()
        self.recorderQueue = Queue()
        self.recordingEvt = Event()

    def get_sensors_info(self):
        """
            Greenlet to get a packet from Emotiv headset.
            Append new data to queues where consumers will read from
        """
        try:
            while self.isRunning:
                packet = self.headset.dequeue()
                values = [packet.sensors[name]['value'] for name in self.sensors]
                if self.plotQueue is not None:
                	self.plotQueue.put_nowait(values)
                if self.recorderQueue is not None and self.isRecording:
                	self.recorderQueue.put_nowait(values)
                
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

    def controller(self):
        """
        Greenlet that controls recording process.
        Performs many iterations of recording response to the stimuli, first left and then right.
        """
        try:
            for i in xrange(self.ITERATIONS):
                # SX
                for i in xrange(self.PAUSE_INTER_INTERATIONS):
                    print ('Seconds to record SX: %i' % (self.PAUSE_INTER_INTERATIONS - i))
                    gevent.sleep(1)

                print ('Start recording SX')
                self.isRecording = True
                self.recordingEvt.set()
                gevent.sleep(self.RECORDING_PERIOD)
                
                self.recordingEvt.clear()
                self.isRecording = False
                print ('Stop recording SX')
                
                # DX
                for i in xrange(self.PAUSE_INTER_INTERATIONS):
                    print ('Seconds to record DX: %i' % (self.PAUSE_INTER_INTERATIONS - i))
                    gevent.sleep(1)
                
                print ('Start recording DX')
                self.recordingEvt.set()
                self.isRecording = True
                gevent.sleep(self.RECORDING_PERIOD)

                self.recordingEvt.clear()
                self.isRecording = False
                print ('Stop recording DX')
        except Exception as e:
            print ('Controller error: %s' % e)
            self.isRunning = False
        finally:
            print ('Controller over')
            self.isRunning = False

    def recorder(self):
        """
        Greenlet that store data read from the headset into a numpy array
        """
        data = None

        try:
            while self.isRunning:
                # Controller greenlets controls the recording
                self.recordingEvt.wait()
                while not self.recorderQueue.empty():
                    values = self.recorderQueue.get()
                    if data is None:
                        data = np.array(values, dtype=int)
                    else:
                        data = np.vstack((data, values))

                gevent.sleep(0)
        except Exception as e:
            print ('Recorder error: %s' % e)
            self.isRunning = False
        finally:
            print ('Recorder over')
            np.savetxt(self.FILENAME, data, fmt="%i")
            self.isRunning = False

    def plot(self, bufferSize = 50):
        """
            Greenlet that plot y once per .1
            The y scale is specified through global config but is dynamically adjusted
        """
        ax = plt.subplot(111)

        canvas = ax.figure.canvas
        plt.grid() # to ensure proper background restore
        background = None

        plotsNum = len(self.sensors)
        buffers = [deque([0]*bufferSize) for i in xrange(plotsNum)]
        lines = [plt.plot(buffers[i], lw=1, label=self.sensors[i]).pop() for i in xrange(plotsNum)]
        plt.legend()
        
        plt.axis([0, bufferSize, self.PLOT_MIN_Y, self.PLOT_MAX_Y])

        try:
            while self.isRunning:
                while not self.plotQueue.empty() and self.isRunning:
                    if background is None:
                        background = canvas.copy_from_bbox(ax.bbox)
                    canvas.restore_region(background)

                    # Getting values from queue
                    values = self.plotQueue.get()
                    # Adjusting Y scale
                    minY = min(min(buffers[0:])) - 100
                    maxY = max(max(buffers[0:])) + 100
                    plt.ylim([minY,maxY])
                    # Updating buffer
                    [buffers[i].appendleft(values[i]) for i in xrange(plotsNum)]
                    [buffers[i].pop() for i in xrange(plotsNum)]
                    # Plot refreshes with new buffer
                    [lines[i].set_ydata(buffers[i]) for i in xrange(plotsNum)]

                    plt.draw()
                    plt.pause(0.000001)
                    gevent.sleep(.1)
        except Exception as e:
            print ('Plot error: %s' % e)
            self.isRunning = False
        finally:
            print 'Plot over'
            self.isRunning = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EEG signal recorder.')
    parser.add_argument('command', metavar='command', type=str, nargs=1, choices=['plot', 'record'])
    args = parser.parse_args(sys.argv[1:])

    # Creating headset Object. No console output. No research headset
    headset = Emotiv(False, SERIAL, False)
    recorder = Recorder(headset)

    # Create a Greenlet for the setup...
    setupGLet = gevent.spawn(headset.setup)
    gevent.sleep(1) # ... and pass the control

    if not setupGLet.started:
        print 'Error: Is the device connected?'
        headset.close()
        quit()

    recorder.isRunning = True
    if args.command[0] == 'plot':
        # Create the sensor data generator
        g1 = gevent.spawn(recorder.get_sensors_info)
        # Create plot routine
        g2 = gevent.spawn(recorder.plot)
        # Kill both at Ctrl+C
        gevent.signal(signal.SIGINT, gevent.killall, [g1, g2])
        # Run them until termination
        gevent.joinall([g1, g2])
    elif args.command[0] == 'record':
        # Create the sensor data generator
        g1 = gevent.spawn(recorder.get_sensors_info)
        # Create plot routine
        g2 = gevent.spawn(recorder.plot)
        # Create recorder and controller routines
        g3 = gevent.spawn(recorder.recorder)
        g4 = gevent.spawn(recorder.controller)
        # Kill both at Ctrl+C
        gevent.signal(signal.SIGINT, gevent.killall, [g1, g2, g3, g4])
        # Run them until termination
        try:
            gevent.joinall([g1, g2, g3, g4])
        except KeyboardInterrupt:
            pass
