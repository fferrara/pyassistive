#!/usr/bin/env python
""" 
Utility for recording sessions. The concurrent execution flow is regulated by producer-consumer pattern, 
with one producer (get_sensors_info) and two consumer, with one data queue each.

Command 'show' plots the values each sensor read by Emotiv headset.

Command 'record' will additionally start a SSVEP recording session, with two stimuli.
Documentation for the stimuli can be found in corresponding C files.
The result of 'record' is to store recorded EEG values in a numpy array and save it in binary format.
"""

import platform
if platform.system() == "Windows":
    import socket  # Needed to prevent gevent crashing on Windows. (surfly / gevent issue #459)
import gevent
import signal
import sys
import os
from subprocess import Popen
import matplotlib.pyplot as plt
from collections import deque
from gevent.queue import Queue
import numpy as np
import scipy.io as sio
import argparse

from emokit.emotiv import Emotiv
from util import config

SERIAL = "SN201308221848GM" # only necessary for OSX

class Recorder(object):
    """
    Recorder class. Producer, consumers and controller greenlets are methods of this class.
    Although not implemented, Recorder should be a Singleton.
    """
    def __init__(self, emotiv):
        self.isRunning = False
        self.isRecording = False

        self.sensors = config.SENSORS
        self.PLOT_MIN_Y = 0
        self.PLOT_MAX_Y = 1000

        #### PROTOCOL DEFINITION ####
        self.ITERATIONS = config.RECORDING_ITERATIONS
        self.RECORDING_PERIOD = config.RECORDING_PERIOD # Recording stimulated SSVEP
        self.PAUSE_INTER_RECORDING = config.PAUSE_INTER_RECORDING
        self.STIMULI_PATH = config.STIMULI_PATH
        self.DATA_PATH = config.DATA_PATH
        self.FILENAME = "emotiv_original_3"

        self.headset = emotiv
        self.plotQueue = Queue()
        self.recorderQueue = Queue()

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
        frequencies = ['64', '80'] # SX DX
        SX = [os.path.join(self.STIMULIPATH, exe) for exe in os.listdir(self.STIMULIPATH) if exe.endswith("Sx.exe")]
        DX = [os.path.join(self.STIMULIPATH, exe) for exe in os.listdir(self.STIMULIPATH) if exe.endswith("Dx.exe")]

        SXwindow = Popen(args=[SX, frequencies[0]])
        DXwindow = Popen(args=[DX, frequencies[1]])

        SXwindow = Popen(args=SX)
        DXwindow = Popen(args=DX)
        gevent.sleep(10)
        try:
            for i in xrange(self.ITERATIONS):
                # SX
                for i in xrange(self.PAUSE_INTER_RECORDING):
                    print ('Seconds to record SX: %i' % (self.PAUSE_INTER_RECORDING - i))
                    gevent.sleep(1)

                print ('Start recording SX')
                self.isRecording = True
                gevent.sleep(self.RECORDING_PERIOD)
                
                self.isRecording = False
                print ('Stop recording SX')
                
                # DX
                for i in xrange(self.PAUSE_INTER_RECORDING):
                    print ('Seconds to record DX: %i' % (self.PAUSE_INTER_RECORDING - i))
                    gevent.sleep(1)
                
                print ('Start recording DX')
                self.isRecording = True
                gevent.sleep(self.RECORDING_PERIOD)

                self.isRecording = False
                print ('Stop recording DX')
        except Exception as e:
            print ('Controller error: %s' % e)
            self.isRunning = False
        finally:
            if SXwindow is not None:
                SXwindow.kill()
            if DXwindow is not None:
                DXwindow.kill()
            print ('Controller over')
            self.isRunning = False

    def recorder(self):
        """
        Greenlet that store data read from the headset into a numpy array
        """
        data = None

        try:
            while self.isRunning or not self.recorderQueue.empty():
                # Controller greenlets controls the recording
                while self.isRecording or not self.recorderQueue.empty():
                    while not self.recorderQueue.empty():
                        buf = self.recorderQueue.get()

                        if data is None:
                            data = np.array(buf, dtype=int)
                        else:
                            data = np.vstack((data, buf))

                    gevent.sleep(1)
                gevent.sleep(0)
        except Exception as e:
            print ('Recorder error: %s' % e)
            self.isRunning = False
        finally:
            print ('Recorder over')
            sio.savemat(os.path.join(self.DATA_PATH, self.FILENAME), {'X' : data})
            self.isRunning = False

    def plot(self, bufferSize = 500):
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
                while not self.plotQueue.empty():
                    # Getting values from queue
                    values = self.plotQueue.get()
                    # Updating buffer
                    [buffers[i].appendleft(values[i]) for i in xrange(plotsNum)]
                    [buffers[i].pop() for i in xrange(plotsNum)]


                if background is None:
                    background = canvas.copy_from_bbox(ax.bbox)
                canvas.restore_region(background)

                # Adjusting Y scale
                minY = min(min(buffers[0:])) - 100
                maxY = max(max(buffers[0:])) + 100
                plt.ylim([minY,maxY])
                
                # Plot refreshes with new buffer
                [lines[i].set_ydata(buffers[i]) for i in xrange(plotsNum)]

                plt.draw()
                plt.pause(0.000001)
                gevent.sleep(1)
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

        print headset.old_model
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
