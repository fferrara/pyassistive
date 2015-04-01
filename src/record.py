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
import winsound
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
    def __init__(self, emotiv, filename):
        self.isRunning = False
        self.isRecording = False

        self.sensors = np.array(['F3','P7','O1','O2','P8','F4'])
        self.PLOT_MIN_Y = 0
        self.PLOT_MAX_Y = 1000

        #### PROTOCOL DEFINITION ####
        self.ITERATIONS = config.RECORDING_ITERATIONS
        self.PERIOD = config.RECORDING_PERIOD # Recording stimulated SSVEP
        self.PAUSE_INTER_RECORDING = 2
        self.STIMULI_PATH = config.STIMULI_PATH
        self.DATA_PATH = config.DATA_PATH
        self.FILENAME = filename
        self.LOW_FREQ = 1
        self.NUM_STIMULI = 3

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
                buf = np.zeros((config.FS, len(self.sensors)))
                for i in range(len(buf)):
                    packet = self.headset.dequeue()
                    values = [packet.sensors[name]['value'] for name in self.sensors]
                    buf[i] = np.array(values)

                gevent.sleep(0) # need cause recording could be over
                if self.plotQueue is not None:
                    self.plotQueue.put_nowait(buf)
                if self.recorderQueue is not None and self.isRecording:
                    self.recorderQueue.put_nowait(buf)
                
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
        # if self.LOW_FREQ:
        #     TOP = [os.path.join(self.STIMULI_PATH, exe) for exe in os.listdir(self.STIMULI_PATH) if exe.endswith("64.exe")]
        #     SX = [os.path.join(self.STIMULI_PATH, exe) for exe in os.listdir(self.STIMULI_PATH) if exe.endswith("69.exe")]
        #     DX = [os.path.join(self.STIMULI_PATH, exe) for exe in os.listdir(self.STIMULI_PATH) if exe.endswith("80.exe")]
        # else:
        #     TOP = [os.path.join(self.STIMULI_PATH, exe) for exe in os.listdir(self.STIMULI_PATH) if exe.endswith("12.exe")]
        #     SX = [os.path.join(self.STIMULI_PATH, exe) for exe in os.listdir(self.STIMULI_PATH) if exe.endswith("13.exe")]
        #     DX = [os.path.join(self.STIMULI_PATH, exe) for exe in os.listdir(self.STIMULI_PATH) if exe.endswith("15.exe")]
        #
        # TOPwindow = Popen(args=TOP)
        # SXwindow = Popen(args=SX)
        # DXwindow = Popen(args=DX)

        stimuliExe = os.path.join(self.STIMULI_PATH, "stimuli_all.exe")
        stimuliWin = Popen(args=stimuliExe)
        gevent.sleep(5)
        try:
            for i in xrange(self.ITERATIONS):
                # TOP
                winsound.Beep(1500, 250)
                for i in xrange(self.PAUSE_INTER_RECORDING):
                    print ('Seconds to record TOP: %i' % (self.PAUSE_INTER_RECORDING - i))
                    gevent.sleep(1)

                print ('Start recording TOP')
                self.isRecording = True
                gevent.sleep(self.PERIOD)
                
                self.isRecording = False
                print ('Stop recording TOP')

                # SX
                winsound.Beep(1500, 250)
                for i in xrange(self.PAUSE_INTER_RECORDING):
                    print ('Seconds to record SX: %i' % (self.PAUSE_INTER_RECORDING - i))
                    gevent.sleep(1)

                print ('Start recording SX')
                self.isRecording = True
                gevent.sleep(self.PERIOD)
                
                self.isRecording = False
                print ('Stop recording SX')
                
                # DX
                winsound.Beep(1500, 250)
                for i in xrange(self.PAUSE_INTER_RECORDING):
                    print ('Seconds to record DX: %i' % (self.PAUSE_INTER_RECORDING - i))
                    gevent.sleep(1)
                
                print ('Start recording DX')
                self.isRecording = True
                gevent.sleep(self.PERIOD)

                self.isRecording = False
                print ('Stop recording DX')
        except Exception as e:
            print ('Controller error: %s' % e)
            self.isRunning = False
        finally:
            # if TOPwindow is not None:
            #     TOPwindow.kill()
            # if SXwindow is not None:
            #     SXwindow.kill()
            # if DXwindow is not None:
            #     DXwindow.kill()
            if stimuliWin is not None:
                stimuliWin.kill()
            print ('Controller over')
            self.isRunning = False

    def recorder(self):
        """
        Greenlet that store data read from the headset into a numpy array
        """
        data = np.empty( (self.ITERATIONS * self.PERIOD * self.NUM_STIMULI, config.FS, len(self.sensors)) )
        counter = 0
        try:
            while self.isRunning or not self.recorderQueue.empty():
                # Controller greenlets controls the recording
                while self.isRecording or not self.recorderQueue.empty():
                    while not self.recorderQueue.empty():
                        buf = self.recorderQueue.get()

                        data[counter] = buf
                        counter += 1

                    gevent.sleep(1)
                gevent.sleep(0)
        except Exception as e:
            print ('Recorder error: %s' % e)
            self.isRunning = False
        finally:
            print ('Recorder over')
            data = data.reshape((self.ITERATIONS * self.PERIOD * self.NUM_STIMULI * config.FS, len(self.sensors)))
            sio.savemat(os.path.join(self.DATA_PATH, self.FILENAME), {'X' : data})
            self.isRunning = False

    def plot(self, bufferSize = 1000):
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
                    for j in range(len(values)):
                        [buffers[i].appendleft(values[j, i]) for i in xrange(plotsNum)]
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

    # Create a Greenlet for the setup...
    setupGLet = gevent.spawn(headset.setup)
    gevent.sleep(1) # ... and pass the control

    if not setupGLet.started:
        print 'Error: Is the device connected?'
        headset.close()
        quit()

        print headset.old_model
    
    if args.command[0] == 'plot':
        recorder = Recorder(headset, None)
        recorder.isRunning = True
        # Create the sensor data generator
        g1 = gevent.spawn(recorder.get_sensors_info)
        # Create plot routine
        g2 = gevent.spawn(recorder.plot)
        # Kill both at Ctrl+C
        gevent.signal(signal.SIGINT, gevent.killall, [g1, g2])
        # Run them until termination
        gevent.joinall([g1, g2])
    elif args.command[0] == 'record':
        filename = raw_input('File name: ')
        recorder = Recorder(headset, filename)
        recorder.isRunning = True
        # Create the sensor data generator
        g1 = gevent.spawn(recorder.get_sensors_info)
        # Create plot routine
        g2 = gevent.spawn(recorder.plot)
        # Create recorder and controller routines
        g3 = gevent.spawn(recorder.recorder)
        g4 = gevent.spawn(recorder.controller)
        # Kill all at Ctrl+C
        gevent.signal(signal.SIGINT, gevent.killall, [g1, g2, g3, g4])
        # Run them until termination
        try:
            gevent.joinall([g1, g2, g3, g4])
        except KeyboardInterrupt:
            pass
