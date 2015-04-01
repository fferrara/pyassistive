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
from jsonrpctcp import client
from socket import error as socket_error
import winsound
from emokit.emotiv import Emotiv
from util import config, preprocessing, featex

SERIAL = "SN201308221848GM" # only necessary for OSX

class Recorder(object):
    """
    Recorder class. Read data from Emotiv and process the signal.
    Although not implemented, Recorder should be a Singleton.
    """
    def __init__(self, emotiv):
        self.isRunning = False
        self.isRecording = False

        self.sensors = np.array(['F3','P7','O1','O2','P8','F4'])
        self.headset = emotiv
        self.dataQueue = Queue()

        self.host = 'localhost'
        self.port = 8888
        self.commandClient = client.connect(self.host, self.port)
        self.commands = [6.4, 8, 6.9]

        # processing parameters
        self.freqs = [6.4, 6.9, 8]
        self.winSize = 6
        self.method = featex.MSI(list(self.freqs), (self.winSize - 1) * config.FS, config.FS)

    def _process_window(self, win, step=4):
        subSamples = (self.winSize - 1) * config.FS
        subWindows = preprocessing.sliding_window(
            win, (subSamples, win.shape[1]), (np.floor(config.FS / step), win.shape[1]))

        temp = []
        for subWindow in subWindows:
            temp.append(self.method.perform(subWindow))

        # computing classification "confidence"
        for fIndex in range(len(self.freqs)):
            if float(temp.count(self.freqs[fIndex])) / len(temp) > 0.8:
                return self.freqs[fIndex]
                break   
        else:
            return None

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
                    window = np.empty((self.winSize, config.FS, len(self.sensors)))
                    for counter in range(self.winSize):
                        buf = self.dataQueue.get()
                        window[counter] = buf

                    window = window.reshape(self.winSize * config.FS, len(self.sensors))
                    print 'Window collected'
                    # processing time window
                    out = self._process_window(window)
                    while out is None:
                        print 'Sliding...'
                        if not self.dataQueue.empty():
                            # slide current window
                            buf = self.dataQueue.get()
                            # concatenate last winSize-1 seconds with last second
                            window = np.vstack((window[config.FS:,:], buf))
                            out = self._process_window(window)

                        gevent.sleep(0)

                    print 'Comando!' + str(out)
                    self.send_receive_command(self.commands.index(out))
                    gevent.sleep(2)
                gevent.sleep(0)
        except Exception as e:
            print ('Recorder error: %s' % e)
            self.isRunning = False
        finally:
            print ('Recorder over')
            self.isRunning = False

    def send_receive_command(self, cmd):
            try:
                returned = self.commandClient.command(cmd)
                # if returned is not None and returned == 'OFF':
                #     self.isPaused = True
                #     print '# Commands %d, Average time for command %0.2f' % (controller.cmds, controller.T / controller.cmds)
            except socket_error:
                print 'Socket Error while communicating with the Control Interface'
                print 'Is the Interface running? Check host and port settings.'

if __name__ == '__main__':
    # Creating headset Object. No console output. No research headset
    headset = Emotiv(False, SERIAL, False)

    # Create a Greenlet for the setup...
    setupGLet = gevent.spawn(headset.setup)
    gevent.sleep(1) # ... and pass the control

    if not setupGLet.started:
        print 'Error: Is the device connected?'
        headset.close()
        quit()

    recorder = Recorder(headset)
    recorder.isRunning = True
    recorder.isRecording = True
    # Create the sensor data generator
    g1 = gevent.spawn(recorder.get_sensors_info)
    # Create recorder and controller routines
    g2 = gevent.spawn(recorder.process_data)
    # Kill all at Ctrl+C
    gevent.signal(signal.SIGINT, gevent.killall, [g1, g2])
    # Run them until termination
    try:
        gevent.joinall([g1, g2])
    except KeyboardInterrupt:
        pass