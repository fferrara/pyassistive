#!/usr/bin/env python
""" 
Utility for creating a bridge between Emotiv headset and MATLAB.
This script gathers sensor values from Emotiv and stacks them into a MATLAB matrix, one new row each second.

The name of the desired variable in MATLAB workspace can be passed as command line argument. The default name is 'data'
"""

import pymatlab
import os
import gevent
import time
import argparse
import sys
import signal
from gevent.queue import Queue
import numpy as np

from emokit.emotiv import Emotiv


SERIAL = "SN201308221848GM" # only necessary for OSX

class MatlabConnect(object):
    def __init__(self, headset, path = ''):
        self.headset = headset
        self.dataQueue = Queue()
        self.isRunning = False
        self.sensors = ['F3','FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4'] 
        
        command = os.path.join(path, 'matlab')
        os.system('%s -automation -desktop' % command)

    def get_sensors_info(self):
        """
            Greenlet to get a packet from Emotiv headset.
            Append new data to queues where consumers will read from
        """
        try:
            while self.isRunning:
                packet = self.headset.dequeue()
                values = [packet.sensors[name]['value'] for name in self.sensors]
                if self.dataQueue is not None:
                    self.dataQueue.put_nowait(values)
                
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

    def matlabBridge(self, varName):
        data = None
        while self.isRunning or not self.dataQueue.empty():
            while not self.dataQueue.empty():
                buf = self.dataQueue.get()

                if data is None:
                    data = np.array(buf, dtype=int)
                else:
                    data = np.vstack((data, buf))

            self.session.putvalue(varName, data)
            gevent.sleep(1)
        print 'Matlab over'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EEG signal recorder.')
    parser.add_argument('var', metavar='var', type=str, nargs='?')
    args = parser.parse_args(sys.argv[1:])

    # Creating headset Object. No console output. No research headset
    headset = Emotiv(False, SERIAL, False)
    connect = MatlabConnect(headset)

    # Matlab needs to be started the current workspace to be used
    raw_input("Press Enter when Matlab is started")
    connect.session = pymatlab.session_factory()

    # Create a Greenlet for the setup...
    setupGLet = gevent.spawn(headset.setup)
    gevent.sleep(1) # ... and pass the control

    if not setupGLet.started:
        print 'Error: Is the device connected?'
        headset.close()
        quit()

    connect.isRunning = True
    # Create the sensor data generator
    g1 = gevent.spawn(connect.get_sensors_info)
    # Create plot routine
    g2 = gevent.spawn(connect.matlabBridge, args.var or 'data')
    # Kill both at Ctrl+C
    gevent.signal(signal.SIGINT, gevent.killall, [g1, g2])
    # Run them until termination
    gevent.joinall([g1, g2])