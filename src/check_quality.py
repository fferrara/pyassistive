#!/usr/bin/env python
""" 
Utility for checking contact qualities. 
It prints the quality field for each sensor read by Emotiv headset.
"""

import platform
if platform.system() == "Windows":
    import socket  # Needed to prevent gevent crashing on Windows. (surfly / gevent issue #459)
import gevent
import signal
import os
from emokit.emotiv import Emotiv

SERIAL = "SN201308221848GM" # only necessary for OSX

def printQuality(headset):
    try:
        while True:
            packet = headset.dequeue()
            if platform.system() == "Windows":
                os.system('cls')
            else:
                os.system('clear')

            print('\n'.join("%s Quality: %s" %
                                (k[1], packet.sensors[k[1]]['quality']) for k in enumerate(packet.sensors)))
            print "Battery: %i" % packet.battery
            gevent.sleep(.001)
    except KeyboardInterrupt:
        print "Stopped"
    except:
    	print "Error"
    finally:
        headset.close()

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

    g1 = gevent.spawn(printQuality, headset)
    g1.join()
    