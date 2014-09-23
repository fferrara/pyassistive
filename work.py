# This is an example of popping a packet from the Emotiv class's packet queue
# and printing the gyro x and y values to the console.

from emokit.emotiv import Emotiv
import platform
if platform.system() == "Windows":
    import socket  # Needed to prevent gevent crashing on Windows. (surfly / gevent issue #459)
import gevent
import time

SERIAL = "SN201308221848GM" # only necessary for OSX

if __name__ == "__main__":
    headset = Emotiv(False, SERIAL, False)
    gevent.spawn(headset.setup)
    gevent.sleep(0)
    try:
        startTime = time.clock()
        print "Starting at {:.2f}".format(startTime)
        while True:
            packet = headset.dequeue()
            #print "Packet n: {0}, value 01 : {1}".format(packet.packets_received, packet.sensors["01"]["value"])
            print "Value 01: ",packet.sensors["O1"]["value"]
            gevent.sleep(0)
    except KeyboardInterrupt:
        headset.close()
    finally:
        headset.close()

def print_info():
    """
        Greenlet that outputs info once per second
        """
