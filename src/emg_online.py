__author__ = 'iena'

import gevent
import signal
import msvcrt
import time
from jsonrpctcp import client
from socket import error as socket_error
from gevent.queue import Queue

class Controller(object):
    def __init__(self):
        self.isRunning = False
        self.isPaused = False
        self.T = 0
        self.cmds = 0

        # defined in global configuration of control interface
        self.COMMANDS = {'C': 0, 'B': 1, 'E': 2}
        self.host = 'localhost'
        self.port = 8888
        ###

        self.commandClient = client.connect(self.host, self.port)

        # For activating the control interface, need 3 clench
        self.clenchCounter = 0

    def wait_command(self):
        while self.isRunning:
            t = time.time()
            c1 = msvcrt.getch() # get command
            T = time.time() - t
            if c1 == '\x03': # ctrl+c
                self.isRunning = False

            key = c1.upper()

            if self.isPaused and key == 'C':  # clench
                self.clenchCounter += 1
                if self.clenchCounter == 3:
                    try:
                        print 'Activated'
                        self.commandClient.turn_on()
                        self.clenchCounter = 0
                        self.isPaused = False
                    except socket_error:
                        print 'Socket Error while communicating with the Control Interface'
                        print 'Is the Interface running? Check host and port settings.'
                else:
                    print 'Clench recognized'
            elif self.isPaused and key != 'C':
                self.clenchCounter = 0
            else:
                self.T += T
                self.cmds += 1

                # send command
                if self.COMMANDS.has_key(key):
                    self.send_receive_command(self.COMMANDS[key])

                # sleep for the user to rest
                time.sleep(2)
                # clear buffer
                while msvcrt.kbhit():
                    msvcrt.getch()
            time.sleep(0.2)  # technical

    def send_receive_command(self, cmd):
        if self.isRunning and not self.isPaused:
            try:
                returned = self.commandClient.command(cmd)
                if returned is not None and returned == 'OFF':
                    self.isPaused = True
                    print '# Commands %d, Average time for command %0.2f' % (controller.cmds, controller.T / controller.cmds)
            except socket_error:
                print 'Socket Error while communicating with the Control Interface'
                print 'Is the Interface running? Check host and port settings.'

if __name__ == '__main__':
    controller = Controller()
    controller.isRunning = True
    controller.isPaused = True

    # Create controller routine
    g1 = gevent.spawn(controller.wait_command())
    # Kill at Ctrl+C
    #gevent.signal(signal.SIGINT, gevent.killall, g1)
    # Run until termination
    try:
        g1.join()
    except KeyboardInterrupt:
        print 'Interrupted'
        pass
