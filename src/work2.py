import msvcrt
import time
import winsound
import sys

if __name__ == '__main__':
    commands = {'C' : 'Clench', 'B': 'Eyebrow', 'E' : 'Eye movement'}
    errors = 0
    undefined = 0

    cmd = raw_input('Command to test: ')
    cmd = cmd.upper()
    T = 0
    i = 0
    while i < 30:
        try:
            print 'Expecting char %d...' % i
            t = time.time()
            c1 = msvcrt.getch()
            print 'First: ', c1
            if c1 == '\x03': # ctrl+c
                break

            # c2 = msvcrt.getch()
            # print 'Second: ', c2
            if (time.time() - t) > 5:
                undefined += 1
                continue

            key = c1.upper()
            print commands[key]
            winsound.Beep(1500, 250)
            T += (time.time() - t)
            if cmd != key:
                errors += 1

            time.sleep(5)
            i+=1
            while msvcrt.kbhit(): # clear buffer
                msvcrt.getch()

            time.sleep(0.1)
        except KeyboardInterrupt:
            break

    # print 'Errors: %d, Undefined: %d' % (errors, undefined)
    print 'T: ', T / 20.