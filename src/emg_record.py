__author__ = 'iena'

import winsound
import os
import time
from util import config, emotiv_engine

if __name__ == '__main__':
    filename = raw_input('File name: ')
    engine = emotiv_engine.EmotivEngine(os.path.join(config.DATA_PATH, filename))

    winsound.Beep(1500, 250)
    print 'Recording OLHAR ESQUERDA...'
    timestamp = 0.0
    while True:
        info = engine.get_expressiv_info()
        if info is None:
            continue
        timestamp = round(info['timestamp'], 1)
        # print timestamp
        # look left
        if timestamp in [80.0, 80.1]:
            print 'Recording OLHAR ESQUERDA...'
            winsound.Beep(1500, 250)

        # eyebrow
        if timestamp in [20.0,20.1,100.0,100.1]:
            print 'Recording LEVANTAR SOBRANCELHA...'
            winsound.Beep(1500, 250)

        # look right
        if timestamp in [40.0, 40.1, 120.0, 120.1]:
            print 'Recording OLHAR DIREITA...'
            winsound.Beep(1500, 250)

        # clench
        if timestamp in [60.0, 60.1, 140.0, 140.1]:
            print 'Recording APERTAR DENTES...'
            winsound.Beep(1500, 250)

        if timestamp in [159.9,160.0, 160.1]:
            print 'OVER'
            break
        time.sleep(0.1)