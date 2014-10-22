#!/usr/bin/env python
import os

FS = 128
NFFT = 4096.
SENSORS = ['F3','FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']

MAIN_DIR = "C:/Users/iena/Documents/dev/smartenv"
DATA_PATH = os.path.join(MAIN_DIR, 'data')
STIMULI_PATH = os.path.join(MAIN_DIR, 'stimuli')

#### PROTOCOL DEFINITION ####
RECORDING_ITERATIONS = 6
RECORDING_PERIOD = 20 # Recording stimulated SSVEP
PAUSE_INTER_RECORDING = 5

#### ONLINE PARAMETERS ####
WAIT_INTER_COMMANDS = 1
