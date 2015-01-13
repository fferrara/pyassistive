""" Configuration file
"""
#!/usr/bin/env python
import os
import numpy as np

#### PROCESSING PARAMETERS ####
FS = 128
NFFT = 4096.
SENSORS = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
WCRITIC = np.array([0., 4., 5., 49., 50., FS / 2.])

#### PATH SETTINGS ####
MAIN_DIR = "C:/Users/iena/Documents/dev/pyassistive"
DATA_PATH = os.path.join(MAIN_DIR, 'data')
STIMULI_PATH = os.path.join(MAIN_DIR, 'stimuli')

#### PROTOCOL DEFINITION ####
RECORDING_ITERATIONS = 4
RECORDING_PERIOD = 20 # Recording stimulated SSVEP
PAUSE_INTER_RECORDING = 5

#### ONLINE PARAMETERS ####
WAIT_INTER_COMMANDS = 1

#### SANDRA PARAMETERS ####
SENSORS_SANDRA = ['P7', 'PO7', 'P5', 'PO3', 'POz', 'PO4', 'P6', 'PO8', 'P8', 'O1', 'O2', 'Oz']
SUBJECTS_SANDRA = ['alessandra', 'andre', 'carloslai', 'celso', 'daniboy', 'doney', 'fernando', 'gilmar', 'jorge', 'joselito', 'julio', 'lucas', 'lucio', 'luis', 'marcelo', 'matheus', 'willian']
