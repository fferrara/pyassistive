import ctypes
import sys
import os

import time

libEDK = ctypes.cdll.LoadLibrary(".\\edk.dll")
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
EE_EmoEngineEventCreate = libEDK.EE_EmoEngineEventCreate
EE_EmoEngineEventCreate.restype = ctypes.c_void_p
eEvent = EE_EmoEngineEventCreate()

EE_EmoEngineEventGetEmoState = libEDK.EE_EmoEngineEventGetEmoState
EE_EmoEngineEventGetEmoState.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
EE_EmoEngineEventGetEmoState.restype = ctypes.c_int

ES_GetTimeFromStart = libEDK.ES_GetTimeFromStart
ES_GetTimeFromStart.argtypes = [ctypes.c_void_p]
ES_GetTimeFromStart.restype = ctypes.c_float

EE_EmoStateCreate = libEDK.EE_EmoStateCreate
EE_EmoStateCreate.restype = ctypes.c_void_p
eState = EE_EmoStateCreate()

ES_ExpressivIsBlink = libEDK.ES_ExpressivIsBlink
ES_ExpressivIsBlink.restype = ctypes.c_int
ES_ExpressivIsBlink.argtypes = [ctypes.c_void_p]

ES_ExpressivIsLeftWink = libEDK.ES_ExpressivIsLeftWink
ES_ExpressivIsLeftWink.restype = ctypes.c_int
ES_ExpressivIsLeftWink.argtypes = [ctypes.c_void_p]

ES_ExpressivIsRightWink = libEDK.ES_ExpressivIsRightWink
ES_ExpressivIsRightWink.restype = ctypes.c_int
ES_ExpressivIsRightWink.argtypes = [ctypes.c_void_p]

ES_ExpressivIsLookingLeft = libEDK.ES_ExpressivIsLookingLeft
ES_ExpressivIsLookingLeft.restype = ctypes.c_int
ES_ExpressivIsLookingLeft.argtypes = [ctypes.c_void_p]

ES_ExpressivIsLookingRight = libEDK.ES_ExpressivIsLookingRight
ES_ExpressivIsLookingRight.restype = ctypes.c_int
ES_ExpressivIsLookingRight.argtypes = [ctypes.c_void_p]

ES_ExpressivGetUpperFaceAction = libEDK.ES_ExpressivGetUpperFaceAction
ES_ExpressivGetUpperFaceAction.restype = ctypes.c_int
ES_ExpressivGetUpperFaceAction.argtypes = [ctypes.c_void_p]

ES_ExpressivGetUpperFaceActionPower = libEDK.ES_ExpressivGetUpperFaceActionPower
ES_ExpressivGetUpperFaceActionPower.restype = ctypes.c_float
ES_ExpressivGetUpperFaceActionPower.argtypes = [ctypes.c_void_p]

ES_ExpressivGetLowerFaceAction = libEDK.ES_ExpressivGetLowerFaceAction
ES_ExpressivGetLowerFaceAction.restype = ctypes.c_int
ES_ExpressivGetLowerFaceAction.argtypes = [ctypes.c_void_p]

ES_ExpressivGetLowerFaceActionPower = libEDK.ES_ExpressivGetLowerFaceActionPower
ES_ExpressivGetLowerFaceActionPower.restype = ctypes.c_float
ES_ExpressivGetLowerFaceActionPower.argtypes = [ctypes.c_void_p]

EXP_EYEBROW = 0x0020  # eyebrow
EXP_FURROW = 0x0040  # furrow
EXP_SMILE = 0x0080  # smile
EXP_CLENCH = 0x0100  # clench
EXP_SMIRK_LEFT = 0x0400  # smirk left
EXP_SMIRK_RIGHT = 0x0800  # smirk right
EXP_LAUGH = 0x0200  # laugh


class EmotivEngine(object):
    def __init__(self, logFile=None):
        if libEDK.EE_EngineConnect("Emotiv Systems-5") != 0:
            print "Emotiv Engine start up failed."
            exit()

        self.log = None
        if logFile is not None:
            self.log = open(logFile, 'w')

            self.header = ['timestamp','look_left','look_right','eyebrow','clench','smirk_left','smirk_right']
            self.log.write(','.join(self.header) + '\n')

        self.userID = ctypes.c_uint(0)
        self.user = ctypes.pointer(self.userID)
        self.composerPort = ctypes.c_uint(1726)
        self.timestamp = ctypes.c_float(0.0)
        self.option = ctypes.c_int(0)
        self.state = ctypes.c_int(0)
        self.eventType = None

    def __del__(self):
        if self.log is not None:
            self.log.close()
            libEDK.EE_EngineDisconnect()
            libEDK.EE_EmoStateFree(eState)
            libEDK.EE_EmoEngineEventFree(eEvent)

    def _get_state(self):
        self.state = libEDK.EE_EngineGetNextEvent(eEvent)
        self.eventType = 0
        if self.state == 0:
            self.eventType = libEDK.EE_EmoEngineEventGetType(eEvent)
            libEDK.EE_EmoEngineEventGetUserId(eEvent, self.user)
            if self.eventType == 64:  # libEDK.EE_Event_enum.EE_EmoStateUpdated
                libEDK.EE_EmoEngineEventGetEmoState(eEvent, eState)
                return eState
        elif self.state != 0x0600:
            print "Internal error in Emotiv Engine ! "

    def get_expressiv_info(self):
        info = {}
        state = self._get_state()

        if self.eventType == 64: # libEDK.EE_Event_enum.EE_EmoStateUpdated
            expressivStates = {}
            expressivStates[EXP_EYEBROW] = 0
            expressivStates[EXP_FURROW] = 0
            expressivStates[EXP_SMILE] = 0
            expressivStates[EXP_CLENCH] = 0
            expressivStates[EXP_SMIRK_LEFT] = 0
            expressivStates[EXP_SMIRK_RIGHT] = 0
            expressivStates[EXP_LAUGH] = 0
            upperFaceAction = ES_ExpressivGetUpperFaceAction(eState)
            upperFacePower = ES_ExpressivGetUpperFaceActionPower(eState)
            lowerFaceAction = ES_ExpressivGetLowerFaceAction(eState)
            lowerFacePower = ES_ExpressivGetLowerFaceActionPower(eState)
            expressivStates[upperFaceAction] = upperFacePower
            expressivStates[lowerFaceAction] = lowerFacePower

            info['timestamp'] = ES_GetTimeFromStart(eState)
            info['eyebrow'] = expressivStates[EXP_EYEBROW]
            info['smirk_left'] = expressivStates[EXP_SMIRK_LEFT]
            info['smirk_right'] = expressivStates[EXP_SMIRK_RIGHT]
            info['clench'] = expressivStates[EXP_CLENCH]
            info['wink_left'] = ES_ExpressivIsLeftWink(eState)
            info['wink_right'] = ES_ExpressivIsRightWink(eState)
            info['look_left'] = ES_ExpressivIsLookingLeft(eState)
            info['look_right'] = ES_ExpressivIsLookingRight(eState)

            if self.log is not None:
                line = []
                for key in self.header:
                    line.append(info[key])
                self.log.write(','.join([str(val) for val in line]) + '\n')
            return info