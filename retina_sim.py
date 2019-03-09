import numpy as np
import matplotlib.pyplot as plt

'''
Just sketching this out right now. Ideally want to have base Classes for
stimuli and cells, which are inherited by the Classes for specific stimuli and
cells (e.g. Bars/Circles and Alphas/DSGCs).

Need to decide the level of complexity of cell state simulation. Actually model
a Vm, and have Ca++ channels that respond to it? Or just model the 'activation'
or 'signal' of the cell that goes up and down with +ve/-ve stimuli?

See learning/neuro folder for simple biophysical cell simulations.
'''


class Stim(object):
    def __init__(self, startPos=(0, 0), startTime=0, vel=0, amp=1):
        self.startPos = startPos
        self.startTime = startTime
        self.vel = vel
        self.amp = amp


class Cell(object):
    def __init__(self, pos=0, diam=10, rf=50):
        self.pos = pos
        self.diam = diam  # soma diameter
        self.rf = rf  # receptive field diameter
        self.Vm = 0

    def decay(self):
        'Decay of activation that occurs with each timestep.'
        self.Vm = np.max(0, self.Vm-0)

    def check(self, stims):
        for stim in stims:
            # if stim is in RF, do corresponding response (exc/inh)
            pass
