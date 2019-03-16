import numpy as np
import matplotlib.pyplot as plt
from retina_sim import NetworkModel, Cell, Stim

# import time as timer

'''
Collection of functions for testing out components of retina_sim, and the
combination thereof into complete experiments.
'''


def testDecay():
    model = NetworkModel()
    cellA = Cell(model)
    cellA.Vm = 1000
    rec = [cellA.Vm]
    time = [model.t]
    for i in range(100):
        cellA.decay()
        model.t += model.dt
        rec.append(cellA.Vm)
        time.append(model.t)
    plt.plot(time, rec)
    plt.show()


def testRF():
    model = NetworkModel()
    cellA = Cell(model, pos=[100, 100])
    plt.imshow(cellA.rfMask)
    plt.show()


def testStimMask():
    model = NetworkModel()
    cellA = Cell(model, pos=[100, 100])
    bar = Stim(model, startPos=[200, 200], orient=45)
    bar.bar()
    bar.check(cellA.rfMask)


def testPop():
    model = NetworkModel()
    model.populate(spacing=40, jitter=10)
    model.plotCells()
    plt.show()


def testRun():
    model = NetworkModel()
    model.populate(pop=10)
    model.newStim(type='bar', theta=45, orient=-45, width=10, length=400,
                  vel=1, startPos=[200, 200])
    # model.newStim(type='circle', theta=45, radius=50, vel=1,
    #               startPos=[200, 200])
    model.run()
    model.plotExperiment()


def dirRun():
    model = NetworkModel(tstop=600)
    model.populate(spacing=40, jitter=10)
    dirs = [0, 45, 90, 135, 180, 225, 270, 315]
    # dirs = [0]
    cx, cy = model.origin
    print("Running... \ndirs: ", end='')
    for d in dirs:
        print(d, end=' ')
        pos = [cx - cx*np.cos(np.deg2rad(d)), cy - cy*np.sin(np.deg2rad(d))]
        # model.newStim(type='circle', theta=d, radius=50, vel=1,
        #               startPos=pos)
        model.newStim(type='bar', theta=d, orient=-d, width=10, length=100,
                      vel=1, startPos=pos)
        model.run()
        model.clearStims()
    print('')  # next line
    model.plotExperiment()


if __name__ == '__main__':
    # testRun()
    dirRun()
    # testPop()
