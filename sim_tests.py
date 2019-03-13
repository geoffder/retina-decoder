import matplotlib.pyplot as plt
from retina_sim import NetworkModel, Cell, Stim


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


def testRun():
    model = NetworkModel()
    model.populate(pop=10)
    model.newStim(type='bar', width=10, length=400, vel=1, startPos=[0, 50])
    model.run()
    model.plotExperiment()
