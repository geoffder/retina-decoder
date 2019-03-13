import numpy as np
import matplotlib.pyplot as plt

from sim_util import rotate, StackPlotter

'''
Just sketching this out right now. Ideally want to have base Classes for
stimuli and cells, which are inherited by the Classes for specific stimuli and
cells (e.g. Bars/Circles and Alphas/DSGCs).

Need to decide the level of complexity of cell state simulation. Actually model
a Vm, and have Ca++ channels that respond to it? Or just model the 'activation'
or 'signal' of the cell that goes up and down with +ve/-ve stimuli?

See learning/neuro folder for simple biophysical cell simulations.
'''


class NetworkModel(object):
    def __init__(self, tstop=500, dt=1, dims=[400, 400]):
        self.tstop = tstop
        self.dt = dt
        self.dims = dims
        self.origin = (dims[0]//2, dims[1]//2)
        self.t = 0
        self.stims = []

    def populate(self, pop=1):
        self.cells = []
        for i in range(pop):
            pos = [np.random.randint(0, dim+1) for dim in self.dims]
            self.cells.append(Cell(self, pos=pos))

    def newStim(self, type='bar', startPos=[0, 0], tOn=0, tOff=None,
                vel=0, theta=0, orient=0, amp=1, change=0, width=10,
                length=100, radius=100):

        stim = Stim(
            self, type=type, startPos=startPos, tOn=tOn, tOff=tOff,
            vel=vel, theta=0, orient=0, amp=1, change=0, width=width,
            length=length, radius=radius
        )

        self.stims.append(stim)

    def step(self):
        for stim in self.stims:
            stim.move()
            for cell in self.cells:
                cell.excite(stim.check(cell.rfMask))
        for cell in self.cells:
            cell.decay()
        self.t += self.dt

    def run(self):
        for t in range(self.tstop//self.dt):
            self.step()

    def plotCells(self):
        net = np.zeros(self.dims)
        for cell in self.cells:
            net += cell.somaMask*1.
            net += cell.rfMask*.2
        fig = plt.imshow(net)
        return fig

    def plotStims(self):
        movies = []
        for stim in self.stims:

            movies.append(np.dstack(stim.rec))
        # movie = np.sum(movies, keepdims=True)
        fig, ax = plt.subplots(1)
        stack = StackPlotter(ax, movies[0], delta=10)
        fig.canvas.mpl_connect('scroll_event', stack.onscroll)
        return fig, ax, stack

    def plotRecs(self):
        fig, axes = plt.subplots(len(self.cells))
        for i, cell in enumerate(self.cells):
            axes[i].plot(np.arange(self.tstop)*self.dt, cell.rec)
        return fig, axes

    def plotExperiment(self):
        self.plotCells()
        stimFig, stimAm, stimStack = self.plotStims()
        self.plotRecs()
        plt.show()


class Stim(object):
    def __init__(self, model, type='bar', startPos=[0, 0], tOn=0, tOff=None,
                 vel=0, theta=0, orient=0, amp=1, change=0, width=10,
                 length=100, radius=100):
        self.model = model  # the network model this stim belongs to
        # basics
        self.type = type
        self.startPos = startPos
        self.pos = startPos
        self.tOn = tOn
        self.tOff = model.tstop if tOff is None else tOff
        self.rec = []  # store masks from each timestep
        # movement
        self.vel = vel  # velocity
        self.theta = theta  # direction of motion
        self.orient = orient  # orientational rotation
        # "contrast"
        self.amp = amp  # -1 to 1 (black to white)
        self.change = change  # rate (and sign/direction) of amplitude change
        # bar
        self.width = width
        self.length = length
        # circle
        self.radius = radius

    def move(self):
        self.pos[0] += self.vel/self.model.dt * np.cos(np.deg2rad(self.theta))
        self.pos[1] += self.vel/self.model.dt * np.sin(np.deg2rad(self.theta))
        self.drawMask()
        self.rec.append(self.mask)

    def drawMask(self):
        if self.type == 'bar':
            x, y = np.ogrid[:self.model.dims[0], :self.model.dims[1]]
            x, y = rotate(self.model.origin, x, y, self.orient)
            self.mask = (
                (np.abs(x-self.pos[0]) <= self.width)
                * (np.abs(y-self.pos[1]) <= self.length)
            )
        elif self.type == 'circle':
            x, y = np.ogrid[:self.model.dims[0], :self.model.dims[1]]
            cx, cy = self.pos  # centre coordniates
            # convert cartesian --> polar coordinates
            r2 = (x - cx)**2 + (y - cy)**2
            # circular mask
            self.mask = r2 <= self.radius**2

    def check(self, rfMask):
        return np.sum(self.mask*rfMask) > 0  # return if stim is in RF at all


class Cell(object):
    def __init__(self, model, pos=[0, 0], diam=10, rf=50, dt=1):
        self.model = model  # the network model this cell belongs to
        self.pos = pos
        self.diam = diam  # soma diameter
        self.somaMask = self.drawMask(diam//2)
        self.rf = rf  # receptive field radius
        self.rfMask = self.drawMask(rf)
        self.Vm = 0
        self.dtau = 10  # decay tau
        self.rec = []

    def drawMask(self, radius):
        x, y = np.ogrid[:self.model.dims[0], :self.model.dims[1]]
        cx, cy = self.pos  # centre coordniates
        # convert cartesian --> polar coordinates
        r2 = (x - cx)**2 + (y - cy)**2
        # circular mask
        mask = r2 <= radius**2
        return mask

    def decay(self):
        'Decay of activation that occurs with each timestep.'
        delta = self.Vm * (1 - np.exp(-self.model.dt/self.dtau))
        self.Vm = np.max([0, self.Vm - delta])
        self.rec.append(self.Vm)  # last thing each time step, so record Vm

    def excite(self, strength):
        self.Vm += strength
