import numpy as np
# from scipy import sparse
import matplotlib.pyplot as plt
from sim_util import rotate, StackPlotter
import time as timer

"""
Need to decide the level of complexity of cell state simulation. Actually model
a Vm, and have Ca++ channels that respond to it? Or just model the 'activation'
or 'signal' of the cell that goes up and down with +ve/-ve stimuli?
-> See learning/neuro folder for simple biophysical cell simulations.
"""


class NetworkModel(object):
    def __init__(self, tstop=500, dt=1, dims=[600, 600]):
        self.tstop = tstop  # run duration / stop time
        self.dt = dt  # timestep
        self.dims = dims  # dimensions of network matrix
        self.margin = 100  # no cells allowed within this distance of edge
        self.origin = (dims[0]//2, dims[1]//2)
        self.t = 0  # current time
        self.runs = 0  # number of completed 'runs'
        self.stims = []
        self.stimMovies = []
        self.timers = np.zeros(3)

    def populate(self, pop=None, spacing=None, jitter=1):
        start = timer.time()
        """
        Use pop to set a cell population number and uniformly distribute cells
        within the network window, or use spacing with jitter to place cells in
        a noisy grid / 'mosaic'.
        """
        self.cells = []
        if pop is not None:
            for i in range(pop):
                pos = [np.random.randint(self.margin, dim-self.margin+1)
                       for dim in self.dims]
                self.cells.append(Cell(self.dims, self.dt, pos=pos))
        elif spacing is not None:
            xspace, yspace = [np.arange(self.margin, dim-self.margin, spacing)
                              for dim in self.dims]
            for i in range(xspace.shape[0]):
                for j in range(yspace.shape[0]):
                    theta = 2 * np.pi * np.random.random()
                    radius = np.random.randn()*jitter
                    pos = [xspace[i] + radius * np.cos(np.deg2rad(theta)),
                           yspace[j] + radius * np.sin(np.deg2rad(theta))]
                    self.cells.append(Cell(self.dims, self.dt, pos=pos))
        else:
            print("No cell density params given. Crashing...")
        print("time to populate:", timer.time()-start)

    def newStim(self, type='bar', startPos=[0, 0], tOn=0, tOff=None,
                vel=0, theta=0, orient=0, amp=1, change=0, width=10,
                length=100, radius=100):
        """
        Generate a new stimulus object for this model. Multiple independent
        stimuli are possible, simply generate each with this function before
        executing run(). A movie will be made with the sums of each stimulus
        mask for viewing and storage.
        """
        stim = Stim(
            self.dims, self.dt, type=type, startPos=startPos, tOn=tOn,
            tOff=tOff, vel=vel, theta=theta, orient=orient, amp=amp,
            change=change, width=width, length=length, radius=radius
        )

        self.stims.append(stim)

    def storeStimMov(self):
        """
        Add together all stims in to one movie and store in a list. Useful
        for experiments with mutliple stimulus presentations.
        """
        movie = np.zeros((*self.dims, self.tstop//self.dt))
        for stim in self.stims:
            movie += np.dstack(stim.rec)
        self.stimMovies.append(movie.astype(np.uint8))

    def clearStims(self):
        self.stims = []

    def step(self):
        """
        Run through a model time-step. First updating positions of stimuli,
        then checking for interactions with each of the cells. Cells also go
        through updates, such as Vm/activation decay.
        """
        for stim in self.stims:
            start = timer.time()
            stim.move()
            self.timers[0] += timer.time() - start
            start = timer.time()
            for cell in self.cells:
                cell.excite(stim.check(cell.rfMask))
                # cell.excite(stim.check(cell.rfMask_sparse))
            self.timers[1] += timer.time() - start
        start = timer.time()
        for cell in self.cells:
            cell.decay()
        self.timers[2] += timer.time() - start
        self.t += self.dt

    def run(self):
        self.timers *= 0
        start = timer.time()
        "Run through from t=0 to t=tstop and store data in lists."
        self.t = 0
        # step through experiment until tstop
        for t in range(self.tstop//self.dt):
            self.step()
        # store recordings from cells, and the presented stim for this run
        for cell in self.cells:
            cell.storeRec()
            cell.reset()
        self.storeStimMov()
        self.runs += 1
        print('total run time:', timer.time()-start)
        print('moving time:', self.timers[0], 'excite time:', self.timers[1],
              'decay time:', self.timers[2])

    def plotCells(self):
        "Plot map of cells and their receptive fields."
        net = np.zeros(self.dims)
        for cell in self.cells:
            net += cell.somaMask*1.
            net += cell.rfMask*.2
        fig, ax = plt.subplots(1)
        ax.imshow(net)
        for i, cell in enumerate(self.cells):
            ax.scatter(cell.pos[1], cell.pos[0], c='r', alpha=.5, s=80,
                       marker='$%s$' % i)
        return fig, ax

    def netMovie(self):
        """
        Take the recordings of each cell and build a movie of their activity
        as a network, analogous to somatic Ca++ recordings.
        """
        movie = np.zeros(
            (*self.dims, self.runs*(self.tstop//self.dt))).astype(np.float)
        x, y = np.ogrid[:self.dims[0], :self.dims[1]]
        for cell in self.cells:
            cx, cy = cell.pos
            r2 = (x - cx)**2 + (y - cy)**2
            r = (cell.diam/2)**2
            movie[r2 <= r, :] += np.concatenate(cell.recs, axis=0)

        # build the plot
        fig, ax = plt.subplots(1)
        stack = StackPlotter(ax, movie, delta=15)
        fig.canvas.mpl_connect('scroll_event', stack.onscroll)
        return fig, ax, stack

    def plotStims(self):
        "Plot stimulus movies."
        movie = np.dstack(self.stimMovies)
        fig, ax = plt.subplots(1)
        stack = StackPlotter(ax, movie, delta=50)
        fig.canvas.mpl_connect('scroll_event', stack.onscroll)
        return fig, ax, stack

    def plotRecs(self):
        "Plot recording of each cell for whole 'experiment'"
        nrows = 10
        ncols = len(self.cells)//nrows
        ncols += 1 if len(self.cells) % nrows > 0 else 0
        fig, axes = plt.subplots(nrows, ncols, figsize=(19, 9))
        for i, cell in enumerate(self.cells):
            rec = np.concatenate(cell.recs, axis=0)
            axes[i % nrows, i//nrows].plot(
                np.arange(rec.shape[0])*self.dt, rec)
            axes[i % nrows, i//nrows].set_ylabel('c%s' % i, rotation=0)
        fig.tight_layout()
        return fig, axes

    def plotExperiment(self):
        "Plot cell map, stimulus movie and cell recordings."
        cellFig, cellAx = self.plotCells()
        stimFig, stimAx, stimStack = self.plotStims()
        recFig, recAx = self.plotRecs()
        movFig, movAx, movStack = self.netMovie()
        plt.show()


class Stim(object):
    """
    Create stimulus objects (e.g. circles and bars). The mask of the stimulus
    will be saved into rec at each timestep.
    """
    def __init__(self, netDims, dt, type='bar', startPos=[0, 0], tOn=0,
                 tOff=None, vel=0, theta=0, orient=0, amp=1, change=0,
                 width=10, length=100, radius=100):
        self.netDims = netDims  # dims of the network model this cell is in
        self.dt = dt  # timestep of the network model
        # basics
        self.type = type  # string indicating stimulus type (e.g. circle)
        self.startPos = startPos  # starting position coordinates
        self.pos = startPos  # current position  coordinates
        self.tOn = tOn  # time stimulus appears
        self.tOff = tOff  # time stimulus turns off
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
        "Move the centre position of this stim as a function of its velocity."
        self.pos[0] += self.vel/self.dt * np.cos(np.deg2rad(self.theta))
        self.pos[1] += self.vel/self.dt * np.sin(np.deg2rad(self.theta))
        self.drawMask()
        # self.mask_sparse = sparse.csc_matrix(self.mask)
        self.rec.append(self.mask)

    def drawMask(self):
        "Draw shape with which this stimulus interacts with cells."
        if self.type == 'bar':
            x, y = np.ogrid[:self.netDims[0], :self.netDims[1]]
            # x, y = x.astype(np.float), y.astype(np.float)
            x, y = rotate(self.pos, x, y, np.radians(self.orient))
            self.mask = (
                (np.abs(x-self.pos[0]) <= self.width)
                * (np.abs(y-self.pos[1]) <= self.length)
            )
        elif self.type == 'circle':
            x, y = np.ogrid[:self.netDims[0], :self.netDims[1]]
            # x, y = x.astype(np.float), y.astype(np.float)
            cx, cy = self.pos  # centre coordniates
            # convert cartesian --> polar coordinates
            r2 = (x - cx)**2 + (y - cy)**2
            # circular mask
            self.mask = r2 <= self.radius**2

    def check(self, rfMask):
        """
        Takes recepetive field mask from cell and compares with the mask of
        this stimulus. If using bool masks, rather than int/float masks,
        np.count_nonzero is much faster (~6x) than np.sum when calculating
        overlap. However, this precludes the use of stimuli with masks that are
        non-boolean (e.g. gradients like sine gratings).
        """
        # return np.sum(self.mask*rfMask) > 0  # return if stim is in RF at all
        return np.count_nonzero(self.mask*rfMask) > 0
        # return self.mask_sparse.multiply(rfMask).count_nonzero()


class Cell(object):
    """
    Base class for cells, plan to make child Classes for different cell types
    that have different excite/inhib functions to allow different stimulus
    preferrences.
    """
    def __init__(self, netDims, dt, pos=[0, 0], diam=20, rf=50):
        self.netDims = netDims  # dims of the network model this cell is in
        self.dt = dt  # timestep of the network model
        self.pos = np.array(pos)  # centre coordinates (constant)
        self.diam = diam  # soma diameter
        self.somaMask = self.drawMask(diam//2)
        self.rf = rf  # receptive field radius
        self.rfMask = self.drawMask(rf)
        # self.rfMask_sparse = sparse.csc_matrix(self.rfMask)
        self.Vm = 0.
        self.dtau = 10  # decay tau
        self.rec = []
        self.recs = []

    def drawMask(self, radius):
        "Draws cell body and receptive field (for display and stimulation)."
        x, y = np.ogrid[:self.netDims[0], :self.netDims[1]]
        # x, y = x.astype(np.float), y.astype(np.float)
        cx, cy = self.pos  # centre coordniates
        # convert cartesian --> polar coordinates
        r2 = (x - cx)**2 + (y - cy)**2
        # circular mask
        mask = r2 <= radius**2
        return mask

    def decay(self):
        "Decay of activation that occurs with each timestep."
        delta = self.Vm * (1 - np.exp(-self.dt/self.dtau))
        self.Vm = np.max([0, self.Vm - delta])
        self.rec.append(self.Vm)  # last thing each time step, so record Vm

    def excite(self, strength):
        self.Vm += strength

    def storeRec(self):
        self.recs.append(self.rec)
        self.rec = []

    def reset(self):
        self.Vm = 0.
