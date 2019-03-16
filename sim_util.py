import numpy as np
# import matplotlib.pyplot as plt

'''
Collection of utility classes and functions that are used by retina_sim, but
are not a direct part of the model itself.
'''


def rotate(origin, X, Y, angle):
    """
    Rotate a points (X[i],Y[i]) counterclockwise an angle around an origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    X, Y = np.array(X), np.array(Y)
    rotX = ox + np.cos(angle) * (X - ox) - np.sin(angle) * (Y - oy)
    rotY = oy + np.sin(angle) * (X - ox) + np.cos(angle) * (Y - oy)
    return rotX, rotY


class StackPlotter(object):
    '''
    Returns Object for cycling through frames of a 3D image stack using the
    mouse scroll wheel. Takes the pyplot axis object and data as the first two
    arguments. Additionally, use delta to set the number of frames each step of
    the wheel skips through.
    '''
    def __init__(self, ax, X, delta=10):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = 0
        self.delta = delta

        self.im = ax.imshow(self.X[:, :, self.ind], cmap='gray', vmin=0,
                            vmax=np.max(X))
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + self.delta) % self.slices
        else:
            self.ind = (self.ind - self.delta) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('t = %s' % self.ind)
        self.im.axes.figure.canvas.draw()
