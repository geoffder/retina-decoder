import numpy as np
import matplotlib.pyplot as plt


def ellipse(origin, ax0, ax1, phi, x, y):
    ox, oy = origin
    xp = (x-ox)*np.cos(phi) + (y-oy)*np.sin(phi)
    yp = -(x-ox)*np.sin(phi) + (y-oy)*np.cos(phi)
    return (xp/ax0)**2 + (yp/ax1)**2 <= 1


x, y = np.ogrid[:600, :600]
mask = ellipse([300, 300], 50, 300, np.radians(45), x, y)

plt.imshow(mask)
plt.show()
