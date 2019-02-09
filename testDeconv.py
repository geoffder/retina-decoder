import pandas as pd
import matplotlib.pyplot as plt
import cnmf_deconvolution as deconv
from scipy import signal

df = pd.read_csv('origROI.csv')
wave = df.values.flatten()

# p = order of autoregression, options are 1 and 2. Indicates how many
# preceding data-points are used at each (time)step of auto-regression
# method_deconvolution: 'oasis' or 'cvxpy'
# to use oasis, compile cnmf.oasis
c, _, _, _, _, sp, _ = deconv.constrained_foopsi(
                        wave, p=1, method_deconvolution='oasis')

filt = signal.savgol_filter(wave, 13, 2)

fig, axes = plt.subplots(5, 1, figsize=(15, 9))
axes[0].plot(wave)
axes[0].set_title('raw')
axes[1].plot(filt)
axes[1].set_title('savgol')
axes[2].plot(c)
axes[2].set_title('de-noised')
axes[3].plot(wave - wave.mean())
# axes[3].plot(c)
axes[3].plot(c - c.mean())
axes[3].set_title('raw + de-noised')
axes[4].plot(wave)
axes[4].plot(filt)
axes[4].set_title('raw + savgol')
fig.tight_layout()
plt.show()

# plt.plot(sp)
# plt.show()
