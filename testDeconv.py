import pandas as pd
import matplotlib.pyplot as plt
import cnmf_deconvolution as deconv

df = pd.read_csv('origROI.csv')
wave = df.values.flatten()

# p = order of autoregression, options are 1 and 2. Indicates how many
# preceding data-points are used at each (time)step of auto-regression
c, _, _, _, _, _, _ = deconv.constrained_foopsi(wave, p=1)

fig, axes = plt.subplots(3, 1, figsize=(15, 5))
axes[0].plot(wave)
axes[0].set_title('raw')
axes[1].plot(c)
axes[1].set_title('de-noised')
axes[2].plot(wave - wave.mean())
# axes[2].plot(c)
axes[2].plot(c - c.mean())
axes[2].set_title('together')
fig.tight_layout()
plt.show()
