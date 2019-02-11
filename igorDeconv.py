import numpy as np
import cnmf_deconvolution as deconv


def fit(wave):
    wave = np.array(wave)
    result, _, _, _, _, _, _ = deconv.constrained_foopsi(
                            wave, p=1, method_deconvolution='oasis')
    return result
