from torch import nn


class DecoderLoss(nn.Module):
    """
    Experimental. Idea is to harshly penalize safe (mean) predictions and
    encourage the model to try to move it's output towards the non-zero
    stimulus values, rather than staying near zero always to minimize loss
    (since most of the video is zero).
    """
    def __init__(self, alpha=1, decay=1):
        super(DecoderLoss, self).__init__()
        self.alpha = alpha
        self.decay_rate = decay

    def decay(self):
        self.alpha *= self.decay_rate

    def forward(self, decoding, targets):
        error = (decoding - targets).pow(2) * (1 + self.alpha*targets.abs())
        return error.mean()
