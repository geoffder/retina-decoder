import numpy as np


def channel_hot_encode(masks, video):
    video = video.transpose(2, 0, 1)  # move time to first dimension
    return np.stack([video*mask for mask in masks], axis=1)


if __name__ == '__main__':
    masks = [np.random.randn(350, 350)*i for i in range(5)]
    video = np.random.randn(350, 350, 60)
    test = channel_hot_encode(masks, video)
