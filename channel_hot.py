import numpy as np


def channel_hot_encode(masks, video):
    "Take shape (TxHxW) and encode to (TxCxHxW) with list of cluster masks."
    return np.stack([video*mask for mask in masks], axis=1)


if __name__ == '__main__':
    masks = [np.random.randn(350, 350)*i for i in range(5)]
    video = np.random.randn(60, 350, 350)
    test = channel_hot_encode(masks, video)
    print('input shape:', video.shape)
    print('number of masks:', len(masks))
    print('output shape:', test.shape)
