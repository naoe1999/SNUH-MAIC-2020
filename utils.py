import numpy as np


def moving_average_padding(a, n=200):
    lpad = (n - 1) // 2
    rpad = n - 1 - lpad
    pw = [(0, 0) for _ in range(len(a.shape)-1)] + [(lpad, rpad)]
    a_pad = np.pad(a, pw, mode='mean')

    ret = np.nancumsum(a_pad, axis=-1, dtype=np.float32)
    ret[..., n:] = ret[..., n:] - ret[..., :-n]
    return ret[..., n - 1:] / n


def preprocess_data(x):

    x_info = x[:, :4]
    x_data = x[:, 4:]

    x_age = (x_info[:, 0] - 20.) / 60.      # age
    x_sex = (x_info[:, 1] - 0.) / 1.        # sex
    x_wgt = (x_info[:, 2] - 40.) / 60.      # weight
    x_hgt = (x_info[:, 3] - 130.) / 60.     # height
    x_info = np.stack([x_age, x_sex, x_wgt, x_hgt], axis=-1)

    x_data = (x_data - 65.) / 65.
    x_data_1s = moving_average_padding(x_data, 99)
    x_data_2s = moving_average_padding(x_data, 199)
    x_data_4s = moving_average_padding(x_data, 399)

    x_data = np.stack([x_data, x_data_1s, x_data_2s, x_data_4s], axis=-1)

    return x_info, x_data

