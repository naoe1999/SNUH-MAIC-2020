from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import LeakyReLU
from utils import preprocess_data
import numpy as np
import os


DATA_DIR = './data'
SAVE_DIR = './output'


if __name__ == '__main__':

    weight_path = os.path.join(SAVE_DIR, 'weights_ep{ep:02d}.hdf5')
    json_path = os.path.join(SAVE_DIR, 'model.json')
    data_path = os.path.join(DATA_DIR, 'x_test.npz')

    # load model
    model = model_from_json(open(json_path, 'rt').read())   # , custom_objects={'LeakyReLU': LeakyReLU})
    model.compile()
    print(model.summary())

    # load data
    print('loading test data ... ', flush=True, end='')
    x_test = np.load(data_path)['arr_0']
    print('done.')
    print(f'number of test data: {len(x_test)}')

    # data preprocess
    print('preprocessing ... ', end='', flush=True)
    x_test_info, x_test_data = preprocess_data(x_test)
    x_test = None
    print('done.')

    # iterate for multiple networks
    for i, ep in enumerate(range(2, 30)):

        # load weights
        wp = weight_path.format(ep=ep)
        model.load_weights(wp)
        print(wp, 'loaded.')

        # predict
        y_out = model.predict(x={'input_data': x_test_data, 'input_info': x_test_info})

        y_out = y_out[:, -1, :]
        y_out = np.maximum(np.minimum(y_out, 1.0), 0.0)

        n_nan = np.sum(np.isnan(y_out))
        print(n_nan, 'nan occurred.')
        y_out[np.isnan(y_out)] = 0.0

        # save result
        idx = 170 + i
        np.savetxt(f'pred_y_pilot{idx}_iannet_ep{ep:02d}.txt', y_out.flatten())
        print('result saved.')

    print('done.')

