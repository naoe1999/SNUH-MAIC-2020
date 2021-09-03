import numpy as np
import pandas as pd
import os


if __name__ == '__main__':

    files = [
        'pred_y_pilot27_cnn29_ensemble_top3_4791.txt',
        'pred_y_pilot35_cnn_branch_ep04_post_4882.txt',
        'pred_y_pilot93_tenet2_ep26_4735.txt',
        'pred_y_pilot115_avgnet_ep11_4849.txt',
        'pred_y_pilot186_iannet_ep18_4821.txt'
    ]

    preds = [pd.read_csv(os.path.join('./results', f), header=None).values.flatten() for f in files]

    gm = np.power(preds[0] * preds[1] * preds[2] * preds[3] * preds[4], 1./5.)

    with open('./results/pred_y_gmean_27_35_93_115_186.txt', 'wt') as fp:
        fp.writelines(['{0:.8E}\n'.format(v) for v in gm])

