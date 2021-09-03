from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, TruePositives, FalsePositives, FalseNegatives, Precision, Recall
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.utils import plot_model
from iannet import build_model, build_model_tenet, build_model_axnet, build_model_iannet
from custom_loss import binary_focal_loss, patient_loss, hypotension_loss
from utils import preprocess_data
import numpy as np
import pickle
import os


BATCH_SIZE = 512
SAVE_DIR = './output'


if __name__ == '__main__':
    print('loading training data ... ', end='', flush=True)
    x_train = np.load('./data/x_train.npz')['arr_0']
    y_train = np.load('./data/y_train.npz')['arr_0']
    print('done.')

    ## for test
    # idx_samples = np.random.choice(np.arange(len(x_train)), 100000, replace=False)
    # x_train = x_train[idx_samples]
    # y_train = y_train[idx_samples]

    n_samples = len(x_train)
    n_positive_seqs = len(np.where(y_train[..., -1] == 1)[0])
    n_negative_seqs = len(np.where(y_train[..., -1] == 0)[0])
    print(f'train sequences total: {n_samples}')
    print(f' positive hypotension: {n_positive_seqs}, ratio: {n_positive_seqs * 100 / n_samples :.2f}%')
    print(f' negative hypotension: {n_negative_seqs}, ratio: {n_negative_seqs * 100 / n_samples :.2f}%')

    # preprocess
    print('preprocessing ... ', end='', flush=True)
    x_train_info, x_train_data = preprocess_data(x_train)
    x_train = None
    print('done.')

    # for output files
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    weight_path = os.path.join(SAVE_DIR, 'weights_ep{epoch:02d}.hdf5')
    json_path = os.path.join(SAVE_DIR, 'model.json')
    jpg_path = os.path.join(SAVE_DIR, 'model.jpg')
    history_path = os.path.join(SAVE_DIR, 'history.pkl')

    # ianNet
    model = build_model_iannet()

    print(model.summary())
    open(json_path, 'wt').write(model.to_json())
    plot_model(model, to_file=jpg_path)

    optimizer = Adam(learning_rate=1e-3)

    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=[MeanAbsoluteError(name='MAE')]
    )

    # model.compile(
    #     optimizer=optimizer,
    #     loss=binary_focal_loss(),   # loss='binary_crossentropy',
    #     metrics=[
    #         'accuracy',
    #          AUC(curve='ROC', name='AUROC'),
    #          AUC(curve='PR', name='AUPRC'),
    #          TruePositives(thresholds=0.1, name='TP'),
    #          FalsePositives(thresholds=0.1, name='FP'),
    #          FalseNegatives(thresholds=0.1, name='FN'),
    #          Precision(thresholds=0.1, name='pre'),
    #          Recall(thresholds=0.1, name='rec')
    #     ]
    #     # loss={
    #     #     'po': patient_loss(),
    #     #     'ho': hypotension_loss(class_weight=(1., 10.))
    #     # },
    #     # loss_weights={'po': 1., 'ho': 2.},
    #     # metrics={
    #     #     'po': 'accuracy',
    #     #     'ho': AUC(curve='PR', name='mPR'),
    #     #     'fo': [
    #     #         AUC(curve='ROC', name='AUROC'),
    #     #         AUC(curve='PR', name='AUPRC'),
    #     #         TruePositives(thresholds=0.1, name='TP'),
    #     #         FalsePositives(thresholds=0.1, name='FP'),
    #     #         FalseNegatives(thresholds=0.1, name='FN'),
    #     #         Precision(thresholds=0.1, name='pre'),
    #     #         Recall(thresholds=0.1, name='rec')
    #     #     ]
    #     # }
    # )

    hist = model.fit(
        x={'input_data': x_train_data, 'input_info': x_train_info}, y=y_train,
        # class_weight={0: 1., 1: 1.},
        validation_split=0.1, batch_size=BATCH_SIZE, epochs=100,
        callbacks=[ModelCheckpoint(filepath=weight_path, verbose=1, save_best_only=False),
                   EarlyStopping(patience=30, verbose=1),
                   ReduceLROnPlateau(factor=0.2, patience=10, verbose=1)]
    )

    with open(history_path, 'wb') as fp:
        pickle.dump(hist.history, fp)

    print('training finished.')

