import numpy as np
import pandas as pd
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model, load_model, model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPool1D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import auc, classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, f1_score, precision_recall_curve
from custom_loss import binary_focal_loss
import tensorflow as tf


def build_model(num_nodes):
    # build a model
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=16, kernel_size=3, padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D())

    for num_node in num_nodes:
        model.add(Conv1D(filters=num_node, kernel_size=3, padding='valid', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=num_node // 2, kernel_size=1, padding='valid', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=num_node, kernel_size=3, padding='valid', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=num_node // 2, kernel_size=1, padding='valid', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=num_node, kernel_size=3, padding='valid', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D())

    model.add(Conv1D(filters=256, kernel_size=3, padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=256, kernel_size=3, padding='valid', activation='relu'))
    model.add(BatchNormalization())

    model.add(GlobalMaxPool1D())
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(4, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    opt = Adam(learning_rate=1e-3)

    # Compile the model
    model.compile(
        loss=[binary_focal_loss(alpha=.25, gamma=2)],
        # loss='binary_crossentropy',
        optimizer=opt,
        metrics=['accuracy',
                 tf.keras.metrics.AUC(),
                 tf.keras.metrics.TruePositives(name='TP'),
                 tf.keras.metrics.FalsePositives(name='FP'),
                 tf.keras.metrics.FalseNegatives(name='FN'),
                 tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall(),
                 tf.keras.metrics.AUC(curve='PR', name='AUPRC')]
    )
    return model


TRAINMODE = False
BATCH_SIZE = 512

num_nodes = [32, 64, 128, 128, 128]
testname = '-'.join([str(num_node) for num_node in num_nodes])
print(testname)

odir = "output"
if not os.path.exists(odir):
    os.mkdir(odir)


if TRAINMODE:
    # training set 로딩
    print('loading train...')
    x_train = np.load('./data/x_train.npz')['arr_0']    # arterial waveform
    y_train = np.load('./data/y_train.npz')['arr_0']    # hypotension

    print('train {} ({} events {:.1f}%)'.format(len(y_train), sum(y_train), 100*np.mean(y_train)))

    # data balancing (random selection)
    idx_positive = np.where(y_train == True)[0]
    num_positive = len(idx_positive)

    idx_negative = np.where(y_train == False)[0]
    # idx_negative = np.sort(np.random.choice(idx_negative, num_positive, replace=False))

    idx_selection = np.sort(np.concatenate((idx_positive, idx_negative)))
    np.random.shuffle(idx_selection)
    np.random.shuffle(idx_selection)

    x_train = x_train[idx_selection]
    y_train = y_train[idx_selection]

    # preprocess
    x_train -= 65
    x_train /= 65
    x_train = pd.DataFrame(x_train)
    x_train = x_train.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1).values
    x_train = x_train[..., None]

    # 출력 폴더를 생성
    weight_path = odir + "/weights_ep{epoch:02d}.hdf5"

    model = build_model(num_nodes)

    hist = model.fit(x_train, y_train, validation_split=0.1, epochs=100, batch_size=BATCH_SIZE, class_weight={0:1, 1:10},
                     callbacks=[ModelCheckpoint(filepath=weight_path, verbose=1, save_best_only=False),
                                EarlyStopping(patience=30, verbose=1, mode='auto'),
                                ReduceLROnPlateau(factor=0.5, patience=6, verbose=1, mode='auto')])

    # 모델을 저장
    config = model.to_json()
    open(odir + "/model.json", "wt").write(config)

    x_train = None
    y_train = None

else:
    jsonfile = open(odir + "/model.json", "r")
    config = jsonfile.read()
    model = model_from_json(config)

    weight_path = odir + "/weights_ep35.hdf5"
    model.load_weights(weight_path)
    model.compile()


# test set 로딩
print('loading test...', flush=True, end='')
x_test = np.load('./data/x_test.npz')['arr_0']
x_test -= 65
x_test /= 65
x_test = pd.DataFrame(x_test).fillna(method='ffill', axis=1).fillna(method='bfill', axis=1).values
x_test = x_test[..., None]
print('done', flush=True)
print(len(x_test))

# 전체 test 샘플을 한번에 예측
y_pred = model.predict(x_test).flatten()

# 결과를 저장
np.savetxt('pred_y.txt', y_pred)

