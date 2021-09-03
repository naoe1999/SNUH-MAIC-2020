from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Concatenate, Multiply, add
from tensorflow.keras.layers import SeparableConv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Conv2D, MaxPooling2D
from tensorflow.keras.layers import ReLU, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed


def build_model():
    in_data = Input(shape=(2000, 4), name='input_data')
    in_info = Input(shape=(4, ), name='input_info')

    x = Conv1D(filters=32, kernel_size=3, padding='valid', activation='relu')(in_data)
    x = BatchNormalization()(x)
    x = Conv1D(filters=32, kernel_size=3, padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x)

    num_nodes = [32, 64, 64, 128, 128]

    for n in num_nodes:
        # shortcut = x
        x = Conv1D(filters=n, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters=n//2, kernel_size=1, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters=n, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters=n//2, kernel_size=1, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters=n, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)

        # shortcut = Conv1D(filters=n, kernel_size=3, padding='same', activation='relu')(shortcut)
        # shortcut = BatchNormalization()(shortcut)
        # x = add([x, shortcut])
        x = MaxPooling1D()(x)

    x = Conv1D(filters=256, kernel_size=3, padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=128, kernel_size=1, padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=256, kernel_size=3, padding='valid', activation='relu')(x)
    feat = BatchNormalization()(x)

    # branch network
    # x_p = MaxPooling1D()(feat)
    # x_p = Conv1D(filters=256, kernel_size=3, padding='valid', activation='relu')(x_p)
    # x_p = BatchNormalization()(x_p)
    # x_p = Conv1D(filters=128, kernel_size=1, padding='valid', activation='relu')(x_p)
    # x_p = BatchNormalization()(x_p)
    # x_p = Conv1D(filters=256, kernel_size=3, padding='valid', activation='relu')(x_p)
    # x_p = BatchNormalization()(x_p)
    #
    # # x_p = GlobalAveragePooling1D()(x_p)
    # x_p = Flatten()(x_p)
    # x_p = Dense(64, activation='relu')(x_p)
    # x_p = Dropout(0.5)(x_p)
    #
    # x_p = Concatenate()([x_p, in_info])
    # x_p = Dense(32, activation='relu')(x_p)
    # x_p = Dropout(0.5)(x_p)
    # out_p = Dense(1, activation='sigmoid', name='po')(x_p)

    # main branch
    # x_h = MaxPooling1D()(feat)
    # x_h = Conv1D(filters=256, kernel_size=3, padding='valid', activation='relu')(x_h)
    # x_h = BatchNormalization()(x_h)
    # x_h = Conv1D(filters=128, kernel_size=1, padding='valid', activation='relu')(x_h)
    # x_h = BatchNormalization()(x_h)
    # x_h = Conv1D(filters=256, kernel_size=3, padding='valid', activation='relu')(x_h)
    # x_h = BatchNormalization()(x_h)

    x_h = GlobalMaxPooling1D()(feat)
    # x_h = Flatten()(x_h)
    x_h = Concatenate()([x_h, in_info])

    x_h = Dense(64, activation='relu')(x_h)
    x_h = BatchNormalization()(x_h)
    x_h = Dropout(0.5)(x_h)

    x_h = Dense(16, activation='relu')(x_h)
    x_h = BatchNormalization()(x_h)
    x_h = Dropout(0.5)(x_h)
    out_h = Dense(1, activation='sigmoid', name='out')(x_h)

    # out_final = Multiply(name='fo')([out_p, out_h])

    model = Model(inputs=[in_data, in_info], outputs=out_h)

    return model


def build_model_tenet():
    in_data = Input(shape=(2000, 4), name='input_data')
    in_info = Input(shape=(4, ), name='input_info')

    num_nodes = [32, 64, 64, 128, 128]

    # forward data
    x = Conv1D(filters=32, kernel_size=5, padding='valid', activation=LeakyReLU(0.1))(in_data)
    x = BatchNormalization()(x)
    x = Conv1D(filters=32, kernel_size=5, padding='valid', activation=LeakyReLU(0.1))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x)

    for n in num_nodes:
        x = Conv1D(filters=n, kernel_size=5, padding='same', activation=LeakyReLU(0.1))(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters=n//2, kernel_size=1, padding='same', activation=LeakyReLU(0.1))(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters=n, kernel_size=5, padding='same', activation=LeakyReLU(0.1))(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters=n//2, kernel_size=1, padding='same', activation=LeakyReLU(0.1))(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters=n, kernel_size=5, padding='same', activation=LeakyReLU(0.1))(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D()(x)

    # reverse data
    xr = in_data[..., ::-1, :]
    xr = Conv1D(filters=32, kernel_size=5, padding='valid', activation=LeakyReLU(0.1))(xr)
    xr = BatchNormalization()(xr)
    xr = Conv1D(filters=32, kernel_size=5, padding='valid', activation=LeakyReLU(0.1))(xr)
    xr = BatchNormalization()(xr)
    xr = MaxPooling1D()(xr)

    for n in num_nodes:
        xr = Conv1D(filters=n, kernel_size=5, padding='same', activation=LeakyReLU(0.1))(xr)
        xr = BatchNormalization()(xr)
        xr = Conv1D(filters=n // 2, kernel_size=1, padding='same', activation=LeakyReLU(0.1))(xr)
        xr = BatchNormalization()(xr)
        xr = Conv1D(filters=n, kernel_size=5, padding='same', activation=LeakyReLU(0.1))(xr)
        xr = BatchNormalization()(xr)
        xr = Conv1D(filters=n // 2, kernel_size=1, padding='same', activation=LeakyReLU(0.1))(xr)
        xr = BatchNormalization()(xr)
        xr = Conv1D(filters=n, kernel_size=5, padding='same', activation=LeakyReLU(0.1))(xr)
        xr = BatchNormalization()(xr)
        xr = MaxPooling1D()(xr)

    # merge
    x_merge = Concatenate()([x, xr])

    x = Conv1D(filters=256, kernel_size=5, padding='valid', activation=LeakyReLU(0.1))(x_merge)
    x = BatchNormalization()(x)
    x = Conv1D(filters=128, kernel_size=1, padding='valid', activation=LeakyReLU(0.1))(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=256, kernel_size=5, padding='valid', activation=LeakyReLU(0.1))(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=128, kernel_size=1, padding='valid', activation=LeakyReLU(0.1))(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=256, kernel_size=5, padding='valid', activation=LeakyReLU(0.1))(x)
    x = BatchNormalization()(x)
    x = GlobalMaxPooling1D()(x)

    x = Concatenate()([x, in_info])

    x = Dense(64, activation=LeakyReLU(0.1))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(16, activation=LeakyReLU(0.1))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    out_h = Dense(1, activation='sigmoid', name='out')(x)

    model = Model(inputs=[in_data, in_info], outputs=out_h)

    return model


def build_model_axnet():
    in_data = Input(shape=(2000, 4), name='input_data')
    in_info = Input(shape=(4,), name='input_info')

    num_nodes = [32, 64, 64, 128, 128, 256]

    x = SeparableConv1D(filters=32, kernel_size=5, padding='valid', depth_multiplier=8, activation=None)(in_data)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling1D()(x)

    for n in num_nodes:
        x = SeparableConv1D(filters=n, kernel_size=3, padding='same', activation=None)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = LeakyReLU(0.2)(x)
        x = MaxPooling1D()(x)

    x = SeparableConv1D(filters=256, kernel_size=3, padding='valid', activation=None)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    feat = LeakyReLU(0.2)(x)

    x = GlobalMaxPooling1D()(feat)
    x = Concatenate()([x, in_info])

    x = Dense(64, activation=None)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = LeakyReLU(0.2)(x)

    x = Dense(16, activation=None)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = LeakyReLU(0.2)(x)

    out = Dense(1, activation='sigmoid', name='out')(x)

    model = Model(inputs=[in_data, in_info], outputs=out)

    return model


def feature_encoder(x_in):
    x = SeparableConv1D(filters=32, kernel_size=3, padding='valid', activation='relu')(x_in)
    x = BatchNormalization()(x)
    x = SeparableConv1D(filters=32, kernel_size=3, padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x)

    num_nodes = [64, 64, 128, 128]

    for n in num_nodes:
        x = Conv1D(filters=n, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters=n // 2, kernel_size=1, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters=n, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters=n // 2, kernel_size=1, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters=n, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D()(x)

    feature = GlobalMaxPooling1D()(x)
    return feature


def build_model_iannet():
    in_data = Input(shape=(2000, 4), name='input_data')
    in_info = Input(shape=(4,), name='input_info')

    feat = feature_encoder(in_data)             # (N, 128)

    x = Concatenate()([feat, in_info])          # (N, 132)
    x = Dense(64, activation='tanh')(x)         # (N, 64)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = RepeatVector(6)(x)                      # (N, 6, 64)
    x = LSTM(16, return_sequences=True)(x)      # (N, 6, 16)

    final_layer = Dense(1, name='hypo')
    preds = TimeDistributed(final_layer)(x)     # (N, 6, 1)

    model = Model(inputs=[in_data, in_info], outputs=preds)
    return model

