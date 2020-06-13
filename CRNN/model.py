import tensorflow.keras as keras
from tensorflow.keras.layers import *
from AttentionKeras.config import *
from keras_self_attention import SeqSelfAttention
from AttentionKeras.library import *
from tensorflow.keras import backend as K
from AttentionKeras.data_provider import DataGenerator
from AttentionKeras.data_utils import load_data


class Model(keras.Model):
    def __init__(self, image_height, no_channel,  show_summaries=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_height = image_height
        self.no_channel = no_channel
        self.batch_size = BATCH_SIZE

        # define input
        self.input_true_label = Input(shape=(None,), name='input_true_label')
        self.input_time_step = Input(shape=(1,), name='input_time_step')
        self.input_label_length = Input(shape=(1,), name='input_label_length')
        self.input_image = Input(shape=(self.image_height, None, self.no_channel), name='input_image')

        # build model
        self.output_model, self.y_func = self.build_model()
        if show_summaries:
            self.output_model.summary()

    def build_model(self):
        x = BatchNormalization()(self.input_image)
        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        x_1 = Conv2D(filters=16, kernel_size=(12, FILTER_SIZE_1), strides=(1, 1), padding='same')(x)
        x_1 = MaxPool2D(pool_size=(6, 1))(x_1)

        x_2 = Conv2D(filters=16, kernel_size=(12, FILTER_SIZE_2), strides=(1, 1), padding='same')(x)
        x_2 = MaxPool2D(pool_size=(6, 1))(x_2)

        x_3 = Conv2D(filters=16, kernel_size=(12, FILTER_SIZE_3), strides=(1, 1), padding='same')(x)
        x_3 = MaxPool2D(pool_size=(6, 1))(x_3)

        x_4 = Conv2D(filters=16, kernel_size=(12, FILTER_SIZE_4), strides=(1, 1), padding='same')(x)
        x_4 = MaxPool2D(pool_size=(6, 1))(x_4)

        x = concatenate([x_1, x_2, x_3, x_4], axis=-1)
        x = Reshape([1, -1, 16 * 2])(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = K.squeeze(x, axis=1)

        # Part Attention
        x = SeqSelfAttention(attention_activation='sigmoid')(x)

        gru = GRU(units=64, return_sequences=True)(x)
        gru = BatchNormalization()(gru)
        gru = Activation('relu')(gru)

        dense = TimeDistributed(Dense(units=NO_CLASSES))(gru)
        dense = Activation('softmax')(dense)

        loss = Lambda(ctc_loss, output_shape=(1,), name='ctc')(self.input_true_label, dense, self.input_time_step, self.input_label_length)
        model = Model([self.input_image, self.input_true_label, self.input_time_step, self.input_label_length], loss)
        y_func = K.function([self.input_image], [dense])

        return model, y_func

    def train(self):
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min')
        optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-9, decay=1e-6, amsgrad=True, clipnorm=5., clipvalue=0.5)
        MODEL_PATH = 'pre-trained2/epoch_{epoch}_{loss:.5f}_{val_loss:.5f}.h5'
        checkpoint = keras.callbacks.ModelCheckpoint(filepath=MODEL_PATH, save_best_only=True, verbose=1)
        train_img_list = load_data(path=PATH_TRAIN)
        val_img_list = load_data(path=PATH_VALID)
        data_gen_train = DataGenerator(train_img_list, trainable=True)
        data_gen_val = DataGenerator(val_img_list, trainable=False)
        step_val = len(val_img_list) // self.batch_size
        step_train = len(train_img_list) // self.batch_size // 2

        self.output_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)
        self.output_model.fit_generator(generator=data_gen_train.gen(), steps_per_epoch=step_train, epochs=20, verbose=1,
                                        callbacks=[checkpoint, early_stop], validation_data=data_gen_val.gen(), validation_steps=step_val)
