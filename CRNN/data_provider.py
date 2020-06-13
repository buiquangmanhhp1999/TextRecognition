import cv2
import numpy as np
from sklearn.utils import shuffle
import os
from AttentionKeras.data_utils import *


class DataGenerator(object):
    def __init__(self, image_list, trainable=True):
        self.trainable = trainable
        self.image_list = shuffle(image_list)
        self.batch_size = BATCH_SIZE
        self.current_train_index = 0
        self.current_val_index = 0

    def gen(self):
        batch_img = []
        batch_label = []

        def format_input(batch_x, batch_y):
            len_batch = len(batch_x)
            max_image_width = max([img.shape[1] for img in batch_x])
            max_label_length = max(len(label) for label in batch_y)
            input_image = np.ones(len_batch, IMAGE_HEIGHT, max_image_width, NO_CHANNEL)
            input_true_label = np.ones((len_batch, max_label_length)) * (NO_CLASSES - 1)
            input_time_step = np.zeros((len_batch, 1))
            input_label_length = np.zeros((len_batch, 1))

            for id in range(len_batch):
                real_width = batch_x[id].shape[1]
                real_label_len = len(batch_y[id])
                tmp = text_to_labels(batch_y[id])
                input_image[id, :, :real_width, :] = batch_x[id]
                input_true_label[id, :real_label_len] = tmp
                input_time_step[id] = self.compute_time_step(real_width) - 2
                input_label_length[id] = real_label_len
            inputs = {'input_image': input_image, 'input_true_label': input_true_label, 'input_time_step': input_time_step,
                      'input_label_length': input_label_length}

            outputs = {'ctc': np.zeros(len_batch)}
            return inputs, outputs

        for i in range(len(self.image_list)):
            img_path, true_label = self.image_list[i].split(" ", 1)
            img_path = os.path.join("../crop_linetext/b-mod_lines/lines", img_path)
            image = self.load_image(img_path)

            true_label = true_label.strip()
            true_label = normalize("NFC", true_label)

            batch_img.append(image)
            batch_label.append(true_label)

            if len(batch_img) == self.batch_size:
                yield format_input(batch_img, batch_label)
                batch_img = []
                batch_label = []

        if len(batch_img) > 0:
            yield format_input(batch_img, batch_label)

    @staticmethod
    def compute_time_step(image_width):
        tmp = image_width
        for i in range(2):
            tmp = (tmp - 1) // 2 + 1
        tmp = (tmp + STRIDE // 4 - 1) // (STRIDE // 4)
        return tmp

    @staticmethod
    def load_image(image_path):
        image = cv2.imread(image_path, 0)
        ratio = image.shape[0] / IMAGE_HEIGHT
        image = cv2.resize(image, (int(image.shape[1] / ratio), IMAGE_HEIGHT))
        image = image / 255.
        image = np.expand_dims(image, axis=-1)
        return image
