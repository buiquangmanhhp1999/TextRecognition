from unicodedata import normalize
import re
from AttentionKeras.config import *
from tensorflow.keras import backend as K


def load_data(path):
    with open(path, "r") as file:
        return file.readlines()


def text_to_labels(text):
    text = normalize("NFC", text)
    text = re.sub("\s+", " ", text)
    return list(map(lambda x: character_list.index(x), text))


def labels_to_text(label):
    return ''.join(list(map(lambda x: character_list[x] if x < len(character_list) else "", label)))


def ctc_loss(args):
    y_true, y_pred, input_length, label_length = args
    # two first steps are often garbage
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
