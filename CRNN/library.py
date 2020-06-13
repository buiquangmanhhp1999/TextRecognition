from unicodedata import normalize


def ctc_loss(args):
    y_true, y_pred, input_length, label_length = args
    # two first steps are often garbage
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)