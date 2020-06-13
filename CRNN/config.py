(character_list) = ' !"#$%\'()*+,-./0123456789:;=?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_abcdefghijklmnopqrstuvwxyz{|}~§©°é'
PATH_TRAIN = "crop_linetext/b-mod_lines/train.easy"
PATH_VALID = "crop_linetext/b-mod_lines/valid.easy"
PATH_TEST = "crop_linetext/b-mod_lines/test.easy"
NO_CLASSES = len(character_list) + 1
BATCH_SIZE = 32
IMAGE_HEIGHT = 32
NO_CHANNEL = 1
FILTER_SIZE_1 = 6
FILTER_SIZE_2 = 5
FILTER_SIZE_3 = 4
FILTER_SIZE_4 = 3
STRIDE = 4
