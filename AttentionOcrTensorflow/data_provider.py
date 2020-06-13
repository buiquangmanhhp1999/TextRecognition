from collections import namedtuple
import functools
import tensorflow as tf
import tf_slim
from inception_preprocessing import apply_with_random_selector, distort_color

tf.compat.v1.disable_eager_execution()

"""
images: [batch_size, H, W, 3]
labels: ground truth label ids [batch_size, seq_length]
label_one_hot: labels in one-hot encoding [batch_size x seq_length x num_char_classes]
"""
InputEndpoints = namedtuple('InputEndPoints', ['images', 'images_orig', 'labels', 'labels_one_hot'])

"""
num_batching_threads: a number of parallel threads to fetch data
queue_capacity" a max number of elements in the batch shuffling queue
min_after_dequeue: a min number elements in the queue after dequeue. used to ensure a level of mixing of elements
"""

ShuffleBatchConfig = namedtuple('ShuffleBatchConfig', ['num_batching_threads', 'queue_capacity', 'min_after_dequeue'])
DEFAULT_SHUFFLE_CONFIG = ShuffleBatchConfig(num_batching_threads=8, queue_capacity=3000, min_after_dequeue=1000)


def augment_image(input_image):
    """
    Augmentation the image with a random modification

    :param input_image: has shape of [height, width, 3]
    :return:
        distorted tensor image of the same shape
    """

    with tf.compat.v1.variable_scope("AugmentImage"):
        height = input_image.get_shape().dims[0]
        width = input_image.get_shape().dims[1]

        # randomly crop, resized to the same size, the crop area is cover at least 0.8 area of the input image
        # Note bounding_boxes has shape of [batch, N, 4] describing the N bounding boxes associated with the image
        # aspect_ratio_range = width / height
        begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(image_size=tf.shape(input_image),
                                                                            bounding_boxes=tf.zeros([0, 0, 4]),
                                                                            min_object_covered=0.8,
                                                                            aspect_ratio_range=[0.8, 1.2],
                                                                            area_range=[0.8, 1.0],
                                                                            use_image_if_no_bounding_boxes=True)

        # crop image
        distorted_image = tf.slice(input_image, begin, size)

        # Randomly chooses one of the 4 interpolation methods
        # tf.image.resize to resize image size which defines in method
        distorted_image = apply_with_random_selector(distorted_image, lambda x, method: tf.compat.v1.image.resize(x, [height, width], method), num_cases=4)
        distorted_image.set_shape([height, width, 3])

        # color distortion
        distorted_image = apply_with_random_selector(distorted_image, functools.partial(distort_color, fast_mode=False), num_cases=4)

        distorted_image = tf.clip_by_value(distorted_image, -1.5, 1.5)

    return distorted_image


def central_crop(image, crop_size):
    """Returns a central crop for the specified size of an image.
  Args:
    image: A tensor with shape [height, width, channels]
    crop_size: A tuple (crop_width, crop_height)
  Returns:
    A tensor of shape [crop_height, crop_width, channels].
  """
    with tf.compat.v1.variable_scope('CentralCrop'):
        target_width, target_height = crop_size
        image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]
        assert_op1 = tf.Assert(
            tf.greater_equal(image_height, target_height),
            ['image_height < target_height', image_height, target_height])
        assert_op2 = tf.Assert(
            tf.greater_equal(image_width, target_width),
            ['image_width < target_width', image_width, target_width])

        # when eager execution is enabled, any callable object in the control_inputs list will be called
        with tf.control_dependencies([assert_op1, assert_op2]):
            offset_width = tf.cast((image_width - target_width) / 2, tf.int32)
            offset_height = tf.cast((image_height - target_height) / 2, tf.int32)
            return tf.image.crop_to_bounding_box(image, offset_height, offset_width,
                                                 target_height, target_width)


def preprocess_image(image, augment=False, central_crop_size=None,
                     num_towers=4):
    """Normalizes image to have values in a narrow range around zero.
  Args:
    image: a [H x W x 3] uint8 tensor.
    augment: optional, if True do random image distortion.
    central_crop_size: A tuple (crop_width, crop_height).
    num_towers: optional, number of shots of the same image in the input image.
  Returns:
    A float32 tensor of shape [H x W x 3] with RGB values in the required
    range.
  """
    with tf.compat.v1.variable_scope('PreprocessImage'):
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        if augment or central_crop_size:
            if num_towers == 1:
                images = [image]
            else:
                images = tf.split(value=image, num_or_size_splits=num_towers, axis=1)
            if central_crop_size:
                view_crop_size = (int(central_crop_size[0] / num_towers), central_crop_size[1])
                images = [central_crop(img, view_crop_size) for img in images]
            if augment:
                images = [augment_image(img) for img in images]
            image = tf.concat(images, 1)

        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.5)

    return image


def get_data(dataset, batch_size, augment=False, central_crop_size=None, shuffle_config=None, shuffle=True):
    """Wraps calls to DatasetDataProviders and shuffle_batch.
  For more details about supported Dataset objects refer to datasets/fsns.py.
  Args:
    dataset: a slim.data.dataset.Dataset object.
    batch_size: number of samples per batch.
    augment: optional, if True does random image distortion.
    central_crop_size: A CharLogit tuple (crop_width, crop_height).
    shuffle_config: A namedtuple ShuffleBatchConfig.
    shuffle: if True use data shuffling.
  Returns:
  """
    if not shuffle_config:
        shuffle_config = DEFAULT_SHUFFLE_CONFIG

    provider = tf_slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=shuffle,
        common_queue_capacity=2 * batch_size,
        common_queue_min=batch_size)
    image_orig, label = provider.get(['image', 'label'])

    image = preprocess_image(image_orig, augment, central_crop_size, num_towers=dataset.num_of_views)
    label_one_hot = tf_slim.one_hot_encoding(label, dataset.num_char_classes)
    # print(image.get_shape())
    # print(image_orig.get_shape())
    # print(label[0].get_shape())
    # print(label_one_hot.get_shape())

    """
    dataset = tf.data.Dataset.from_tensor_slices((image, image_orig, label, label_one_hot))
    dataset = dataset.shuffle(buffer_size=shuffle_config.min_after_dequeue, reshuffle_each_iteration=True).batch(batch_size=batch_size)

    images = tf.constant(list(dataset.map(lambda x_img, x_img_orig, y_label, y_label_one_hot: x_img)))
    images_orig = tf.constant(list(dataset.map(lambda x_img, x_img_orig, y_label, y_label_one_hot: x_img_orig)))
    labels = tf.constant(list(dataset.map(lambda x_img, x_img_orig, y_label, y_label_one_hot: y_label)))
    labels_one_hot = tf.constant(list(dataset.map(lambda x_img, x_img_orig, y_label, y_label_one_hot: y_label_one_hot)))
    """

    images, images_orig, labels, labels_one_hot = (tf.compat.v1.train.shuffle_batch(
        [image, image_orig, label, label_one_hot],
        batch_size=batch_size,
        num_threads=shuffle_config.num_batching_threads,
        capacity=shuffle_config.queue_capacity,
        min_after_dequeue=shuffle_config.min_after_dequeue))

    return InputEndpoints(
        images=images,
        images_orig=images_orig,
        labels=labels,
        labels_one_hot=labels_one_hot)
