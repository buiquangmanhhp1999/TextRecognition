from collections import namedtuple
import tensorflow as tf
import tf_slim
from tf_slim.nets import inception
import logging
from sequence_layers import get_layer_class
from libraries import logits_to_log_prob, get_softmax_loss_fn, char_accuracy, sequence_accuracy, variables_to_restore
import tensorflow_addons as tfa
import sys
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops

OutputEndpoints = namedtuple("OutputEndpoints",
                             ["chars_logit", "chars_log_prob", "predicted_chars", 'predicted_scores', 'predicted_text'])
ModelParams = namedtuple('ModelParams', ['num_char_classes', 'seq_length', 'num_views', 'null_code'])
ConvTowerParams = namedtuple('ConvTowerParams', ['final_endpoint'])
SequenceLogitsParams = namedtuple('SequenceLogitsParams',
                                  ['use_attention', 'use_autoregression', 'num_lstm_units', 'weight_decay'
                                      , 'lstm_state_clip_value'])
SequenceLossParams = namedtuple('SequenceLossParams', ['label_smoothing', 'ignore_nulls', 'average_across_timesteps'])
EncodeCoordinatesParams = namedtuple('EncodeCoordinatesParams', ['enabled'])


def sequence_loss_by_example(logits,
                             targets,
                             weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None,
                             name=None):
    """Weighted cross-entropy loss for a sequence of logits (per example).
  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: Function (labels, logits) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
      **Note that to avoid confusion, it is required for the function to accept
      named arguments.**
    name: Optional name for this operation, default: "sequence_loss_by_example".
  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.
  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
    if len(targets) != len(logits) or len(weights) != len(logits):
        raise ValueError("Lengths of logits, weights, and targets must be the same "
                         "%d, %d, %d." % (len(logits), len(weights), len(targets)))
    with ops.name_scope(name, "sequence_loss_by_example",
                        logits + targets + weights):
        log_perp_list = []
        for logit, target, weight in zip(logits, targets, weights):
            if softmax_loss_function is None:
                # TODO(irving,ebrevdo): This reshape is needed because
                # sequence_loss_by_example is called with scalars sometimes, which
                # violates our general scalar strictness policy.
                target = array_ops.reshape(target, [-1])
                crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
                    labels=target, logits=logit)
            else:
                crossent = softmax_loss_function(labels=target, logits=logit)
            log_perp_list.append(crossent * weight)
        log_perps = math_ops.add_n(log_perp_list)
        if average_across_timesteps:
            total_size = math_ops.add_n(weights)
            total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
            log_perps /= total_size
    return log_perps


def sequence_loss(logits,
                  targets,
                  weights,
                  average_across_timesteps=True,
                  average_across_batch=True,
                  softmax_loss_function=None,
                  name=None):
    """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.
  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    softmax_loss_function: Function (labels, logits) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
      **Note that to avoid confusion, it is required for the function to accept
      named arguments.**
    name: Optional name for this operation, defaults to "sequence_loss".
  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).
  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
    with ops.name_scope(name, "sequence_loss", logits + targets + weights):
        cost = math_ops.reduce_sum(
            sequence_loss_by_example(
                logits,
                targets,
                weights,
                average_across_timesteps=average_across_timesteps,
                softmax_loss_function=softmax_loss_function))
        if average_across_batch:
            batch_size = array_ops.shape(targets[0])[0]
            return cost / math_ops.cast(batch_size, cost.dtype)
        else:
            return cost


class CharsetMapper(object):
    """
    A simple class to map tensor ids into strings.

    It works only when the character set is 1:1 mapping between individual
    characters and individual ids.

    Make sure you call tf.tables_initializer().run() as part of the init op.
    """

    def __init__(self, charset, default_character='?'):
        """
        Creates a lookup table.
        Args:
        charset: a dictionary with id-to-character mapping.
        """
        list_keys = []
        list_values = []

        for key, value in charset.items():
            list_keys.append(key)
            list_values.append(value)

        list_keys = tf.cast(tf.constant(list_keys), tf.int64)
        list_values = tf.constant(list_values)
        self.table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(list_keys, list_values),
                                               default_value=default_character)

    def get_text(self, ids):
        """
        Returns a string corresponding to a sequence of character ids
        :param ids: a tensor with shape [batch_size, max_sequence_length]
        """

        return tf.compat.v1.reduce_join(self.table.lookup(tf.cast(ids, tf.int64)), axis=1)


class Model(object):
    """Attention OCR Model"""

    def __init__(self, num_char_classes, seq_length, num_views, null_code, mparams=None, charset=None):
        """
        Initialized model parameters
        Args:
            num_char_classes: size of character set
            seq_length: number of characters in a sequence
            num_views: number of views(in FSNS number of view is 4)
            null_code:
            mparams: a dictionary with hyper parameters
            charset: a dictionary with a mapping between character ids and utf8 strings
        """

        super(Model, self).__init__()
        self._params = ModelParams(num_char_classes=num_char_classes, seq_length=seq_length, num_views=num_views,
                                   null_code=null_code)
        self._mparams = self.default_mparams()
        if mparams:
            self._mparams.update(mparams)
        self._charset = charset

    @staticmethod
    def default_mparams():
        """set parameters value for method"""
        return {
            'conv_tower_fn': ConvTowerParams(final_endpoint='Mixed_5d'),
            'sequence_logit_fn': SequenceLogitsParams(use_attention=True, use_autoregression=True, num_lstm_units=256,
                                                      weight_decay=0.00004, lstm_state_clip_value=10.0),
            'sequence_loss_fn': SequenceLossParams(label_smoothing=0.1, ignore_nulls=True,
                                                   average_across_timesteps=False),
            'encode_coordinates_fn': EncodeCoordinatesParams(enabled=False)
        }

    def set_mparam(self, function, **kwargs):
        self._mparams[function] = self._mparams[function]._replace(**kwargs)

    def conv_tower_fn(self, images, is_training=True, reuse=None):
        """
        compute convolutional features using the inceptionV3 model
        :param images: have format [batch_size, height, width, channel]
        :param is_training: training or not
        :param reuse:
        :return: a tensor of shape [batch_size, OH, OW, N], where OWxOH is resolution of output feature map and N is number
                of output features
        """
        mparams = self._mparams['conv_tower_fn']
        logging.debug('Using final_endpoint=%s', mparams.final_endpoint)

        # tf.get_variable_scope().reuse_variables() will always procedure None as result. if only use when attribute reuse
        # of the current scope is True

        with tf.compat.v1.variable_scope('conv_tower_fn/INCEPTION'):
            if reuse:
                tf.compat.v1.get_variable_scope().reuse_variables()
            # stores the default argument for the given set of list_ops.
            with tf_slim.arg_scope(inception.inception_v3_arg_scope()):
                with tf_slim.arg_scope([tf_slim.batch_norm, tf_slim.dropout], is_training=is_training):
                    net, _ = inception.inception_v3_base(images, final_endpoint=mparams.final_endpoint)

            return net

    def encode_coordinates_fn(self, net):
        """
        Adds one-hot encoding of coordinates to different views in the networks.
        For each "pixel" of a feature map it adds a one hot encoded x and y
        coordinates.
        :param net: a tensor of shape=[batch_size, height, width, num_features]
        :return: a tensor with the same height and width, but altered feature_size.
        """

        mparams = self._mparams['encode_coordinates_fn']
        if mparams.enabled:
            batch_size, h, w, _ = net.shape.as_list()

            # create two matrix has shape (w, h) or (w, h)
            x, y = tf.meshgrid(tf.range(w), tf.range(h))
            w_loc = tf_slim.one_hot_encoding(x, num_classes=w)  # shape of (w, h, w)
            h_loc = tf_slim.one_hot_encoding(y, num_classes=h)  # shape of (w, h, h)
            loc = tf.concat([h_loc, w_loc], axis=2)  # shape of (w, h, w + h)
            loc = tf.tile(tf.expand_dims(loc, 0), [batch_size, 1, 1, 1])  # shape of (batch_size, w, h, w + h)

            return tf.concat([net, loc], 3)  # shape of (batch_size, w, h, w + h + num_features)
        else:
            return net

    @staticmethod
    def pool_views_fn(nets):
        """
        Combines output of multiple convolutional towers into a single tensor
        :param nets: a tensor of shape [batch_size, height, width, num_features]
        :return: a tensor of shape [batch_size, seq_length, features_size]
        """

        with tf.compat.v1.variable_scope('pool_views_fn/STCK'):
            net = tf.concat(nets, 1)  # shape of [height, batch_size * width, num_features]
            batch_size = net.get_shape().dims[0]
            feature_size = net.get_shape().dims[3]
            return tf.reshape(net, [batch_size, -1, feature_size])

    def sequence_logit_fn(self, net, labels_one_hot):
        mparams = self._mparams['sequence_logit_fn']

        with tf.compat.v1.variable_scope('sequence_logit_fn/SQLR'):
            layer_class = get_layer_class(mparams.use_attention, mparams.use_autoregression)
            layer = layer_class(net, labels_one_hot, self._params, mparams)
            return layer.create_logits()

    def char_prediction(self, chars_logit):
        """
        return confidence scores (softmax values) for predicted characters

        :param chars_logit: chars logits, a tensor with shape [batch_size x seq_length x num_char_classes]
        :return:
            A tuple (ids, log_prob, scores), where:
            ids - predicted characters, a int32 tensor with shape
            [batch_size x seq_length];
            log_prob - a log probability of all characters, a float tensor with
            shape [batch_size, seq_length, num_char_classes];
            scores - corresponding confidence scores for characters, a float
                    tensor with shape [batch_size x seq_length].
        """

        log_prob = logits_to_log_prob(chars_logit)
        ids = tf.cast(tf.argmax(log_prob, axis=2), name='predicted_chars', dtype=tf.int32)

        mask = tf.cast(tf_slim.one_hot_encoding(ids, self._params.num_char_classes), tf.bool)
        all_scores = tf.nn.softmax(chars_logit)
        selected_scores = tf.boolean_mask(all_scores, mask, name='char_scores')
        scores = tf.reshape(selected_scores, shape=(-1, self._params.seq_length))

        return ids, log_prob, scores

    def _create_lstm_inputs(self, net):
        """
        Split an input tensor into a list of tensors(features) to make the model "location aware" according to horizontal
        and vertical axis

        :param self:
        :param net: a feature map which shape [batch_size, num_features, feature_size]
                    batch_size: number of images
                    num_features: equal to chars in sequence

        :raises: if num_features is less the sequence length
        :return:
            A list with sequence length tensors of shape [batch_size, feature_size]
        """
        num_features = net.get_shape().dims[1]
        if num_features < self._params.seq_length:
            raise AssertionError('Incorrect dimension of input tensor'
                                 '%d should be bigger than %d (shape%s)' % (
                                     num_features, self._params.seq_length, net.get_shape()))
        elif num_features > self._params.seq_length:
            logging.warning('Ignoring some features: use %d of %d(shape=%s)', self._params.seq_length, num_features,
                            net.get_shape())
            net = tf.slice(net, [0, 0, 0], [-1, self._params.seq_length, -1])

        return tf.unstack(net, axis=1)  # split input tensor into seq_length theo axis 1

    def create_base(self, images, labels_one_hot, scope='AttentionOcr', reuse=None):
        """
        Creates a base part of the Model (no gradients, losses or summaries).
            Args:
              images: A tensor of shape [batch_size, height, width, channels].
              labels_one_hot: Optional (can be None) one-hot encoding for ground truth
                labels. If provided the function will create a model for training.
              scope: Optional variable_scope.
              reuse: whether or not the network and its variables should be reused. To
                be able to reuse 'scope' must be given.
            Returns:
              A named tuple OutputEndpoints.

        """
        logging.debug('images: %s', images)

        # if labels_one_hot is None ==> is_training = False else True
        is_training = labels_one_hot is not None

        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            views = tf.split(value=images, num_or_size_splits=self._params.num_views, axis=2)
            logging.debug('Views=%d single view: %s', len(views), views[0])

            # part1: take input images to CNN to extract features
            nets = [self.conv_tower_fn(v, is_training, reuse=(i != 0)) for i, v in enumerate(views)]
            logging.debug('Conv Tower: %s', nets[0])

            # part2: make views to one hot format
            nets = [self.encode_coordinates_fn(net) for net in nets]
            logging.debug('Conv tower w/encoded coordinates: %s', nets[0])

            # part3: combines multiple convolutional tower into single towers
            net = self.pool_views_fn(nets)
            logging.debug('Pooled views: %s', net)

            # part4: use LSTM
            chars_logit = self.sequence_logit_fn(net, labels_one_hot)
            logging.debug('chars_logit: %s', chars_logit)

            # part5: predict characters
            predicted_chars, chars_log_prob, predicted_scores = (self.char_prediction(chars_logit))

            if self._charset:
                character_mapper = CharsetMapper(self._charset)
                predicted_text = character_mapper.get_text(predicted_chars)
            else:
                predicted_text = tf.constant([])

            return OutputEndpoints(chars_logit=chars_logit, chars_log_prob=chars_log_prob,
                                   predicted_chars=predicted_chars,
                                   predicted_scores=predicted_scores, predicted_text=predicted_text)

    def label_smoothing_regularization(self, chars_labels, weight=0.1):
        """
        Applies a label smoothing regularization. ==> to avoid over_confidence
            Uses the same method as in https://arxiv.org/abs/1512.00567.
            Args:
              chars_labels: ground truth ids of characters,
                shape=[batch_size, seq_length];
              weight: label-smoothing regularization weight.
            Returns:
              A sensor with the same shape as the input.
        """

        one_hot_labels = tf.one_hot(chars_labels, depth=self._params.num_char_classes, axis=-1)
        pos_weight = 1.0 - weight
        neg_weight = weight / self._params.num_char_classes

        return one_hot_labels * pos_weight + neg_weight

    def sequence_loss_fn(self, chars_logits, chars_labels):
        """
        Loss function for char sequence.
            Depending on values of hyper parameters it applies label smoothing and can
            also ignore all null chars after the first one.
            Args:
              chars_logits: logits for predicted characters,
                shape=[batch_size, seq_length, num_char_classes];
              chars_labels: ground truth ids of characters,
                shape=[batch_size, seq_length];

            Returns:
              A Tensor with shape [batch_size] - the log-perplexity for each sequence.
        """
        # Batch size: 32
        # Seq length: 37
        # Num char classes: 134
        """
        mparams = self._mparams['sequence_loss_fn']
        with tf.compat.v1.variable_scope('sequence_loss_fn/SLF'):
            if mparams.label_smoothing > 0:
                smoothed_one_hot_labels = self.label_smoothing_regularization(chars_labels, mparams.label_smoothing)
                labels_list = smoothed_one_hot_labels
            else:
                labels_list = chars_labels
                # labels_list = tf.unstack(chars_labels, axis=1)

            batch_size, seq_length, _ = chars_logits.shape.as_list()

            if mparams.ignore_nulls:
                weights = tf.ones((batch_size, seq_length), dtype=tf.float32)
            else:
                # suppose that reject character is the last in the charset
                reject_char = tf.constant(self._params.num_char_classes - 1, shape=(batch_size, seq_length),
                                          dtype=tf.int64)
                know_char = tf.not_equal(chars_labels, reject_char)
                weights = tf.compat.v1.to_float(know_char)

            # label_rank = len(labels_list.get_shape())
            # logits shape [32, 37, 134], [32, 37], [32, 37]
            loss = tfa.seq2seq.sequence_loss(chars_logits, labels_list, weights, average_across_timesteps=True)
            tf.compat.v1.losses.add_loss(loss)
            return loss
        """

        mparams = self._mparams['sequence_loss_fn']
        with tf.compat.v1.variable_scope('sequence_loss_fn/SLF'):
            if mparams.label_smoothing > 0:
                smoothed_one_hot_labels = self.label_smoothing_regularization(
                    chars_labels, mparams.label_smoothing)
                labels_list = tf.unstack(smoothed_one_hot_labels, axis=1)
            else:
                # NOTE: in case of sparse softmax we are not using one-hot
                # encoding.
                labels_list = tf.unstack(chars_labels, axis=1)

            batch_size, seq_length, _ = chars_logits.shape.as_list()
            if mparams.ignore_nulls:
                weights = tf.ones((batch_size, seq_length), dtype=tf.float32)
            else:
                # Suppose that reject character is the last in the charset.
                reject_char = tf.constant(
                    self._params.num_char_classes - 1,
                    shape=(batch_size, seq_length),
                    dtype=tf.int64)
                known_char = tf.not_equal(chars_labels, reject_char)
                weights = tf.compat.v1.to_float(known_char)

            logits_list = tf.unstack(chars_logits, axis=1)
            weights_list = tf.unstack(weights, axis=1)
            loss = sequence_loss(
                logits_list,
                labels_list,
                weights_list,
                softmax_loss_function=get_softmax_loss_fn(mparams.label_smoothing),
                average_across_timesteps=mparams.average_across_timesteps)
            tf.compat.v1.losses.add_loss(loss)
            return loss

    def create_loss(self, data, endpoints):
        """
            Creates all losses required to train the model.
            Args:
                data: InputEndpoints namedtuple.
                endpoints: Model namedtuple.
            Returns:
                Total loss.
        """

        self.sequence_loss_fn(endpoints.chars_logit, data.labels)
        total_loss = tf_slim.losses.get_total_loss()
        tf.summary.scalar('TotalLoss', total_loss)

        return total_loss

    def create_summaries(self, data, endpoints, charset, is_training):
        """
        Creates all summaries for the model.
        Args:
          data: InputEndpoints namedtuple.
          endpoints: OutputEndpoints namedtuple.
          charset: A dictionary with mapping between character codes and
            unicode characters. Use the one provided by a dataset.charset.
          is_training: If True will create summary prefixes for training job,
            otherwise - for evaluation.
        Returns:
          A list of evaluation ops
        """

        def sname(label):
            prefix = 'train' if is_training else 'eval'
            return '%s/%s' % (prefix, label)

        max_outputs = 4

        charset_mapper = CharsetMapper(charset)
        pr_text = charset_mapper.get_text(endpoints.predicted_chars[:max_outputs, :])
        tf.summary.text(sname('text/pr'), pr_text)
        gt_text = charset_mapper.get_text(data.labels[:max_outputs, :])
        tf.summary.text(sname('text/gt'), gt_text)
        tf.summary.image(sname('image'), data.images, max_outputs=max_outputs)

        if is_training:
            tf.summary.image(
                sname('image/orig'), data.images_orig, max_outputs=max_outputs)
            for var in tf.compat.v1.trainable_variables():
                tf.summary.histogram(var.op.name, var)
            return None

        else:
            names_to_values = {}
            names_to_updates = {}

            def use_metric(name, value_update_tuple):
                names_to_values[name] = value_update_tuple[0]
                names_to_updates[name] = value_update_tuple[1]

            use_metric('CharacterAccuracy', char_accuracy(endpoints.predicted_chars, data.labels, streaming=True,
                                                          rej_char=self._params.null_code))

            # Sequence accuracy computed by cutting sequence at the first null char
            use_metric('SequenceAccuracy', sequence_accuracy(endpoints.predicted_chars, data.labels, streaming=True,
                                                             rej_char=self._params.null_code))
            for name, value in names_to_values.items():
                summary_name = 'eval/' + name
                tf.summary.scalar(summary_name, tf.compat.v1.Print(value, [value], summary_name))
            return list(names_to_updates.values())

    @staticmethod
    def create_init_fn_to_restore(master_checkpoint,
                                  inception_checkpoint=None):
        """Creates an init operations to restore weights from various checkpoints.
        Args:
          master_checkpoint: path to a checkpoint which contains all weights for
            the whole model.
          inception_checkpoint: path to a checkpoint which contains weights for the
            inception part only.
        Returns:
          a function to run initialization ops.
        """
        all_assign_ops = []
        all_feed_dict = {}

        def assign_from_checkpoint(variables, checkpoint):
            logging.info('Request to re-store %d weights from %s',
                         len(variables), checkpoint)
            if not variables:
                logging.error('Can\'t find any variables to restore.')
                sys.exit(1)
            assign_op, feed_dict = tf_slim.assign_from_checkpoint(checkpoint, variables)
            all_assign_ops.append(assign_op)
            all_feed_dict.update(feed_dict)

        # logging.info('variables_to_restore:\n%s' % variables_to_restore().keys())
        # logging.info('moving_average_variables:\n%s' % [v.op.name for v in tf.compat.v1.moving_average_variables()])
        # logging.info('trainable_variables:\n%s' % [v.op.name for v in tf.compat.v1.trainable_variables()])
        if master_checkpoint:
            assign_from_checkpoint(variables_to_restore(), master_checkpoint)

        if inception_checkpoint:
            variables = variables_to_restore(
                'AttentionOcr/conv_tower_fn/INCE', strip_scope=True)
            assign_from_checkpoint(variables, inception_checkpoint)

        def init_assign_fn(sess):
            logging.info('Restoring checkpoint(s)')
            sess.run(all_assign_ops, all_feed_dict)

        return init_assign_fn
