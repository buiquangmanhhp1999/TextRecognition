"""
A sequence layer produce a sequence of characters from extracted image features. All of them are using RNNs.
This modules provides implementation which uses 'attention' mechanism to spatially 'pool' image features and also can use
a previously predicted character to predict the next
"""
import collections
import abc
import logging
import tensorflow as tf
import tf_slim
from libraries import orthogonal_initializer, rnn_decoder, attention_decoder


SequenceLayerParams = collections.namedtuple('SequenceLogitsParams', [
    'num_lstm_units', 'weight_decay', 'lstm_state_clip_value'
])


class SequenceLayerBase(object):
    def __init__(self, net, labels_one_hot, model_params, method_params):
        """
        Stores arguments in member variable for further use
        :param net: shape [batch_size, num_features, feature_size] which contains some extracted image features
        :param labels_one_hot: [batch_size, seq_length, num_char_classes]- ground truth labels for the input features
        :param model_params: a namedtuple with model parameters
        :param method_params: A SequenceLayerParams
        """
        self._params = model_params
        self._mparams = method_params
        self._net = net
        self._labels_one_hot = labels_one_hot
        self._batch_size = net.get_shape().dims[0]

        # Initialize parameters for char logits which will be computed on the fly
        # inside an LSTM decoder.
        self._char_logits = {}
        regularizer = tf_slim.l2_regularizer(self._mparams.weight_decay)

        self._softmax_w = tf_slim.model_variable(
            'softmax_w',
            [self._mparams.num_lstm_units, self._params.num_char_classes],
            initializer=orthogonal_initializer,
            regularizer=regularizer)

        self._softmax_b = tf_slim.model_variable(
            'softmax_b', [self._params.num_char_classes],
            initializer=tf.zeros_initializer(),
            regularizer=regularizer)

    @abc.abstractmethod
    def get_train_input(self, prev, i):
        """Returns a sample to be used to predict a character during training.
        This function is used as a loop_function for an RNN decoder. Use in training case
        Args:
          prev: output tensor from previous step of the RNN. A tensor with shape:
            [batch_size, num_char_classes].
          i: index of a character in the output sequence.
        Returns:
          A tensor with shape [batch_size, ?] - depth depends on implementation
          details.
        """
        pass

    @abc.abstractmethod
    def get_eval_input(self, prev, i):
        """Returns a sample to be used to predict a character during inference.
        This function is used as a loop_function for an RNN decoder. Use in test case
        Args:
          prev: output tensor from previous step of the RNN. A tensor with shape:
            [batch_size, num_char_classes].
          i: index of a character in the output sequence.
        Returns:
          A tensor with shape [batch_size, ?] - depth depends on implementation
          details.
        """
        raise AssertionError('Not implemented')

    @abc.abstractmethod
    def unroll_cell(self, decoder_inputs, initial_state, loop_function, cell):
        """Unrolls an RNN cell for all inputs.
        This is a placeholder to call some RNN decoder. It has a similar to
        tf.seq2seq.rnn_decode interface.
        Args:
          decoder_inputs: A list of 2D Tensors* [batch_size x input_size]. In fact,
            most of existing decoders in presence of a loop_function use only the
            first element to determine batch_size and length of the list to
            determine number of steps.
          initial_state: 2D Tensor with shape [batch_size x cell.state_size].
          loop_function: function will be applied to the i-th output in order to
            generate the i+1-st input (see self.get_input).
          cell: rnn_cell.RNNCell defining the cell function and size.
        Returns:
          A tuple of the form (outputs, state), where:
            outputs: A list of character logits of the same length as
            decoder_inputs of 2D Tensors with shape [batch_size x num_characters].
            state: The state of each cell at the final time-step.
              It is a 2D Tensor of shape [batch_size x cell.state_size].
        """
        pass

    def is_training(self):
        """Returns True if the layer is created for training stage."""
        return self._labels_one_hot is not None

    def char_logit(self, inputs, char_index):
        """Creates logits for a character if required.
        Args:
          inputs: A tensor with shape [batch_size, ?] (depth is implementation
            dependent).
          char_index: A integer index of a character in the output sequence.
        Returns:
          A tensor with shape [batch_size, num_char_classes]
        """
        if char_index not in self._char_logits:
            self._char_logits[char_index] = tf.compat.v1.nn.xw_plus_b(inputs, self._softmax_w,
                                                                      self._softmax_b)
        return self._char_logits[char_index]

    def char_one_hot(self, logit):
        """Creates one hot encoding for a logit of a character.
        Args:
          logit: A tensor with shape [batch_size, num_char_classes].
        Returns:
          A tensor with shape [batch_size, num_char_classes]
        """
        prediction = tf.argmax(logit, axis=1)
        return tf_slim.one_hot_encoding(prediction, self._params.num_char_classes)

    def get_input(self, prev, i):
        """A wrapper for get_train_input and get_eval_input.
        Args:
          prev: output tensor from previous step of the RNN. A tensor with shape:
            [batch_size, num_char_classes].
          i: index of a character in the output sequence.
        Returns:
          A tensor with shape [batch_size, ?] - depth depends on implementation
          details.
        """
        if self.is_training():
            return self.get_train_input(prev, i)
        else:
            return self.get_eval_input(prev, i)

    def create_logits(self):
        """
        Creates character sequence logits for a net specified in the constructor.
        A "main" method for the sequence layer which glues together all pieces

        :return:
            A tensor with shape [batch_size, seq_length, num_char_classes]
        """
        # because default is orthogonal_initializer, so don't use orthogonal_initializer
        with tf.compat.v1.variable_scope("LSTM"):
            first_label = self.get_input(prev=None, i=0)
            decoder_inputs = [first_label] + [None] * (self._params.seq_length - 1)
            # lstm_cell = tf.keras.layers.LSTMCell(units=self._mparams.num_lstm_units, recurrent_dropout=self._mparams.lstm_state_clip_value)

            lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(
                self._mparams.num_lstm_units,
                use_peepholes=False,
                cell_clip=self._mparams.lstm_state_clip_value,
                state_is_tuple=True,
                initializer=orthogonal_initializer)

            lstm_outputs, _ = self.unroll_cell(
                decoder_inputs=decoder_inputs,
                initial_state=lstm_cell.zero_state(self._batch_size, tf.float32),
                loop_function=self.get_input,
                cell=lstm_cell)

        with tf.compat.v1.variable_scope('logits'):
            logits_list = [
                tf.expand_dims(self.char_logit(logit, i), axis=1)
                for i, logit in enumerate(lstm_outputs)
            ]

        return tf.concat(logits_list, 1)


class NetSlice(SequenceLayerBase):
    """
    A layer which uses a subset of image features to predict each character.
    """

    def __init__(self, *args, **kwargs):
        super(NetSlice, self).__init__(*args, **kwargs)
        self._zero_label = tf.zeros(
            [self._batch_size, self._params.num_char_classes])

    def get_image_feature(self, char_index):
        """Returns a subset of image features for a character.
    Args:
      char_index: an index of a character.
    Returns:
      A tensor with shape [batch_size, ?]. The output depth depends on the
      depth of input net.
    """
        batch_size, features_num, _ = [d.value for d in self._net.get_shape()]
        slice_len = int(features_num / self._params.seq_length)
        # In case when features_num != seq_length, we just pick a subset of image
        # features, this choice is arbitrary and there is no intuitive geometrical
        # interpretation. If features_num is not dividable by seq_length there will
        # be unused image features.
        net_slice = self._net[:, char_index:char_index + slice_len, :]
        feature = tf.reshape(net_slice, [batch_size, -1])
        logging.debug('Image feature: %s', feature)
        return feature

    def get_eval_input(self, prev, i):
        """See SequenceLayerBase.get_eval_input for details."""
        del prev
        return self.get_image_feature(i)

    def get_train_input(self, prev, i):
        """See SequenceLayerBase.get_train_input for details."""
        return self.get_eval_input(prev, i)

    def unroll_cell(self, decoder_inputs, initial_state, loop_function, cell):
        """See SequenceLayerBase.unroll_cell for details."""
        return rnn_decoder(decoder_inputs=decoder_inputs, initial_state=initial_state, cell=cell,
                           loop_function=self.get_input)


class NetSliceWithAutoregression(NetSlice):
    """A layer similar to NetSlice, but it also uses auto regression.
  The "auto regression" means that we use network output for previous character
  as a part of input for the current character.
  """

    def __init__(self, *args, **kwargs):
        super(NetSliceWithAutoregression, self).__init__(*args, **kwargs)

    def get_eval_input(self, prev, i):
        """See SequenceLayerBase.get_eval_input for details."""
        if i == 0:
            prev = self._zero_label
        else:
            logit = self.char_logit(prev, char_index=i - 1)
            prev = self.char_one_hot(logit)
        image_feature = self.get_image_feature(char_index=i)
        return tf.concat([image_feature, prev], 1)

    def get_train_input(self, prev, i):
        """See SequenceLayerBase.get_train_input for details."""
        if i == 0:
            prev = self._zero_label
        else:
            prev = self._labels_one_hot[:, i - 1, :]
        image_feature = self.get_image_feature(i)
        return tf.concat([image_feature, prev], 1)


class Attention(SequenceLayerBase):
    """A layer which uses attention mechanism to select image features."""

    def __init__(self, *args, **kwargs):
        super(Attention, self).__init__(*args, **kwargs)
        self._zero_label = tf.zeros(
            [self._batch_size, self._params.num_char_classes])

    def get_eval_input(self, prev, i):
        """See SequenceLayerBase.get_eval_input for details."""
        del prev, i
        # The attention_decoder will fetch image features from the net, no need for
        # extra inputs.
        return self._zero_label

    def get_train_input(self, prev, i):
        """See SequenceLayerBase.get_train_input for details."""
        return self.get_eval_input(prev, i)

    def unroll_cell(self, decoder_inputs, initial_state, loop_function, cell):
        return attention_decoder(decoder_inputs=decoder_inputs, initial_state=initial_state, attention_states=self._net,
                                 cell=cell, loop_function=self.get_input)


class AttentionWithAutoregression(Attention):
    """A layer which uses both attention and auto regression."""

    def __init__(self, *args, **kwargs):
        super(AttentionWithAutoregression, self).__init__(*args, **kwargs)

    def get_train_input(self, prev, i):
        """See SequenceLayerBase.get_train_input for details."""
        if i == 0:
            return self._zero_label
        else:
            # TODO(gorban): update to gradually introduce gt labels.
            return self._labels_one_hot[:, i - 1, :]

    def get_eval_input(self, prev, i):
        """See SequenceLayerBase.get_eval_input for details."""
        if i == 0:
            return self._zero_label
        else:
            logit = self.char_logit(prev, char_index=i - 1)
            return self.char_one_hot(logit)


def get_layer_class(use_attention, use_autoregression):
    """A convenience function to get a layer class based on requirements.
  Args:
    use_attention: if True a returned class will use attention.
    use_autoregression: if True a returned class will use auto regression.
  Returns:
    One of available sequence layers (child classes for SequenceLayerBase).
  """
    if use_attention and use_autoregression:
        layer_class = AttentionWithAutoregression
    elif use_attention and not use_autoregression:
        layer_class = Attention
    elif not use_attention and not use_autoregression:
        layer_class = NetSlice
    elif not use_attention and use_autoregression:
        layer_class = NetSliceWithAutoregression
    else:
        raise AssertionError('Unsupported sequence layer class')

    logging.debug('Use %s as a layer class', layer_class.__name__)
    return layer_class