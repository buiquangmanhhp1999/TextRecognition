import tensorflow as tf
import numpy as np
from six.moves import xrange
import tf_slim

from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops, init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops


def Linear(args, output_size, bias, bias_initializer=None, kernel_initializer=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch, n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.
  Returns:
    A 2D Tensor with shape `[batch, output_size]` equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1] is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, "
                             "but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = tf.compat.v1.get_variable_scope()
    with tf.compat.v1.variable_scope(scope) as outer_scope:
        weights = tf.compat.v1.get_variable(
            "weights", [total_arg_size, output_size],
            dtype=dtype,
            initializer=kernel_initializer)
        if len(args) == 1:
            res = math_ops.matmul(args[0], weights)
        else:
            res = math_ops.matmul(array_ops.concat(args, 1), weights)
        if not bias:
            return res
        with tf.compat.v1.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            if bias_initializer is None:
                bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
            biases = tf.compat.v1.get_variable("biases", [output_size], dtype=dtype, initializer=bias_initializer)
        return nn_ops.bias_add(res, biases)


def orthogonal_initializer(shape, dtype=tf.float32, *args, **kwargs):
    """
  Generates orthonormal matrices with random values.
  Orthonormal initialization is important for RNNs:
    http://arxiv.org/abs/1312.6120
    http://smerity.com/articles/2016/orthogonal_init.html
  For non-square shapes the returned matrix will be semi-orthonormal: if the
  number of columns exceeds the number of rows, then the rows are orthonormal
  vectors; but if the number of rows exceeds the number of columns, then the
  columns are orthonormal vectors.
  We use SVD decomposition to generate an orthonormal matrix with random
  values. The same way as it is done in the Lasagne library for Theano. Note
  that both u and v returned by the svd are orthogonal and random. We just need
  to pick one with the right shape.
  Args:
    shape: a shape of the tensor matrix to initialize.
    dtype: a dtype of the initialized tensor.
    *args: not used.
    **kwargs: not used.
  Returns:
    An initialized tensor.
  """

    del args
    del kwargs
    flat_shape = (shape[0], np.prod(shape[1:]))
    w = np.random.randn(*flat_shape)
    u, _, v = np.linalg.svd(w, full_matrices=False)
    w = u if u.shape == flat_shape else v
    return tf.constant(w.reshape(shape), dtype=dtype)


def rnn_decoder(decoder_inputs, initial_state, cell, loop_function=None,
                scope=None):
    """RNN decoder for the sequence-to-sequence model.
  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor with shape [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    loop_function: If not None, this function will be applied to the i-th output
      in order to generate the i+1-st input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    scope: VariableScope for the created subgraph; defaults to "rnn_decoder".
  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing generated outputs.
      state: The state of each cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
        (Note that in some cases, like basic RNN cell or GRU cell, outputs and
         states can be the same. They are different for LSTM cells though.)
  """
    with tf.compat.v1.variable_scope.variable_scope(scope or "rnn_decoder"):
        state = initial_state
        outputs = []
        prev = None
        for i, inp in enumerate(decoder_inputs):
            if loop_function is not None and prev is not None:
                with tf.compat.v1.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            if i > 0:
                tf.compat.v1.get_variable_scope().reuse_variables()
            output, state = cell(inp, state)
            outputs.append(output)
            if loop_function is not None:
                prev = output
    return output, state


def attention_decoder(decoder_inputs, initial_state, attention_states, cell, output_size=None, num_heads=1,
                      loop_function=None, dtype=None, scope=None, initial_state_attention=False):
    """RNN decoder with attention for the sequence-to-sequence model.
  In this context "attention" means that, during decoding, the RNN can look up
  information in the additional tensor attention_states, and it does this by
  focusing on a few entries from the tensor. This model has proven to yield
  especially good results in a number of sequence-to-sequence tasks. This
  implementation is based on http://arxiv.org/abs/1412.7449 (see below for
  details). It is recommended for complex sequence-to-sequence tasks.
  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
    output_size: Size of the output vectors; if None, we use cell.output_size.
    num_heads: Number of attention heads that read from attention_states.
    loop_function: If not None, this function will be applied to i-th output
      in order to generate i+1-th input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    dtype: The dtype to use for the RNN initial state (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.
  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors of
        shape [batch_size x output_size]. These represent the generated outputs.
        Output i is computed from input i (which is either the i-th element
        of decoder_inputs or loop_function(output {i-1}, i)) as follows.
        First, we run the cell on a combination of the input and previous
        attention masks:
          cell_output, new_state = cell(linear(input, prev_attn), prev_state).
        Then, we calculate new attention masks:
          new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
        and then we calculate the output:
          output = linear(cell_output, new_attn).
      state: The state of each decoder cell the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  Raises:
    ValueError: when num_heads is not positive, there are no inputs, shapes
      of attention_states are not set, or input size cannot be inferred
      from the input.
  """
    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")
    if attention_states.get_shape()[2] is None:
        raise ValueError("Shape[2] of attention_states must be known: %s" %
                         attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size

    print("Decoder input: ", decoder_inputs[0].get_shape())
    print("attention state: ", attention_states.get_shape())
    with tf.compat.v1.variable_scope(
            scope or "attention_decoder", dtype=dtype) as scope:
        dtype = scope.dtype

        batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
        attn_length = attention_states.get_shape()[1]
        if attn_length is None:
            attn_length = array_ops.shape(attention_states)[1]
        attn_size = attention_states.get_shape()[2]

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = array_ops.reshape(attention_states,
                                   [-1, attn_length, 1, attn_size])
        hidden_features = []
        v = []
        attention_vec_size = attn_size  # Size of query vectors for attention.
        for a in xrange(num_heads):
            k = tf.compat.v1.get_variable("AttnW_%d" % a,
                                          [1, 1, attn_size, attention_vec_size])
            hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
            v.append(tf.compat.v1.get_variable("AttnV_%d" % a, [attention_vec_size]))

        state = initial_state

        def attention(query):
            """Put attention masks on hidden using hidden_features and query."""
            ds = []  # Results of attention reads will be stored here.
            if nest.is_sequence(query):  # If the query is a tuple, flatten it.
                query_list = nest.flatten(query)
                for q in query_list:  # Check that ndims == 2 if specified.
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                query = array_ops.concat(query_list, 1)
            for a in xrange(num_heads):
                with tf.compat.v1.variable_scope("Attention_%d" % a):
                    y = Linear(query, attention_vec_size, True)
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y),
                                            [2, 3])
                    a = nn_ops.softmax(s)
                    # Now calculate the attention-weighted vector d.
                    d = math_ops.reduce_sum(array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
                    ds.append(array_ops.reshape(d, [-1, attn_size]))
            return ds

        outputs = []
        prev = None
        batch_attn_size = array_ops.stack([batch_size, attn_size])
        attns = [
            array_ops.zeros(
                batch_attn_size, dtype=dtype) for _ in xrange(num_heads)
        ]
        for a in attns:  # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])
        if initial_state_attention:
            attns = attention(initial_state)
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                tf.compat.v1.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with tf.compat.v1.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            # Merge input and previous attentions into one vector of the right size.
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)

            inputs = [inp] + attns
            x = Linear(inputs, input_size, True)
            # Run the RNN.
            cell_output, state = cell(x, state)
            # Run the attention mechanism.
            if i == 0 and initial_state_attention:
                with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=True):
                    attns = attention(state)
            else:
                attns = attention(state)

            with tf.compat.v1.variable_scope("AttnOutputProjection"):
                inputs = [cell_output] + attns
                output = Linear(inputs, output_size, True)
            if loop_function is not None:
                prev = output
            outputs.append(output)

    return outputs, state


def logits_to_log_prob(logits):
    """Computes log probabilities using numerically stable trick.
  This uses two numerical stability tricks:
  1) softmax(x) = softmax(x - c) where c is a constant applied to all
  arguments. If we set c = max(x) then the softmax is more numerically
  stable.
  2) log softmax(x) is not numerically stable, but we can stabilize it
  by using the identity log softmax(x) = x - log sum exp(x)
  Args:
    logits: Tensor of arbitrary shape whose last dimension contains logits.
  Returns:
    A tensor of the same shape as the input, but with corresponding log
    probabilities.
  """

    with tf.compat.v1.variable_scope('log_probabilities'):
        reduction_indices = len(logits.shape.as_list()) - 1
        max_logits = tf.reduce_max(
            logits, axis=reduction_indices, keepdims=True)
        safe_logits = tf.subtract(logits, max_logits)
        sum_exp = tf.reduce_sum(
            tf.exp(safe_logits),
            axis=reduction_indices,
            keepdims=True)
        log_probs = tf.subtract(safe_logits, tf.math.log(sum_exp))
    return log_probs


def get_softmax_loss_fn(label_smoothing):
    """
    returns sparse or dense loss function depending on the label_smoothing
    :param label_rank:
    :param label_smoothing: weight for label smoothing
    :return:
        a function which takes label and predictions as arguments and returns
        a softmax loss for the selected type of labels (sparse or dense)
    """
    """
        if label_smoothing > 0 and label_rank == 3:
        def loss_fn(labels, logits):
            return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    elif label_rank == 2:
        def loss_fn(labels, logits):
            return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

    return loss_fn
    """
    if label_smoothing > 0:

        def loss_fn(labels, logits):
            return (tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=labels))
    else:

        def loss_fn(labels, logits):
            return tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels)

    return loss_fn


def char_accuracy(predictions, targets, rej_char, streaming=False):
    """Computes character level accuracy.
  Both predictions and targets should have the same shape
  [batch_size x seq_length].
  Args:
    predictions: predicted characters ids.
    targets: ground truth character ids.
    rej_char: the character id used to mark an empty element (end of sequence).
    streaming: if True, uses the streaming mean from the slim.metric module.
  Returns:
    a update_ops for execution and value tensor whose value on evaluation
    returns the total character accuracy.
  """
    with tf.compat.v1.variable_scope('CharAccuracy'):
        predictions.get_shape().assert_is_compatible_with(targets.get_shape())

        targets = tf.compat.v1.to_int32(targets)
        const_rej_char = tf.constant(rej_char, shape=targets.get_shape())
        weights = tf.compat.v1.to_float(tf.not_equal(targets, const_rej_char))
        correct_chars = tf.compat.v1.to_float(tf.equal(predictions, targets))
        accuracy_per_example = tf.compat.v1.div(
            tf.reduce_sum(tf.multiply(correct_chars, weights), 1),
            tf.reduce_sum(weights, 1))
        if streaming:
            return tf_slim.metrics.streaming_mean(accuracy_per_example)
        else:
            return tf.reduce_mean(accuracy_per_example)


def sequence_accuracy(predictions, targets, rej_char, streaming=False):
    """Computes sequence level accuracy.
  Both input tensors should have the same shape: [batch_size x seq_length].
  Args:
    predictions: predicted character classes.
    targets: ground truth character classes.
    rej_char: the character id used to mark empty element (end of sequence).
    streaming: if True, uses the streaming mean from the slim.metric module.
  Returns:
    a update_ops for execution and value tensor whose value on evaluation
    returns the total sequence accuracy.
  """

    with tf.compat.v1.variable_scope('SequenceAccuracy'):
        predictions.get_shape().assert_is_compatible_with(targets.get_shape())

        targets = tf.compat.v1.to_int32(targets)
        const_rej_char = tf.constant(
            rej_char, shape=targets.get_shape(), dtype=tf.int32)
        include_mask = tf.not_equal(targets, const_rej_char)
        include_predictions = tf.compat.v1.to_int32(
            tf.where(include_mask, predictions,
                     tf.zeros_like(predictions) + rej_char))
        correct_chars = tf.compat.v1.to_float(tf.equal(include_predictions, targets))
        correct_chars_counts = tf.cast(
            tf.reduce_sum(correct_chars, reduction_indices=[1]), dtype=tf.int32)
        target_length = targets.get_shape().dims[1].value
        target_chars_counts = tf.constant(
            target_length, shape=correct_chars_counts.get_shape())
        accuracy_per_example = tf.compat.v1.to_float(
            tf.equal(correct_chars_counts, target_chars_counts))
        if streaming:
            return tf_slim.metrics.streaming_mean(accuracy_per_example)
        else:
            return tf.reduce_mean(accuracy_per_example)


def variables_to_restore(scope=None, strip_scope=False):
    """Returns a list of variables to restore for the specified list of methods.
  It is supposed that variable name starts with the method's scope (a prefix
  returned by _method_scope function).
  Args:
    methods_names: a list of names of configurable methods.
    strip_scope: if True will return variable names without method's scope.
      If methods_names is None will return names unchanged.
    model_scope: a scope for a whole model.
  Returns:
    a dictionary mapping variable names to variables for restore.
  """
    if scope:
        variable_map = {}
        method_variables = tf_slim.get_variables_to_restore(include=[scope])
        for var in method_variables:
            if strip_scope:
                var_name = var.op.name[len(scope) + 1:]
            else:
                var_name = var.op.name
            variable_map[var_name] = var

        return variable_map
    else:
        return {v.op.name: v for v in tf_slim.get_variables_to_restore()}
