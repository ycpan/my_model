import tensorflow as tf
from my_model.embeddings.bert_base.bert import modeling
from sklearn.feature_extraction.text import CountVectorizer
def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
  """Looks up words embeddings for id tensor.

  Args:
    input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
      ids.
    vocab_size: int. Size of the embedding vocabulary.
    embedding_size: int. Width of the word embeddings.
    initializer_range: float. Embedding initialization range.
    word_embedding_name: string. Name of the embedding table.
    use_one_hot_embeddings: bool. If True, use one-hot method for word
      embeddings. If False, use `tf.nn.embedding_lookup()`. One hot is better
      for TPUs.

  Returns:
    float Tensor of shape [batch_size, seq_length, embedding_size].
  """
  # This function assumes that the input is of shape [batch_size, seq_length,
  # num_inputs].
  #
  # If the input is a 2D tensor of shape [batch_size, seq_length], we
  # reshape to [batch_size, seq_length, 1].
  if input_ids.shape.ndims == 2:
    input_ids = tf.expand_dims(input_ids, axis=[-1])

  embedding_table = tf.get_variable(
      name=word_embedding_name,
      shape=[vocab_size, embedding_size],
      initializer=create_initializer(initializer_range))

  if use_one_hot_embeddings:
    flat_input_ids = tf.reshape(input_ids, [-1])
    one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
    output = tf.matmul(one_hot_input_ids, embedding_table)
  else:
    output = tf.nn.embedding_lookup(embedding_table, input_ids)

  #input_shape = get_shape_list(input_ids)

  #output = tf.reshape(output,
  #                    input_shape[0:-1] + [input_shape[-1] * embedding_size])
  return (output, embedding_table)


def embedding(inputs, vocab_size, args,zero_pad=True, scale=True, scope="embedding", reuse=None):
    """Embeds a given tensor.
    Args:
        inputs: A `Tensor` with type `int32` or `int64` containing the ids to be looked up in `lookup table`.
        vocab_size: An int. Vocabulary size.
        num_units: An int. Number of embedding hidden units.
        zero_pad: A boolean. If True, all the values of the fist row (id 0) should be constant zeros.
        scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
        scope: Optional scope for `variable_scope`.
        reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
        A `Tensor` with one more rank than inputs's. The last dimensionality should be `num_units`.

    For example,

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
        [ 0.09754146  0.67385566]
        [ 0.37864095 -0.35689294]]

    [[-1.01329422 -1.09939694]
        [ 0.7521342   0.38203377]
        [-0.04973143 -0.06210355]]]
    ```

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
        [[[-0.19172323 -0.39159766]
            [-0.43212751 -0.66207761]
            [ 1.03452027 -0.26704335]]

        [[-0.11634696 -0.35983452]
            [ 0.50208133  0.53509563]
            [ 1.22204471 -0.96587461]]]
    ```
    """
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table', dtype=tf.float32, shape=[vocab_size, args.dim],
                                       #initializer=tf.contrib.layers.xavier_initializer())
                                       initializer=tf.truncated_normal_initializer(stddev=args.embedding_stddev))
        if zero_pad:
            lookup_table = tf.concat((lookup_table, tf.zeros(shape=[1, args.dim])), 0)
        #outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        #if scale:
        #    outputs = outputs * (num_units ** 0.5)
        outputs = tf.gather(lookup_table,inputs)

    return outputs


def get_bert_embedding(bert_config_file,is_training,input_ids,mask,segment_ids,use_one_hot_embeddings):
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    model = modeling.BertModel(
        config = bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
        )

    output_layer = model.get_sequence_output()
    return output_layer
def get_count_embedding(vectorizer,corpus,vocab):
   X = vectorizer.fit_transform(corpus) 
   #vectorizer.vocabulary_.get(vocab)
   return X.toarray()
