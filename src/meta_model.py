import tensorflow as tf

from models import great_transformer, ggnn, hoppity, util


class VloggrBase(tf.keras.layers.Layer):
    def __init__(self, config):
        super(VloggrBase, self).__init__()
        self.config = config

    def build(self, _):
        self.prediction = tf.keras.layers.Dense(1)  # single pointer: bug location

        self.pos_enc = tf.constant(util.positional_encoding(self.config['base']['hidden_dim'], 5000))

        join_dicts = lambda d1, d2: {**d1, **d2}
        base_config = self.config['base']
        desc = self.config['configuration'].split(' ')
        self.stack = []
        for kind in desc:
            if kind == 'ggnn':
                self.stack.append(ggnn.GGNN(join_dicts(self.config['ggnn'], base_config)))
            elif kind == 'great':
                self.stack.append(great_transformer.Transformer(join_dicts(self.config['transformer'], base_config)))
            elif kind == 'transformer':
                joint_config = join_dicts(self.config['transformer'], base_config)
                joint_config['num_edge_types'] = None
                self.stack.append(great_transformer.Transformer(joint_config))
            elif kind == 'hoppity':
                self.stack.append(hoppity.Hoppity(join_dicts(self.config['ggnn'], base_config)))  #share the same config with GGNN
            else:
                raise ValueError('Unknown model component provided:', kind)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
                                  tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                                  tf.TensorSpec(shape=(None, 4), dtype=tf.int32),
                                  tf.TensorSpec(shape=(), dtype=tf.bool)])
    def call(self, tokens, token_mask, edges, training):
        states = tokens

        states += self.pos_enc[:tf.shape(states)[1]]
        # Pass states through all the models (may be empty) in the parsed stack.
        for model in self.stack:
            if isinstance(model, ggnn.GGNN):  # For GGNNs, pass edges as-is
                states = model(states, edges, training=training)
            elif isinstance(model,
                            great_transformer.Transformer):  # For Transformers, reverse edge directions to match query-key direction and add attention mask.
                mask = tf.cast(token_mask, dtype='float32')
                mask = tf.expand_dims(tf.expand_dims(mask, 1), 1)
                attention_bias = tf.stack([edges[:, 0], edges[:, 1], edges[:, 3], edges[:, 2]], axis=1)
                states = model(states, mask, attention_bias,
                               training=training)  # Note that plain transformers will simply ignore the attention_bias.

            elif isinstance(model, hoppity.Hoppity):
                states = model(states, edges, training=training)

                states_mask = tf.expand_dims(token_mask, axis=-1)  # [batch, seq-len, 1]
                states_mask = tf.repeat(states_mask, repeats=tf.shape(states)[2],
                                        axis=-1)  # [batch, seq-len, dimension]
                states_masked = states + (1.0 - states_mask) * tf.float32.min
                graph_reps = tf.reduce_max(states_masked, axis=1)  # [batch, dimension]
                batch_seq_length = tf.shape(states)[1]
                graph_reps = tf.expand_dims(graph_reps, axis=1)
                batch_graph_reps = tf.repeat(graph_reps, repeats=batch_seq_length, axis=1)
                predictions = tf.reduce_sum(tf.multiply(states, batch_graph_reps), axis=-1)
                predictions = tf.expand_dims(predictions, axis=1)
                return predictions
            else:
                raise ValueError('Model not yet supported:', model)

        predictions = tf.transpose(self.prediction(states),
                        [0, 2, 1])  # Convert to [batch, 1, seq-length] for convenience.
        return predictions

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 1, None), dtype=tf.float32),
                                  tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                                  tf.TensorSpec(shape=(None,), dtype=tf.int32),
                                  tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                                  tf.TensorSpec(shape=(None, 2), dtype=tf.int32)])
    def get_loss(self, predictions, token_mask, error_locations, map_line_tensor, ids):
        # Mask out infeasible tokens in the logits
        seq_mask = token_mask
        predictions += (1.0 - tf.expand_dims(seq_mask, 1)) * tf.float32.min

        loc_predictions = predictions[:, 0]
        pred_locs = tf.argmax(loc_predictions, axis=-1, output_type=tf.int32)

        loc_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(error_locations, loc_predictions)
        loc_loss = tf.reduce_mean(loc_loss)
        top_k = self.config['evaluation']['top']
        if top_k > 1:
            loc_accs = tf.keras.metrics.sparse_top_k_categorical_accuracy(error_locations, loc_predictions, top_k)
            avg_loc_distances = tf.constant(0)
        else:
            error_loc = tf.stack([tf.range(tf.size(error_locations)), error_locations], axis=1)
            error_line = tf.gather_nd(map_line_tensor, error_loc)

            pred_locs = tf.stack([tf.range(tf.size(pred_locs)), pred_locs], axis=1)
            pred_line = tf.gather_nd(map_line_tensor, pred_locs)
            loc_accs = tf.cast(tf.equal(error_line, pred_line), dtype=tf.float32)
            # get the prediction distance at line level
            loc_distances = tf.cast(tf.math.abs(pred_line - error_line), dtype=tf.float32)
            avg_loc_distances = tf.reduce_mean(loc_distances)

        correct_indices = tf.boolean_mask(ids, tf.cast(loc_accs, dtype=tf.bool))

        total_loc_acc = tf.reduce_sum(loc_accs) / tf.cast(tf.shape(error_locations)[0], 'float32')

        return avg_loc_distances, loc_loss, total_loc_acc, correct_indices

    # Used to initialize the model's variables
    def run_dummy_input(self):
        self(tf.ones((3, 3, 256), dtype='float32'), tf.ones((3, 3), dtype='float32'), tf.ones((2, 4), dtype='int32'), True)
