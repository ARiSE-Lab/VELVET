import tensorflow as tf


class GGNN(tf.keras.layers.Layer):
	def __init__(self, model_config, shared_embedding=None, vocab_dim=None):
		super(GGNN, self).__init__()
		self.num_edge_types = model_config['num_edge_types']
		self.time_steps = model_config['time_steps']
		self.num_layers = len(self.time_steps)
		self.residuals = model_config['residuals']
		self.hidden_dim = model_config['hidden_dim']
		self.add_type_bias = model_config['add_type_bias']
		self.dropout_rate = model_config['dropout_rate']

	def build(self, _):
		# Small util functions
		random_init = tf.random_normal_initializer(stddev=self.hidden_dim ** -0.5)
		def make_weight(name=None):
			return tf.Variable(random_init([self.hidden_dim, self.hidden_dim]), name=name)
		def make_bias(name=None):
			return tf.Variable(random_init([self.hidden_dim]), name=name)
		
		# Set up type-transforms and GRUs
		self.type_weights = [[make_weight('type-' + str(j) + '-' + str(i)) for i in range(self.num_edge_types)] for j in range(self.num_layers)]
		self.type_biases = [[make_bias('bias-' + str(j) + '-' + str(i)) for i in range(self.num_edge_types)] for j in range(self.num_layers)]
		self.rnns = [tf.keras.layers.GRUCell(self.hidden_dim) for _ in range(self.num_layers)]
		for ix, rnn in enumerate(self.rnns):
			# Initialize the GRUs input dimension based on whether any residuals will be passed in.
			if str(ix) in self.residuals:
				rnn.build(self.hidden_dim*(1 + len(self.residuals[str(ix)])))
			else:
				rnn.build(self.hidden_dim)

	@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, 4), dtype=tf.int32), tf.TensorSpec(shape=(), dtype=tf.bool)])
	def call(self, states, edge_ids, training):
		# Collect some basic details about the graphs in the batch.
		edge_type_ids = tf.dynamic_partition(edge_ids[:, 1:], edge_ids[:, 0], self.num_edge_types)
		message_sources = [type_ids[:, 0:2] for type_ids in edge_type_ids]
		message_targets = [tf.stack([type_ids[:, 0], type_ids[:, 2]], axis=1) for type_ids in edge_type_ids]
		
		# Initialize the node_states with embeddings; then, propagate through layers and number of time steps for each layer.
		layer_states = [states]
		for layer_no, steps in enumerate(self.time_steps):
			for step in range(steps):
				if str(layer_no) in self.residuals:
					residuals = [layer_states[ix] for ix in self.residuals[str(layer_no)]]
				else:
					residuals = None
				new_states = self.propagate(layer_states[-1], layer_no, edge_type_ids, message_sources, message_targets, residuals=residuals)
				if training: new_states = tf.nn.dropout(new_states, rate=self.dropout_rate)
				# Add or overwrite states for this layer number, depending on the step.
				if step == 0:
					layer_states.append(new_states)
				else:
					layer_states[-1] = new_states
		# Return the final layer state.
		return layer_states[-1]
	
	def propagate(self, in_states, layer_no, edge_type_ids, message_sources, message_targets, residuals=None):
		# Collect messages across all edge types.
		messages = tf.zeros_like(in_states)
		for type_index in range(self.num_edge_types):
			type_ids = edge_type_ids[type_index]
			if tf.shape(type_ids)[0] == 0:
				continue
			# Retrieve source states and compute type-transformation.
			edge_source_states = tf.gather_nd(in_states, message_sources[type_index])
			type_messages = tf.matmul(edge_source_states, self.type_weights[layer_no][type_index])
			if self.add_type_bias:
				type_messages += self.type_biases[layer_no][type_index]
			messages = tf.tensor_scatter_nd_add(messages, message_targets[type_index], type_messages)
		
		# Concatenate residual messages, if applicable.
		if residuals is not None:
			messages = tf.concat(residuals + [messages], axis=2)
		
		# Run GRU for each node.
		new_states, _ = self.rnns[layer_no](messages, tf.expand_dims(in_states, 0))
		return new_states[0]
