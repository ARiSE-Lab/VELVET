import os
import random
import json
import tensorflow as tf


class DataLoader():

    def __init__(self, data_path, data_config):
        self.data_path = data_path
        self.config = data_config
        # self.vocabulary = vocabulary

    def batcher(self, mode="train"):
        data_path = self.get_data_path(mode)
        dataset = tf.data.Dataset.list_files(data_path + '/*.txt', shuffle=mode != 'eval', seed=42)
        dataset = dataset.interleave(lambda x: tf.data.TextLineDataset(x).shuffle(
            buffer_size=1000) if mode == 'train' else tf.data.TextLineDataset(x), cycle_length=4, block_length=16)
        dataset = dataset.prefetch(1)
        if mode == "train":
            dataset = dataset.repeat()
        # debug = self.to_batch(dataset, mode)
        # for d in debug:
            # print(tf.size(d)[0])
        ds = tf.data.Dataset.from_generator(lambda mode: self.to_batch(dataset, mode), output_types=(
        tf.dtypes.float32, tf.dtypes.int32, tf.dtypes.int32, tf.dtypes.int32, tf.dtypes.int32), args=(mode,))
        ds = ds.prefetch(1)
        return ds

    def get_data_path(self, mode):
        if mode == "train":
            return os.path.join(self.data_path, "train")
        elif mode == "dev":
            return os.path.join(self.data_path, "dev")
        elif mode == "eval":
            return os.path.join(self.data_path, "eval")
        else:
            raise ValueError("Mode % not supported for batching; please use \"train\", \"dev\", or \"eval\".")

    def to_sample(self, json_data):
        def parse_edges(edges):
            # Reorder edges to [edge type, source, target] and double edge type index to allow reverse edges
            # relations = [[2 * EDGE_TYPES[rel[3]], rel[0], rel[1]] for rel in edges if rel[
            # 3] in EDGE_TYPES]  # Note: we reindex edge types to be 0-based and filter unsupported edge types (useful for ablations)
            # relations += [[rel[0] + 1, rel[2], rel[1]] for rel in relations]  # Add reverse edges
            relations = [[rel[1], rel[0], rel[2]] for rel in edges]
            return relations

        # tokens = [self.vocabulary.translate(t)[:self.config["max_token_length"]] for t in json_data["source_tokens"]]
        nodes = json_data["node_features"]
        edges = parse_edges(json_data["graph"])
        error_location = json_data["targets"][0][0]
        map_line = json_data["map_line"]
        sample_idx = json_data["index"]
        # repair_targets = json_data["repair_targets"]
        # repair_candidates = [t for t in json_data["repair_candidates"] if isinstance(t, int)]
        # return (tokens, edges, error_location, repair_targets, repair_candidates)
        return (nodes, edges, error_location, map_line, sample_idx)

    def to_batch(self, sample_generator, mode):
        if isinstance(mode, bytes): mode = mode.decode('utf-8')
        def sample_len(sample):
            return len(sample[0])

        def make_batch(buffer):
            pivot = sample_len(random.choice(buffer))
            buffer = sorted(buffer, key=lambda b: abs(sample_len(b) - pivot))
            batch = []
            max_seq_len = 0
            for sample in buffer:
                max_seq_len = max(max_seq_len, sample_len(sample))
                if max_seq_len * (len(batch) + 1) > self.config[
                    'max_batch_size']:  # if adding the current one will exceed the batch token size, then break
                    break
                batch.append(sample)
            batch_dim = len(batch)
            buffer = buffer[batch_dim:]
            batch = list(
                zip(*batch))  # change the length from the count of samples to the count of attributes of each sample

            token_tensor = tf.ragged.constant(batch[0], dtype=tf.dtypes.float32).to_tensor(shape=(len(batch[0]), max(len(b) for b in batch[0]), self.config["w2v_dimension"]))

            # Add batch axis to all edges and flatten
            edge_batches = tf.repeat(tf.range(batch_dim), [len(edges) for edges in batch[
                1]])  # the process of assigning each edge a batch label
            edge_tensor = tf.concat(batch[1], axis=0)
            edge_tensor = tf.stack([edge_tensor[:, 0], edge_batches, edge_tensor[:, 1], edge_tensor[:, 2]], axis=1)

            # Error location is just a simple constant list
            error_location = tf.constant(batch[2], dtype=tf.dtypes.int32)
            # map_node_tensor = tf.constant(batch[3], dtype=tf.dtypes.int32)
            map_line_tensor = tf.ragged.constant(batch[3], dtype=tf.dtypes.int32).to_tensor(shape=(len(batch[0]), max(len(b) for b in batch[0])))
            sample_index = tf.constant(batch[4], dtype=tf.dtypes.int32)



            return buffer, (token_tensor, edge_tensor, error_location, map_line_tensor, sample_index)

        buffer = []
        num_samples = 0
        for line in sample_generator:
            json_sample = json.loads(line.numpy())
            sample = self.to_sample(json_sample)
            if sample_len(sample) > self.config['max_node_size']: # let's firstly feed the nodes as sequence. Later should use tokens
                continue
            buffer.append(sample)
            num_samples += 1
            if mode == 'dev' and num_samples >= self.config['max_valid_samples']:
                break
            if sum(sample_len(sample) for l in buffer) > self.config['max_buffer_size'] * self.config['max_batch_size']:
                buffer, batch = make_batch(buffer)
                yield batch
                # return batch
        # Drain the buffer upon completion
        while buffer:
            buffer, batch = make_batch(buffer)
            yield batch
            # return batch
