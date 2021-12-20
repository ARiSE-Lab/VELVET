import sys
sys.path.append('.')

import argparse
import yaml

import numpy as np

from checkpoint_tracker import Tracker, TrackerFineTune
from meta_model import VloggrBase
from data_loader import *

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("data_path", help="Path to data root")
	ap.add_argument("config", help="Path to config file")
	ap.add_argument("-m", "--models", help="Directory to store trained models (optional)")
	ap.add_argument("-l", "--log", help="Path to store training log (optional)")
	ap.add_argument("-a", "--analysis_file", help="Path to store the analysis results")
	ap.add_argument("-e", "--eval_only", help="Whether to run just the final model evaluation")
	ap.add_argument("-f", "--fine_tune", help="Whether to fine-tune the pre-trained model")
	ap.add_argument("-t", "--ensemble_test", help="Whether use ensemble test")
	ap.add_argument("-x", "--model_x", help="Directory to restore trained model x for ensemble")
	ap.add_argument("-y", "--model_y", help="Directory to restore trained model y for ensemble")
	ap.add_argument("-p", "--log_p", help="Path to store training log p for ensemble")
	ap.add_argument("-q", "--log_q", help="Path to store training log q for ensemble")
	args = ap.parse_args()
	config = yaml.safe_load(open(args.config))
	print("Training with configuration:", config)
	data = DataLoader(args.data_path, config["data"])
	if args.eval_only:
		if args.models is None or args.log is None:
			raise ValueError("Must provide a path to pre-trained models when running final evaluation")
		if args.analysis_file:
			test(data, config, args.models, args.log, args.analysis_file)
		else:
			test(data, config, args.models, args.log)
	elif args.ensemble_test:
		if args.model_x is None or args.log_p is None or args.model_y is None or args.log_q is None:
			raise ValueError("Must provide two pretrained models for ensembling")
		ensemble_test(data, config, args.model_x, args.log_p, args.model_y, args.log_q, args.analysis_file)
	elif args.fine_tune:
		if args.models is None or args.log is None:
			raise ValueError("Must provide a path to pre-trained models when running fine tuning")
		fine_tune(data, config, args.models, args.log)
	else:
		train(data, config, args.models, args.log)


def fine_tune(data, config, model_path, log_path):
	model = VloggrBase(config['model'])
	model.run_dummy_input()
	print("Model initialized, training {:,} parameters".format(
		np.sum([np.prod(v.shape) for v in model.trainable_variables])))
	optimizer = tf.optimizers.Adam(config["training"]["learning_rate"])

	tracker = TrackerFineTune(model, model, model_path, log_path)
	tracker.restore(best_model=True)
	if tracker.ckpt_restore.step.numpy() > 0:
		print("Restored from step:", tracker.ckpt_restore.step.numpy() + 1)
	else:
		print("Step:", tracker.ckpt_restore.step.numpy() + 1)

	mbs = 0
	distances, losses, accs, counts = get_metrics()
	while tracker.ckpt_save.step < config["training"]["max_steps"]:
		# These are just for console logging, not global counts
		for batch in data.batcher(mode='train'):
			mbs += 1
			tokens, edges, error_loc, line_map, ids = batch
			# token_mask = tf.clip_by_value(tf.abs(tf.reduce_sum(tokens, -1)), 0, 1)
			token_mask = tf.cast(tf.not_equal(tf.reduce_sum(tokens, -1), tf.constant(0, dtype=tf.float32)), tf.float32)
			with tf.GradientTape() as tape:
				# tf.config.experimental_run_functions_eagerly(True)
				pointer_preds = model(tokens, token_mask, edges, training=True)
				dist, ls, acs, _ = model.get_loss(pointer_preds, token_mask, error_loc, line_map, ids)
				loss = ls
			grads = tape.gradient(loss, model.trainable_variables)
			grads, _ = tf.clip_by_global_norm(grads, 0.25)
			optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))

			# Update statistics
			num_buggy = tf.reduce_sum(tf.clip_by_value(error_loc, 0, 1))
			samples = tf.shape(token_mask)[0]
			prev_samples = tracker.get_samples()
			curr_samples = tracker.update_samples(samples)
			update_metrics(distances, losses, accs, counts, token_mask, dist, ls, acs, num_buggy)

			# Every few minibatches, print the recent training performance
			if mbs % config["training"]["print_freq"] == 0:
				avg_losses = ["{0:.3f}".format(l.result().numpy()) for l in losses]
				avg_accs = ["{0:.2%}".format(a.result().numpy()) for a in accs]
				print("MB: {0}, seqs: {1:,}, tokens: {2:,}, loss: {3}, accs: {4}".format(mbs, curr_samples,
																						 counts[1].result().numpy(),
																						 ", ".join(avg_losses),
																						 ", ".join(avg_accs)))
				[l.reset_states() for l in losses]
				[a.reset_states() for a in accs]

			# Every valid_interval samples, run an evaluation pass and store the most recent model with its heldout accuracy
			if prev_samples // config["data"]["valid_interval"] < curr_samples // config["data"]["valid_interval"]:
				avg_accs = evaluate(data, config, model)
				tracker.save_checkpoint(model, avg_accs)
				if tracker.ckpt_save.step >= config["training"]["max_steps"]:
					break
				else:
					print("Step:", tracker.ckpt_save.step.numpy() + 1)


def test(data, config, model_path, log_path, analy_path=None):
	model = VloggrBase(config['model'])
	model.run_dummy_input()
	tracker = Tracker(model, model_path, log_path)
	tracker.restore(best_model=True)
	evaluate(data, config, model, is_heldout=False, analy_path=analy_path)

def ensemble_test(data, config, model_path_1, log_path_1, model_path_2, log_path_2, analy_path):
	model_1 = VloggrBase(config['model'])
	model_2 = VloggrBase(config['model_2'])
	model_1.run_dummy_input()
	model_2.run_dummy_input()
	tracker_1 = Tracker(model_1, model_path_1, log_path_1)
	tracker_1.restore(best_model=True)
	tracker_2 = Tracker(model_2, model_path_2, log_path_2)
	tracker_2.restore(best_model=True)
	ensemble_evaluate(data, config, model_1, model_2, is_heldout=False, analy_path=analy_path)

def train(data, config, model_path=None, log_path=None):
	# tf.config.experimental_run_functions_eagerly(True)
	model = VloggrBase(config['model'])
	model.run_dummy_input()
	print("Model initialized, training {:,} parameters".format(
		np.sum([np.prod(v.shape) for v in model.trainable_variables])))
	optimizer = tf.optimizers.Adam(config["training"]["learning_rate"])

	# Restore model from checkpoints if present; also sets up logger
	if model_path is None:
		tracker = Tracker(model)
	else:
		tracker = Tracker(model, model_path, log_path)
	tracker.restore()
	if tracker.ckpt.step.numpy() > 0:
		print("Restored from step:", tracker.ckpt.step.numpy() + 1)
	else:
		print("Step:", tracker.ckpt.step.numpy() + 1)

	mbs = 0
	distances, losses, accs, counts = get_metrics()
	while tracker.ckpt.step < config["training"]["max_steps"]:
	# These are just for console logging, not global counts
		for batch in data.batcher(mode='train'):
			mbs += 1
			tokens, edges, error_loc, line_map, ids = batch
			# token_mask = tf.clip_by_value(tf.abs(tf.reduce_sum(tokens, -1)), 0, 1)
			token_mask = tf.cast(tf.not_equal(tf.reduce_sum(tokens, -1), tf.constant(0, dtype=tf.float32)), tf.float32)
			with tf.GradientTape() as tape:
				# tf.config.experimental_run_functions_eagerly(True)
				pointer_preds = model(tokens, token_mask, edges, training=True)
				dist, ls, acs, _ = model.get_loss(pointer_preds, token_mask, error_loc, line_map, ids)
				loss = ls
			grads = tape.gradient(loss, model.trainable_variables)
			grads, _ = tf.clip_by_global_norm(grads, 0.25)
			optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))

			# Update statistics
			num_buggy = tf.reduce_sum(tf.clip_by_value(error_loc, 0, 1))
			samples = tf.shape(token_mask)[0]
			prev_samples = tracker.get_samples()
			curr_samples = tracker.update_samples(samples)
			update_metrics(distances, losses, accs, counts, token_mask, dist, ls, acs, num_buggy)

			# Every few minibatches, print the recent training performance
			if mbs % config["training"]["print_freq"] == 0:
				avg_losses = ["{0:.3f}".format(l.result().numpy()) for l in losses]
				avg_accs = ["{0:.2%}".format(a.result().numpy()) for a in accs]
				print("MB: {0}, seqs: {1:,}, tokens: {2:,}, loss: {3}, accs: {4}".format(mbs, curr_samples,
																						 counts[1].result().numpy(),
																						 ", ".join(avg_losses),
																						 ", ".join(avg_accs)))
				[l.reset_states() for l in losses]
				[a.reset_states() for a in accs]

			# Every valid_interval samples, run an evaluation pass and store the most recent model with its heldout accuracy
			if prev_samples // config["data"]["valid_interval"] < curr_samples // config["data"]["valid_interval"]:
				avg_accs = evaluate(data, config, model)
				tracker.save_checkpoint(model, avg_accs)
				if tracker.ckpt.step >= config["training"]["max_steps"]:
					break
				else:
					print("Step:", tracker.ckpt.step.numpy() + 1)


def ensemble_evaluate(data, config, model_1, model_2, is_heldout=False, analy_path=None):
	print("Testing pre-trained model on full eval data")
	distances, losses, accs, counts = get_metrics()
	mbs = 0
	# tf.config.experimental_run_functions_eagerly(True)
	analy_results = list()
	for batch in data.batcher(mode='dev' if is_heldout else 'eval'):
		ana_res = dict()
		mbs += 1
		tokens, edges, error_loc, line_map, ids = batch
		# token_mask = tf.clip_by_value(tf.abs(tf.reduce_sum(tokens, -1)), 0, 1)
		token_mask = tf.cast(tf.not_equal(tf.reduce_sum(tokens, -1), tf.constant(0, dtype=tf.float32)), tf.float32)
		pointer_preds_1 = model_1(tokens, token_mask, edges, training=False)
		pointer_preds_2 = model_2(tokens, token_mask, edges, training=False)
		dist, ls, acs, correctIDs = ensemble_results(pointer_preds_1, pointer_preds_2, token_mask, error_loc, line_map, ids, config["model"])
		ana_res["batch_shape"] = tf.shape(token_mask).numpy().tolist()
		ana_res["correct_preds"] = [i.numpy().tolist() for i in correctIDs]
		analy_results.append(ana_res)
		if not is_heldout:
			print(ana_res)
		num_buggy = tf.reduce_sum(tf.clip_by_value(error_loc, 0, 1))
		update_metrics(distances, losses, accs, counts, token_mask, dist, ls, acs, num_buggy)
		if is_heldout and counts[0].result() > config['data']['max_valid_samples']:
			break
		if not is_heldout and mbs % config["training"]["print_freq"] == 0:
			avg_distance = "{0:.3f}".format(distances.result().numpy())
			avg_losses = ["{0:.3f}".format(l.result().numpy()) for l in losses]
			avg_accs = ["{0:.2%}".format(a.result().numpy()) for a in accs]
			print("Testing progress: MB: {0}, seqs: {1:,}, tokens: {2:,}, loss: {3}, accs: {4}, distances: {5}".format(
				mbs, counts[
					0].result().numpy(), counts[1].result().numpy(), ", ".join(avg_losses), ", ".join(avg_accs),
				avg_distance))

	avg_distance = "{0:.3f}".format(distances.result().numpy())
	avg_accs = [a.result().numpy() for a in accs]
	avg_accs_str = ", ".join(["{0:.2%}".format(a) for a in avg_accs])
	avg_loss_str = ", ".join(["{0:.3f}".format(l.result().numpy()) for l in losses])
	print("Evaluation result: seqs: {0:,}, tokens: {1:,}, loss: {2}, accs: {3}, distances: {4}".format(
		counts[0].result().numpy(),
		counts[1].result().numpy(),
		avg_loss_str, avg_accs_str, avg_distance))
	if analy_path is not None:
		with open(analy_path, 'w') as ap:
			json.dump(analy_results, ap, indent=2)

	return avg_accs

def evaluate(data, config, model, is_heldout=True, analy_path=None):  # Similar to train, just without gradient updates
	if is_heldout:
		print("Running evaluation pass on heldout data")
	else:
		print("Testing pre-trained model on full eval data")

	distances, losses, accs, counts = get_metrics()
	mbs = 0
	# tf.config.experimental_run_functions_eagerly(True)
	analy_results = list()
	for batch in data.batcher(mode='dev' if is_heldout else 'eval'):
		ana_res = dict()
		mbs += 1
		tokens, edges, error_loc, line_map, ids = batch
		# token_mask = tf.clip_by_value(tf.abs(tf.reduce_sum(tokens, -1)), 0, 1)
		token_mask = tf.cast(tf.not_equal(tf.reduce_sum(tokens, -1), tf.constant(0, dtype=tf.float32)), tf.float32)
		pointer_preds = model(tokens, token_mask, edges, training=False)
		dist, ls, acs, correctIDs = model.get_loss(pointer_preds, token_mask, error_loc, line_map, ids)
		ana_res["batch_shape"] = tf.shape(token_mask).numpy().tolist()
		ana_res["correct_preds"] = [i.numpy().tolist() for i in correctIDs]
		analy_results.append(ana_res)
		if not is_heldout:
			print(ana_res)
		num_buggy = tf.reduce_sum(tf.clip_by_value(error_loc, 0, 1))
		update_metrics(distances, losses, accs, counts, token_mask, dist, ls, acs, num_buggy)
		if is_heldout and counts[0].result() > config['data']['max_valid_samples']:
			break
		if not is_heldout and mbs % config["training"]["print_freq"] == 0:
			avg_distance = "{0:.3f}".format(distances.result().numpy())
			avg_losses = ["{0:.3f}".format(l.result().numpy()) for l in losses]
			avg_accs = ["{0:.2%}".format(a.result().numpy()) for a in accs]
			print("Testing progress: MB: {0}, seqs: {1:,}, tokens: {2:,}, loss: {3}, accs: {4}, distances: {5}".format(mbs, counts[
				0].result().numpy(), counts[1].result().numpy(), ", ".join(avg_losses), ", ".join(avg_accs), avg_distance))

	avg_distance = "{0:.3f}".format(distances.result().numpy())
	avg_accs = [a.result().numpy() for a in accs]
	avg_accs_str = ", ".join(["{0:.2%}".format(a) for a in avg_accs])
	avg_loss_str = ", ".join(["{0:.3f}".format(l.result().numpy()) for l in losses])
	print("Evaluation result: seqs: {0:,}, tokens: {1:,}, loss: {2}, accs: {3}, distances: {4}".format(counts[0].result().numpy(),
																					   counts[1].result().numpy(),
																					   avg_loss_str, avg_accs_str, avg_distance))
	if analy_path is not None:
		with open(analy_path, 'w') as ap:
			json.dump(analy_results, ap, indent=2)

	return avg_accs


def ensemble_results(predictions_1, predictions_2, token_mask, error_locations, map_line_tensor, ids, config):
	# Mask out infeasible tokens in the logits
	seq_mask = token_mask
	predictions = 0.35 * predictions_1 + 0.65 * predictions_2
	predictions += (1.0 - tf.expand_dims(seq_mask, 1)) * tf.float32.min

	# Localization loss is simply calculated with sparse CE
	loc_predictions = predictions[:, 0]
	pred_locs = tf.argmax(loc_predictions, axis=-1, output_type=tf.int32)

	loc_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(error_locations, loc_predictions)
	loc_loss = tf.reduce_mean(loc_loss)
	top_k = config['evaluation']['top']
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


def get_metrics():
	distances = tf.keras.metrics.Mean()
	losses = [tf.keras.metrics.Mean() for _ in range(1)]
	accs = [tf.keras.metrics.Mean()]
	counts = [tf.keras.metrics.Sum(dtype='int32') for _ in range(2)]
	return distances, losses, accs, counts

def update_metrics(distances, losses, accs, counts, token_mask, dist, ls, acs, num_buggy_samples):
	loc_loss = ls
	overall_acc = acs
	num_samples = tf.shape(token_mask)[0]
	distances.update_state(dist, sample_weight=num_samples)
	counts[0].update_state(num_samples)
	counts[1].update_state(tf.reduce_sum(token_mask))
	losses[0].update_state(loc_loss)
	accs[0].update_state(overall_acc, sample_weight=num_samples)


if __name__ == '__main__':
	main()