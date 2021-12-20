import os

import tensorflow as tf
import time

class Tracker(object):
	def __init__(self, model, model_path='checkpoints', log_path='log.txt', suffix=None):
		self.log_path = log_path
		self.model_path = model_path
		self.ckpt = tf.train.Checkpoint(model=model, step=tf.Variable(0), samples=tf.Variable(0), time=tf.Variable(0.0))
		self.manager = tf.train.CheckpointManager(self.ckpt, self.model_path, max_to_keep=None)
		self.log = []
	
	def restore(self, best_model=False, model_idx=0):
		self.log = []
		if self.manager.checkpoints and os.path.exists(self.log_path):
			with open(self.log_path) as f:
				for l in f:
					l = l.rstrip().split(': ')
					scores = [float(v.replace('%', ''))/100 if '%' in v else v for v in l[1].split(',')]
					self.log.append((l[0], scores))
			if best_model:
				# best = max(enumerate(self.log), key=lambda e: e[1][1][-1])[0]  # [-1] simply pulls the last accuracy value, which is the joint loc & rep accuracy
				best = sorted(enumerate(self.log), key=lambda e: e[1][1][-1], reverse=True)[model_idx][0]
				print("Restoring top model from step:", best + 1)
				status = self.ckpt.restore(self.manager.checkpoints[best])
			else:
				status = self.ckpt.restore(self.manager.latest_checkpoint)
			status.assert_existing_objects_matched()
			status.assert_consumed()
		self.time = time.time()
	
	def get_samples(self):
		return self.ckpt.samples.numpy()
	
	def update_samples(self, num_samples):
		self.ckpt.samples.assign_add(num_samples)
		return self.ckpt.samples.numpy()
	
	def save_checkpoint(self, model, scores):
		self.ckpt.step.assign_add(1)
		self.ckpt.time.assign_add(time.time() - self.time)
		self.time = time.time()
		
		s = self.ckpt.step.numpy()
		c = self.ckpt.samples.numpy()
		t = self.ckpt.time.numpy()
		self.log.append(((s, t), scores))
		with open(self.log_path, 'a') as f:
			f.write(str(s))
			f.write(', ')
			f.write(str(c))
			f.write(', ')
			f.write('{0:.3f}'.format(t))
			f.write(': ')
			f.write(', '.join(['{0:.2%}'.format(s) for s in scores]))
			f.write('\n')
		self.manager.save()


class TrackerFineTune(object):
	def __init__(self, model_restore, model_save, model_path='checkpoints_pretrain', log_path='log_pretrain.txt',
				 save_model_path='checkpoints', save_log_path='log.txt', suffix=None):
		self.log_path = log_path
		self.save_log_path = save_log_path
		self.model_path = model_path
		self.save_model_path = save_model_path
		self.ckpt_restore = tf.train.Checkpoint(model=model_restore, step=tf.Variable(0), samples=tf.Variable(0),
												time=tf.Variable(0.0))
		self.ckpt_save = tf.train.Checkpoint(model=model_save, step=tf.Variable(0), samples=tf.Variable(0),
											 time=tf.Variable(0.0))
		self.manager_restore = tf.train.CheckpointManager(self.ckpt_restore, self.model_path, max_to_keep=None)
		self.manager_save = tf.train.CheckpointManager(self.ckpt_save, self.save_model_path, max_to_keep=None)
		self.log = []

	def restore(self, best_model=False):
		if self.manager_restore.checkpoints and os.path.exists(self.log_path):
			with open(self.log_path) as f:
				for l in f:
					l = l.rstrip().split(': ')
					scores = [float(v.replace('%', '')) / 100 if '%' in v else v for v in l[1].split(',')]
					self.log.append((l[0], scores))
			if best_model:
				# best = max(enumerate(self.log), key=lambda e: e[1][1][-1])[0]  # [-1] simply pulls the last accuracy value, which is the joint loc & rep accuracy
				best = sorted(enumerate(self.log), key=lambda e: e[1][1][-1], reverse=True)[0][0]
				print("Restoring top model from step:", best + 1)
				status = self.ckpt_restore.restore(self.manager_restore.checkpoints[best])
			else:
				status = self.ckpt_restore.restore(self.manager_restore.latest_checkpoint)
			status.assert_existing_objects_matched()
			status.assert_consumed()
		self.time = time.time()

	def get_samples(self):
		return self.ckpt_save.samples.numpy()

	def update_samples(self, num_samples):
		self.ckpt_save.samples.assign_add(num_samples)
		return self.ckpt_save.samples.numpy()

	def save_checkpoint(self, model, scores):
		self.ckpt_save.step.assign_add(1)
		self.ckpt_save.time.assign_add(time.time() - self.time)
		self.time = time.time()

		s = self.ckpt_save.step.numpy()
		c = self.ckpt_save.samples.numpy()
		t = self.ckpt_save.time.numpy()
		self.log.append(((s, t), scores))
		with open(self.save_log_path, 'a') as f:
			f.write(str(s))
			f.write(', ')
			f.write(str(c))
			f.write(', ')
			f.write('{0:.3f}'.format(t))
			f.write(': ')
			f.write(', '.join(['{0:.2%}'.format(s) for s in scores]))
			f.write('\n')
		self.manager_save.save()

