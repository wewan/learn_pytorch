"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
# from torchviz import make_dot
import numpy as np
import os
from torchsummary import summary
from mlp_pytorch import MLP
# import cifar10_utils
from cifar10_utils import *

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None



def accuracy(predictions, targets):
	"""
	Computes the prediction accuracy, i.e. the average of correct predictions
	of the network.

	Args:
	predictions: 2D float array of size [batch_size, n_classes]
	labels: 2D int array of size [batch_size, n_classes]
	        with one-hot encoding. Ground truth labels for
	        each sample in the batch
	Returns:
	accuracy: scalar float, the accuracy of predictions,
	          i.e. the average correct predictions over the whole batch

	TODO:
	Implement accuracy computation.
	"""

	########################
	# PUT YOUR CODE HERE  #
	#######################
	assert  predictions.size(0) == targets.size(0)
	accuracy = torch.mul(predictions,targets).sum()/predictions.size(0)
	# raise NotImplementedError
	########################
	# END OF YOUR CODE    #
	#######################

	return accuracy.item()

def choose_op(prms,op='Adam',lr=2e-3):
	if op == 'Adam':
		return torch.optim.Adam(prms,lr=lr)
	elif op == 'sgd':
		return torch.optim.SGD(prms, lr=lr, momentum=0.9)

# def test():
# 	with torch.no_grad():
# 		model.eval()

def train():
	"""
	Performs training and evaluation of MLP model.

	TODO:
	Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
	"""

	### DO NOT CHANGE SEEDS!
	# Set the random seeds for reproducibility
	np.random.seed(42)

	## Prepare all functions
	# Get number of units in each hidden layer specified in the string such as 100,100
	if FLAGS.dnn_hidden_units:
		dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
		dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
	else:
		dnn_hidden_units = []

	########################
	# PUT YOUR CODE HERE  #
	#######################
	# flags
	steps = FLAGS.max_steps
	batch = FLAGS.batch_size
	freq  = FLAGS.eval_freq
	lr    = FLAGS.learning_rate

	# data
	data_dict = get_cifar10(data_dir = CIFAR10_FOLDER, one_hot = True, validation_size = 10000)
	d_train = data_dict['train']
	d_valid = data_dict['validation']
	d_test  = data_dict['test']

	# model
	n_input = torch.from_numpy(d_train.images[0]).numel()
	n_class = torch.from_numpy(d_train.labels[0]).numel()
	mlp_model = MLP(n_input,dnn_hidden_units,n_class)
	# print(mlp_model)
	# summary(mlp_model,(200,3072))
	# loss & optim
	loss_fn = torch.nn.MSELoss(reduction='sum')
	# loss_fn = torch.nn.CrossEntropyLoss()
	optim = choose_op(mlp_model.parameters(),op='Adam',lr=lr)
	get_d_lenth = lambda x: x.shape[1] * x.shape[2] * x.shape[3]
	data_lenth = get_d_lenth(d_train.images)

	for i in range(steps):
		# print(i)

		x_train,y_train = d_train.next_batch(batch)
		# x_train = x_train.reshape(x_train.shape[0],data_lenth)
		x_train = torch.from_numpy(x_train)
		y_train = torch.from_numpy(y_train).float()

		out = mlp_model(x_train)
		loss = loss_fn(out,y_train)
		print(i, loss.item())
		optim.zero_grad()

		loss.backward()
		optim.step()
		if i ==500:
			acc = accuracy(out, y_train)
			print("keep")

		if (i+1)%freq ==0:
			acc = accuracy(out,y_train)
			print('#'*10)
	print('Finish')








	# raise NotImplementedError
	########################
	# END OF YOUR CODE    #
	#######################

def print_flags():
	"""
	Prints all entries in FLAGS variable.
	"""
	for key, value in vars(FLAGS).items():
		print(key + ' : ' + str(value))

def main():
	"""
	Main function
	"""
	# Print all Flags to confirm parameter settings
	print_flags()

	if not os.path.exists(FLAGS.data_dir):
		os.makedirs(FLAGS.data_dir)

	# Run the training operation
	train()

if __name__ == '__main__':
	# Command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
	                  help='Comma separated list of number of units in each hidden layer')
	parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
	                  help='Learning rate')
	parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
	                  help='Number of steps to run trainer.')
	parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
	                  help='Batch size to run trainer.')
	parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
	                    help='Frequency of evaluation on the test set')
	parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
	                  help='Directory for storing input data')
	FLAGS, unparsed = parser.parse_known_args()

	main()