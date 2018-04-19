from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

# Enabling eager execution. Once enabled, it can not be disabled within the same program.
tf.enable_eager_execution()

# Downloading the Iris classificaiton problem dataset from TensorFlow
train_dataset_url = 'http://download.tensorflow.org/data/iris_training.csv'
train_dataset_fp = tf.keras.utils.get_file(fname = os.path.basename(train_dataset_url), origin = train_dataset_url)

# Inspect the data
print(train_dataset_fp) # viewing the first five rows of data

# Parse dataset
def parse_csv(line):
	example_defaults = [[0.], [0.], [0.], [0.], [0]] # Sets field types
	parsed_line = tf.decode_csv(line, example_defaults)
	# First 4 fields are features, combine them into a tensor
	features = tf.reshape(parsed_line[:-1], shape = (4,))
	# Last field is label
	label = tf.reshape(parsed_line[-1:], shape = ())
	return(features, label)

# Create the dataset
train_dataset = tf.data.TextLineDataset(train_dataset_fp)
train_dataset = train_dataset.skip(1) # skip the first header row
train_dataset = train_dataset.map(parse_csv) # parse each row
train_dataset = train_dataset.shuffle(buffer_size=1000) # randomize
train_dataset = train_dataset.batch(32)

# Viewing a single example entry from batch
features, label = tfe.Iterator(train_dataset).next()
print('example features: ', features[0])
print('example label: ', label[0])

# Creating a model using Keras. Here we are going to create a dense neural network with two hidden layers.
model = tf.keras.Sequential([
	tf.keras.layers.Dense(10, activation = 'relu', input_shape = (4,)),
	tf.keras.layers.Dense(10, activation = 'relu'),
	tf.keras.layers.Dense(3)
	])

# Defining the loss and gradient functions
def loss(model, x, y):
	y_ = model(x)
	return(tf.losses.sparse_softmax_cross_entropy(labels = y, logits = y_))

def grad(model, inputs, targets):
	with tfe.GradientTape() as tape:
		loss_value = loss(model, inputs, targets)
	return(tape.gradient(loss_value, model.variables))

# Creating an optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)

# Training loop

# keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
	epoch_loss_avg = tfe.metrics.Mean()
	epoch_accuracy = tfe.metrics.Accuracy()

	# training loop with batches of 32
	for x, y in tfe.Iterator(train_dataset):
		# optimize the model
		grads = grad(model, x, y)
		optimizer.apply_gradients(zip(grads, model.variables),
								global_step = tf.train.get_or_create_global_step())

		# track progress
		epoch_loss_avg(loss(model, x, y)) # add current batch  loss
		# compare predicted label to actual label
		epoch_accuracy(tf.argmax(model(x), axis = 1, output_type = tf.int32), y)

	train_loss_results.append(epoch_loss_avg.result())
	train_accuracy_results.append(epoch_accuracy.result())

	if epoch % 50 == 0:
		print('Epoch {:03d}: loss: {:.3f}, Accuracy: {:.3%}'.format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))

# Visualize the loss and accuracy functions over time
fig, axes = plt.subplots(2, sharex = True, figsize = (12, 8))
fig.suptitle = ('Training Metrics')

axes[0].set_ylabel('Loss', fontsize = 14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel('Accuracy', fontsize = 14)
axes[1].set_xlabel('Epoch', fontsize = 14)
axes[1].plot(train_accuracy_results)

plt.show()