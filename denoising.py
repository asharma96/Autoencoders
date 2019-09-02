# Python 2.7

from load_fashion import load
from load_fashion import display
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def log_likelihood(rand_state, noise, x, y, W, W_p, bias_1, bias_2):
    x_tilde = corrupt(rand_state, x, noise)
    z = reconstruct(x, y, W, W_p, bias_1, bias_2)

    cross_ent = - np.mean(np.sum(x*np.log(z) + (1-x) * np.log(1-z), axis = 1))

    return cross_ent

# sigmoid function 
def sig(x):
    return 1 / (1 + np.exp(- x))

# returns decoded (output)
def decode(y, W_p, bias_1):
    return sig(np.dot(y, W_p) + bias_1)

# returns encoded (hidden)
def encode(x_tilde, W, bias_2):
    return sig(np.dot(x_tilde, W) + bias_2)

# returns reconstructed arbitrary image - encodes and decodes input
def reconstruct(x, y, W, W_p, bias_1, bias_2):
    y = encode(x, W, bias_2)
    z = decode(y, W_p, bias_1)
    return z

# noise image randomly
def corrupt(range, x, noise):
    return range.binomial(1, 1-noise, x.shape) * x

# train denoising autoencoder
def train_ae(x, noise, iterations, learn_rate, num_in_visible, num_in_hidden, rand_state, y, W, W_p, bias_1, bias_2):
	# initialize
	# x = training_data  
	# rand_state = np.random.RandomState(0)
	# W = np.array(rand_state.uniform(-float(1 / num_in_visible), float(1 / num_in_visible), (num_in_visible, num_in_hidden))) # random but also uniform
	# W_p = W.T
	# bias_1, bias_2 = np.zeros(num_in_visible), np.zeros(num_in_hidden)    
	# bias_2 = np.zeros(num_in_hidden)

	# optimize
	costs = []
	for i in range(iterations):
		# noise image
		if i == 0:
			x_tilde = corrupted = corrupt(rand_state, x, noise)
		else:
			x_tilde = corrupt(rand_state, x, noise)
		# print("Training Sample - Corrupted: (See plot):")
		# display(x_tilde, "Training Sample - Corrupted - Iteration " + str(i))

		# parameters
		y = encode(x_tilde, W, bias_2)
		z = decode(y, W_p, bias_1)
		d_bias_1 = x-z
		d_bias_2 = np.dot(d_bias_1, W)*y*(1 - y)
		dW =  np.dot(x_tilde.T, d_bias_2) + np.dot(d_bias_1.T, y)

		# update
		W = W + learn_rate * dW
		bias_1 = bias_1 + learn_rate * np.mean(d_bias_1, axis = 0)
		bias_2 = bias_2 + learn_rate * np.mean(d_bias_2, axis = 0)

		# cost
		cost = log_likelihood(rand_state, noise, x, y, W, W_p, bias_1, bias_2)
		costs.append(cost)
		# print ("Current Iteration Completed: " + str(i) + " Current Cost: " + str(cost))
	return y, W, W_p, bias_1, bias_2, costs

# run denoising autoencoder
def run(train, test, label = None):
	training_data = train
	# initialize parameters	and keep constant while varying noise
	noise = 0.5
	iterations = 100
	learn_rate = 0.1
	num_in_visible = 784
	num_in_hidden = 1176 # overcomplete 

    # initialize nn
	y = 0
	x = training_data  
	rand_state = np.random.RandomState(0)
	W = np.array(rand_state.uniform(-float(1 / num_in_visible), float(1 / num_in_visible), (num_in_visible, num_in_hidden))) # random but also uniform
	W_p = W.T
	bias_1, bias_2 = np.zeros(num_in_visible), np.zeros(num_in_hidden)    

	# training phase over entire training data (can display originals & noised versions)
	for i in range(0, len(training_data) - 99, 100):
		img = training_data[i:i+100]
		training_sample = np.array(img) / 255
		print("Training Samples " + str(i) + " to " + str(i + 99))# + " - Original (See plot):")
		# print(img)
		display(img, "Training Samples " + str(i) + " to " + str(i + 99) + " - Original") # display(img, title, invert = False)

		# update - train on given sample
		y, W, W_p, bias_1, bias_2, cost_values = train_ae(training_sample, noise, iterations, learn_rate, num_in_visible, num_in_hidden, rand_state, y, W, W_p, bias_1, bias_2)

	# testing phase (display original and reconstructed version)
	testing_sample = np.array(test[24:25]) / 255 # example testing sample
	print("Testing Sample - Original (See plot):")
	# print(testing_sample)
	display(testing_sample, "Testing Sample - Original") # display(img, title, invert = False)
	
	corrupted = corrupt(rand_state, testing_sample, noise)
	display(corrupted, "Testing Sample - Corrupted")

	reconstructed = reconstruct(corrupted, y, W, W_p, bias_1, bias_2,)
	print("Testing Sample - Reconstructed (See plot):")
	# print(reconstructed)
	display(reconstructed, "Testing Sample - Reconstructed")

from pca import pca_test
# main method - parse input data, run autoencoder
def main():
	# load Fashion-MNIST data
	X_train, y_train = load('data/fashion', 'train')
	X_test, y_test = load('data/fashion', 't10k')

	# run denoising autoencoder
	print("Denoising on Raw Data")
	run(X_train, X_test) # train_label = None -> unsupervised

	# run pca on raw data (same test sample) to compare with DAE
	print("PCA on Raw Data")
	pca_test(X_test[0:25])	

if __name__ == "__main__":
    main()