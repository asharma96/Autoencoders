from load_fashion import load
from load_fashion import display
import numpy as np
from numpy.linalg import eig

# scratch implementation
def pca(data):
	mean = np.mean(data.T, axis=1)

	X = data - mean
	cov = np.cov(X.T)
	eigenvals, eigenvecs = eig(cov)
	lower_dimension_data = eigenvecs.T.dot(cov.T)
	
	return lower_dimension_data.T

from sklearn.decomposition import PCA
def pca_test(testing):
	# testing phase (display original and reconstructed version)
	testing_sample = np.array(np.array(testing)) # example testing sample
	print("PCA Testing Sample - Original (See plot):")
	# print(testing_sample)
	display(testing_sample[-1], "PCA Testing Sample - Original") # display(img, title, invert = False)

	pca = PCA(0.95) # give PCA good performance to prove its nonlinearity weakness regardless

	lower_dimensional_data = pca.fit_transform(testing_sample)

	approximation = pca.inverse_transform(lower_dimensional_data)

	reconstructed = approximation[-1].reshape(28, 28)
	print("PCA Testing Sample - Reconstructed (See plot):")
	# print(reconstructed)
	display(reconstructed, "PCA Testing Sample - Reconstructed")

	return lower_dimensional_data

def main():
	# load Fashion-MNIST test data
	X_train, y_train = load('data/fashion', 'train')

	pca_test(X_train)

if __name__ == '__main__':
	main()