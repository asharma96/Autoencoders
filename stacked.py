import numpy
import sys
from load_fashion import load
from load_fashion import display

X_train, y_train = load('data/fashion', 'train')
X_test, y_test = load('data/fashion', 't10k')
display(X_train[5]) # display uninverted image with display(x, False)