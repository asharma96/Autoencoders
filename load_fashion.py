# Python 2.7

import os
import gzip
import numpy as np
from matplotlib import pyplot
import matplotlib as mpl

def load(path, trainortest):
    path_images = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % trainortest)

    path_labels = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % trainortest)

    with gzip.open(path_labels, 'rb') as label_p:
        labels = np.frombuffer(label_p.read(), dtype=np.uint8,
                               offset=8).astype(float)

    with gzip.open(path_images, 'rb') as img_p:
        images = np.frombuffer(img_p.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784).astype(float)

    return images, labels

def display(inp, title, invert=True):
    inp = np.reshape(inp, (-1, 28, 28))
    if invert:
        inp = 255 - inp

    if isinstance(inp, list):
        inp = np.array(inp)
    img_a = inp.shape[1]
    img_b = inp.shape[2]
    n = int(np.ceil(np.sqrt(inp.shape[0])))

    img = np.ones((img_a * n, img_b * n))

    for i in range(n):
        for j in range(n):
            f = i * n + j
            if f < inp.shape[0]:
                im = inp[f]
                img[i * img_a:(i + 1) * img_a,
                j * img_b:(j + 1) * img_b] = im

    f = pyplot.figure()
    ax = f.add_subplot(1,1,1)
    ax.set_title(title)
    imgplot = ax.imshow(img, cmap = mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    pyplot.show() 