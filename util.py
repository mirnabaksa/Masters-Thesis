import time
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE

import matplotlib.patheffects as PathEffects
import seaborn as sns
import numpy as np
import io

import tensorflow as tf
from pytorch_lightning import _logger as log

def timeNow():
    return time.time()

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlotFromFile(filename = "data.txt"):
    train = []
    validation = []
    with open(filename, "r") as f:
        for line in f:
            t, v = line.split(",")
            train.append(float(t))
            validation.append(float(v))
            
    showPlot(train[10:], validation[10:], filename = "figures/loss-clipped.png")


def showPlot(train, validation = None, filename = "figures/loss.png"):

    if validation:
        plt.plot(validation)

    plt.plot(train)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['validation','train'], loc='upper left')

    plt.savefig(filename)
    plt.close()

def showDataPlot(X, filename = "data", title = None):
    plt.figure(figsize=(12,6))
    plt.plot(X)

    if (title):
        plt.title(title)
    plt.ylabel('data')
    plt.xlabel('time')

    plt.savefig('figures/' + filename + ".png")
    plt.close()

     
def knn(X, y,  k):
    print("Fitting KNN...")
    neigh = KNeighborsClassifier(n_neighbors = k, weights = "distance")
    neigh.fit(X, y)
    return neigh


def visualize(X, y, three_d = False, test = False, subtitle = "Data"):
    print("Visualising...")
    X = np.array(X)
    y = np.array(y)
    tsne = TSNE(3 if three_d else 2, random_state = 2334)
    train_tsne_embeds = tsne.fit_transform(X)

    #if test:
    #    return scatter_test(train_tsne_embeds, y, three_d, subtitle)

    return scatter(train_tsne_embeds, y, three_d, subtitle)


import torchvision
import PIL.Image
from torchvision.transforms import ToTensor
from mpl_toolkits.mplot3d import Axes3D


distinct_labels = ["Bacillus anthracis", "Ecoli", "Yersinia pestis", "Pseudomonas koreensis", "Pantonea agglomerans", "Klebsiella pneumoniae"]
#distinct_labels = ["Enterococcus faecalis", "Staphylococcus aureus", "Listeria monocytogenes", "Lactobacillus fermentum", "Bacillus subtilis", "Escherichia coli"]
#distinct_labels = [0,1,2,3]
palette = np.array(sns.hls_palette(len(distinct_labels)))
colours = ListedColormap(palette)
marker = "o"

from matplotlib.lines import Line2D

def scatter(x, labels, three_d = False, subtitle=None):
    f = plt.figure( edgecolor='black')
    ax = f.add_subplot(111,projection=('3d' if three_d else None))
    ax.axis('off')
    plt.tight_layout()
    
    classes = [distinct_labels[i] for i in set(labels)]
    c = [palette[i] for i in labels]

    values = labels
    if not three_d:
        ax.scatter(x[:,0], x[:,1], marker = "o", linewidth=1, facecolors=c, edgecolors=(0,0,0,1), s = 50, alpha = 0.3)
    else:
        ax.scatter(x[:,0], x[:,1], x[:,2],  marker = "o", linewidth=1, facecolors=c, edgecolors=(0,0,0,1), s = 160, alpha = 0.3)
    

    '''custom_lines = []
    cl = []
    for i, label in enumerate(classes):
        custom_lines.append(Line2D([0], [0], color=palette[i], lw=4))
        cl.append(label)

    ax.legend(custom_lines, cl, prop={'size': 20}, loc = "upper right")'''

    buf = io.BytesIO()
    plt.savefig(buf, format = 'png')
    buf.seek(0)

    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    return image

from collections import OrderedDict
def scatter_test(x, labels, three_d = False, subtitle=None):
    train_size = int(0.7 * len(x))
    test_size = len(x) - train_size

    train_x, test_x = x[:train_size], x[train_size:]
    train_labels, test_labels = labels[:train_size], labels[train_size:]

    set_labels = set(labels)
    classes = [distinct_labels[i] for i in set_labels]

    test_palette = np.array(sns.hls_palette(6, l=.3, s=.8))
    test_colours = ListedColormap(test_palette)

    f = plt.figure(figsize=(8, 8), edgecolor='black')
    ax = f.add_subplot(111, projection=('3d' if three_d else None))
    if not three_d:
        ax.axis('off')

    train_x, train_y = train_x[:,0], train_x[:,1]
    test_x, test_y, = test_x[:,0], test_x[:,1]

    train_z = train_x[:,2] if three_d else 0
    test_z = test_x[:,2] if three_d else 0

    for label in set_labels:
        idx_train = np.where(train_labels == label)
        idx_test = np.where(test_labels == label)

        x, y = train_x[idx_train], train_y[idx_train]
        t_x, t_y  = test_x[idx_test], test_y[idx_test]
        z = train_z[idx_train] if three_d else 0
        t_z = test_z[idx_test] if three_d else 0

        text_label = distinct_labels[label]
        if three_d:
            ax.scatter(x, y, z, marker = marker, linewidth=3, label = text_label, facecolors='none', s = 100,  edgecolors = [colours.colors[label]])
            ax.scatter(t_x, t_y, t_z, marker = marker, linewidth=3, label = text_label + " test", facecolors='none',s = 100,  edgecolors = [test_colours.colors[label]])
        else:
            ax.scatter(x, y,  marker = marker, linewidth=3, label = text_label, facecolors='none',s = 100,  edgecolors = [colours.colors[label]])
            ax.scatter(t_x, t_y,  marker = marker, linewidth=3, label = text_label + " test", facecolors='none', s = 100, edgecolors = [test_colours.colors[label]])

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), prop={'size': 20})

    if not three_d:
        ax.set_yticks([])
        ax.set_xticks([])
    
    buf = io.BytesIO()
    plt.savefig(buf, format = 'png')
    buf.seek(0)

    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    return image


def plotOutput(in_vec, out_vec, target_vec):
    means_in = []
    means_out = []
    means_target = []
    #means_in = in_vec
    #means_out = out_vec
    #means_target = target_vec


    stdev_in = []
    stdev_out = []
    for i in range(len(in_vec)):
        means_in.append(in_vec[i][0])
        means_out.append(out_vec[i][0])
        means_target.append(target_vec[i][0])
        #stdev_in.append(in_vec[i][1])
        #stdev_out.append(out_vec[i][1])

    x = np.arange(0, len(means_in))
    plt.figure(figsize=(15,4))
    plt.plot(x, means_in,  "b")
    plt.plot(x, means_out, "r")
    plt.plot(x, means_target, "g")
    plt.legend(["in", "out", "target"])
    plt.title("means")

    buf = io.BytesIO()
    plt.savefig(buf, format = 'png')
    buf.seek(0)

    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    return image


