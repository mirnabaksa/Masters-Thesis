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

     
def knn(X, y,  k = 3):
    print("Fitting KNN...")
    neigh = KNeighborsClassifier(n_neighbors = k)
    neigh.fit(X, y)
    return neigh


def visualize(X, y, three_d = False, test = False, subtitle = "Data"):
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


distinct_labels = ["bacillus_anthracis", "ecoli", "pseudomonas_koreensis", "pantonea_agglomerans", "yersinia_pestis", "klebsiella_pneumoniae"]
palette = np.array(sns.hls_palette(6))
colours = ListedColormap(palette)
marker = "o"

from matplotlib.lines import Line2D

def scatter(x, labels, three_d = False, subtitle=None):
    f = plt.figure(figsize=(8, 8), edgecolor='black')
    ax = f.add_subplot(111,projection=('3d' if three_d else None))
    if not three_d:
        ax.axis('off')
    
    classes = [(i,distinct_labels[i]) for i in set(labels)]
    c = [palette[i] for i in labels]

    values = labels
    ax.scatter(x[:,0], x[:,1], s = 40, c = c)
    #for label in labels:
    #    idx = np.where(labels == label)
    #    text_label = distinct_labels[label]
    #    if three_d:
    #        ax.scatter(x[idx,0], x[idx,1], x[idx,2],  marker = marker, linewidth=3,  label = text_label,s = 100,  facecolors='none', edgecolors = [colours.colors[label]])
    #    else:
    #        ax.scatter(x[idx,0], x[idx,1], marker = marker, linewidth=3, label = text_label, s = 100, facecolors='none', edgecolors = [colours.colors[label]])

    custom_lines = []
    cl = []
    for i, label in classes:
        custom_lines.append(Line2D([0], [0], color=palette[i], lw=4))
        cl.append(label)

    ax.legend(custom_lines, cl)

    if not three_d:
        ax.set_yticks([])
        ax.set_xticks([])

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


