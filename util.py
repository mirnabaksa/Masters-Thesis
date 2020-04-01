import time
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE

import matplotlib.patheffects as PathEffects
import seaborn as sns
import numpy as np


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
    plt.plot(train)
    if validation:
        plt.plot(validation)

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','validation'], loc='upper left')

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


def visualize(X, y, distinct_labels, name = "tsne.png"):
    X = np.array(X)
    y = np.array(y)
    print("Visualizing...")
    tsne = TSNE()
    train_tsne_embeds = tsne.fit_transform(X)
    scatter(train_tsne_embeds, y, distinct_labels, name, "Data")


def scatter(x, labels, distinct_labels, name, subtitle=None):
    print("Scattering...")
    palette = np.array(sns.color_palette("hls", 10))
    colors = []
    for label in labels:
        colors.append(palette[distinct_labels.index(label)])
  
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')

    sc = ax.scatter(x[:,0], x[:,1], s = 40,  c=colors)
    ax.axis('off')
    ax.axis('tight')
    
    txts = []
    '''
    for label in distinct_labels:
        xtext, ytext = np.median(x[labels == label, :], axis=0)
        txt = ax.text(xtext, ytext, label, fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    '''
        
    if subtitle != None:
        plt.suptitle(subtitle)
        
    plt.savefig("figures/" + name)

