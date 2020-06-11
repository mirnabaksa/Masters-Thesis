import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import time

from Model import TripletEncoder, ConvEncoder
from SignalDataset import SignalTripletDataset, TripletTestDataset, TripletStatsDataset, StatsSubsetDataset, StatsDataset

from util import knn, visualize, showPlot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

def collate(input):
    a, p, n, labels = map(list, zip(*input))
    a, len_a = get_input(a)
    p, len_p = get_input(p)
    n, len_n = get_input(n)
    
    return a, p, n, labels

def get_input(in_batch):
    #padded = pad_sequence(in_batch, batch_first = True, padding_value = 0)
    #lens = [len(x) for x in in_batch]
    #out = pack_padded_sequence(padded, lens, batch_first = True, enforce_sorted = False)
    return padded, torch.IntTensor(lens)

def triplet_loss(a, p, n, margin=0.25) : 
    d = nn.PairwiseDistance(p=2)
    distance = d(a, p) - d(a, n) + margin 
    loss = torch.mean(torch.max(distance, torch.zeros_like(distance))) 
    return loss

def train(train_dataset, validation_dataset, vector_size = 1, iterations = 1000, hidden_size = 3, batch_size = 64):
    print("Training...")
    print("Start time (1 hour behind)...", time.strftime("%H:%M:%S", time.localtime()))
    start_time = time.time()
    train = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, pin_memory = True)
    validation = DataLoader(validation_dataset, batch_size = batch_size, shuffle = True, pin_memory = True)

    encoder = ConvEncoder()
    #encoder = TripletEncoder(vector_size, hidden_size, n_layers = 1, dropout = 0.2, bidirectional = True)
    #if torch.cuda.device_count() > 1:
    #    print("Using", torch.cuda.device_count(), "GPUs!")
    #    encoder = nn.DataParallel(encoder)
    
    encoder = encoder.to(device, non_blocking = True)
    optimizer = optim.Adam(encoder.parameters())
    criterion = triplet_loss

    train_losses = []
    validation_losses = []

    for iter in range(iterations):
        loss_acc = 0
        encoder.train()
        
        for in_a, in_p, in_n, l in train: 
            in_a = in_a.to(device, non_blocking = True)
            in_p = in_p.to(device, non_blocking = True)
            in_n = in_n.to(device, non_blocking = True)

            optimizer.zero_grad()
            #print("###")
            
            #print("Before ", in_a.shape, in_p.shape, in_n.shape)
            a, p, n = encoder(in_a, in_p, in_n) #, len_a, len_p, len_n)
            #print("After ", a)
            #print(p)
            #print(n)

            #print(a.shape)
            #print(p.shape)
            #print(n.shape)
            #print("%")
            #print(a.shape)
            #print(p.shape)
            #print(n.shape)
            
            loss = criterion(a, p, n)
            
            #exit(0)
            loss.backward()
            loss_acc += loss
            optimizer.step()

        
        train_losses.append(loss_acc/len(train))

        
        with torch.no_grad():
            val_loss_acc = 0
            encoder.eval()

            for a, p, n, l in validation:
                a = a.to(device, non_blocking = True)
                p = p.to(device, non_blocking = True)
                n = n.to(device, non_blocking = True)
                #len_a = len_a.to(device, non_blocking = True)
                #len_p = len_p.to(device, non_blocking = True)
                #len_n = len_n.to(device, non_blocking = True)
            
                
                a,p,n = encoder(a,p,n) #, len_a, len_p, len_n)

                val_loss = criterion(a,p,n)
                val_loss_acc += val_loss.item()

            validation_losses.append(val_loss_acc/len(validation))

        if iter%1 == 0:
            print("Iteration:", iter, 
            " Train loss: ", "{0:.8f}".format(loss_acc/len(train)), 
            " Validation loss: ", "{0:.8f}".format(val_loss_acc/len(validation))
            )
        
    
        if iter%5 == 0:
            torch.save(encoder, "models/triplet.pt")
            showPlot(train_losses, validation_losses, filename = "figures/triplet-loss.png")
    
    showPlot(train_losses, validation_losses, filename = "figures/datatriplet-loss.png")
    torch.save(encoder, "models/triplet.pt")

def evaluate(test_dataloader):
    model = torch.load("models/triplet.pt")
    return
    for a,p,n,l in test_dataloader:
        #output = model.get_latent(a)
        print("Target")
        print()
        print("Model out")
        #print(output.squeeze())
        print()

def get_latent(dataloader, model):
    print("Collecting latent vector...")
    X = []
    y = []

    for a,p,n, label in dataloader:

        if (isinstance(model, nn.DataParallel)):
            hidden = model.module.get_latent(a.to(device, non_blocking = True))
        else:
            hidden = model.get_latent(a.to(device, non_blocking = True))

        #print(hidden.shape)
        #exit(0)
        vector = hidden.squeeze().tolist()
        if dataloader.batch_size == 1:
            vector = [vector]
        
        X.extend(vector)
        y.extend(label) 
    
    return X, y


def predict(predictor, model, dataloader):
    print("Predicting...")
    correct = 0
    for a, p, n, l in dataloader:
        X = a.to(device, non_blocking = True)
        labels = l
 
        if (isinstance(model, nn.DataParallel)):
            encoder_hidden = model.module.get_latent(a.to(device, non_blocking = True))
        else:
            encoder_hidden = model.get_latent(a.to(device, non_blocking = True))

        X = encoder_hidden.squeeze().tolist()
        if dataloader.batch_size == 1:
            X = [X]
        pred = predictor.predict(X)
        
        for i in range(len(pred)):
            if pred[i] == labels[i]:
                correct += 1
    
    n = len(dataloader) * dataloader.batch_size
    print("Accuracy: ", correct / n)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help = "Training", action = "store_true")
    parser.add_argument("--testdata", help = "Use test data", action = "store_true")
    parser.add_argument("--stats", help = "Use real stats data", action = "store_true")
    args = parser.parse_args()

    if (args.testdata):
        dataset = TripletTestDataset()
        vec_size = 1
        print("Using test data")
    if (args.stats):
        dataset = StatsDataset("csv/perfect-stats4class.csv")
    #else:
    #    dataset = SignalTripletDataset("../Signals/full_dataset/", "csv/dataset_triplet-2class.csv", raw = True)
    #    print("Using real data")
   
    train_size = int(0.7 * len(dataset)) #int(0.8 * len(dataset))
    val_test_size = (len(dataset) - train_size) // 2
    if((train_size + 2 * val_test_size) != len(dataset)):
        train_size += 1

    print(len(dataset), train_size, val_test_size)
    print("Dataset length: ", str(train_size) + " train, " + str(val_test_size) + " test!")
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,  val_test_size, val_test_size]) 

    if (args.stats):
        print("stats")
        train_dataset = StatsSubsetDataset(train_dataset, wrapped = True, minmax = False)
        vec_size = train_dataset.get_vector_len()
        train_dataset = TripletStatsDataset(train_dataset)

        validation_dataset = TripletStatsDataset(StatsSubsetDataset(validation_dataset, wrapped = True, minmax = False))
        test_dataset = TripletStatsDataset(StatsSubsetDataset(test_dataset, wrapped = True, minmax = False))

    if (args.train):
        train(train_dataset, validation_dataset, vector_size = vec_size)

    #showPlotFromFile("data.txt")

    print("Starting evaluation phase")
    dataloader = DataLoader(train_dataset, batch_size = 1, shuffle = True, pin_memory = True)
    model = torch.load("models/triplet.pt")

    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = True, pin_memory = True)
    train_dataloader = DataLoader(train_dataset, batch_size = 1, shuffle = True, pin_memory = True)
    #print("Evaluating train")
    #evaluate(train_dataloader)
    #print("Evaluating test")
    #evaluate(test_dataloader)
   
    X, y = get_latent(dataloader, model)
    #for point in zip(X, y):
    #    print(point)
    print(X[0])

    test_X, test_y = get_latent(test_dataloader, model)

    predictor = knn(X, y, 3)
    predict(predictor, model, test_dataloader)
    
    visualize(test_X, test_y, train_dataset.get_distinct_labels(), "test-tsne-triplet.png")
    visualize(X, y, train_dataset.get_distinct_labels(), "train-tsne-triplet.png")