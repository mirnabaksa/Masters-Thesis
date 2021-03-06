import os
import time
from datetime import timedelta
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'

import torch
import torch.nn as nn
from torch import optim
from torch.nn import MSELoss
from torch.nn.utils.rnn import  pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

import numpy as np

from util import asMinutes, timeSince, showPlot, timeNow, knn, visualize, showPlotFromFile
from models import LSTMAutoEncoder, DenseAutoEncoder, ConvAutoencoder
from SignalDataset import Signal, SignalDataset, TestDataset, StatsTestDataset, StatsDataset, StatsSubsetDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)

def collate(input):
    in_batch, labels = map(list, zip(*input))

    #reversed = []
    #for point in target:
    #    point.reverse()
    #    reversed.append(torch.FloatTensor(point))

    #reversed = pad_sequence(reversed, batch_first = True, padding_value = 0)
    batch_size = len(in_batch)
    padded = pad_sequence(in_batch, batch_first = True, padding_value = 0)
    lens = [len(x) for x in in_batch]
    max_len = max(lens)

    #print(batch_size)
    padded = padded.view(batch_size, 1, -1)
    #print(padded.shape)
    #print(padded)
    return padded, padded, labels, max_len, torch.IntTensor(lens)
  

def train(train_dataset, validation_dataset, vector_size = 1, iterations = 15000, hidden_size = 40, batch_size = 32):
    print("Training...")
    print("Start time (2 hours behind)...", time.strftime("%H:%M:%S", time.localtime()))
    start_time = time.time()
    train = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, pin_memory = True)
    if(validation_dataset):
        validation = DataLoader(validation_dataset, batch_size = batch_size, shuffle = True, pin_memory = True)
    f =  open('data.txt', 'w')

    #model = ConvAutoencoder()
    #print(model)
    model = LSTMAutoEncoder(vector_size, hidden_size, vector_size, num_layers = 2, dropout = 0.2, bidirectional = True)
    #if torch.cuda.device_count() > 1:
    #    print("Using", torch.cuda.device_count(), "GPUs!")
    #    model = nn.DataParallel(model)
    
    model = model.to(device, non_blocking = True)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.L1Loss()

    train_losses = []
    validation_losses = []

    for iter in range(iterations):
        loss_acc = 0
        model.train()
        
        for input_tensor, _ in train:
            target_tensor = input_tensor
            input_tensor = input_tensor.to(device, non_blocking = True)
            target_tensor = target_tensor.to(device, non_blocking = True)
            optimizer.zero_grad()

            outputs = model(input_tensor)
            batch_loss = criterion(outputs.squeeze(), target_tensor.squeeze())
            batch_loss.backward()
            loss_acc += batch_loss.item()
            optimizer.step()
        
        train_losses.append(loss_acc/len(train))

        if(validation_dataset):
            with torch.no_grad():
                val_loss_acc = 0
                model.eval()

                for input_tensor, _ in validation:
                    target_tensor = input_tensor

                    input_tensor = input_tensor.to(device, non_blocking = True)
                    target_tensor = target_tensor.to(device, non_blocking = True)

                    outputs = model(input_tensor)

                    val_loss = criterion(outputs.squeeze(), target_tensor.squeeze())
                    val_loss_acc += val_loss.item()
        
                validation_losses.append(val_loss_acc/len(validation)) 

        f.write(str(loss_acc/len(train)) + ("," + str(val_loss_acc/len(validation)) + "\n" if (validation_dataset) else "\n"))
    
        if iter%1 == 0:
            print("Iteration:", iter, 
            " Train loss: ", "{0:.5f}".format(loss_acc/len(train)), 
            " Validation loss: ", "{0:.5f}".format(validation_losses[-1]) if validation_dataset else ""
            )

        if iter%5 == 0:
            torch.save(model, "models/autoencoder.pt")
            showPlot(train_losses, validation_losses, filename = "figures/autoencoder-loss.png")


        
    f.close()
    showPlot(train_losses, validation_losses, filename = "figures/autoencoder-loss.png")
    torch.save(model, "models/autoencoder.pt")

    print("End time (1 hour behind)...", time.strftime("%H:%M:%S", time.localtime()))

    end_time = time.time()
    hours, rem = divmod(end_time-start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time elapsed_ {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)) 

def evaluate(test_dataloader, p = True):
    model = torch.load("models/autoencoder.pt")

    with torch.no_grad():
        for input_tensor, _ in test_dataloader:
                target_tensor = input_tensor
                input_tensor = input_tensor.to(device, non_blocking = True)
                target_tensor = target_tensor.to(device, non_blocking = True)
            
                outputs = model(input_tensor)
                if p:
                    #print("Latent: ", model.module.get_latent(input_tensor))
                    print("Input: " , input_tensor.squeeze())
                    print("Output: ", outputs.squeeze())
                    print()

        

def get_latent(dataloader, model):
    print("Collecting latent vector...")
    X = []
    y = []
    i = 0
    for input_tensor,  label in dataloader:
        input_tensor = input_tensor.to(device, non_blocking = True)
        #print(input_tensor)
        if (isinstance(model, nn.DataParallel)):
            latent = model.module.get_latent(input_tensor).tolist()
        else:
            latent = model.get_latent(input_tensor).tolist()

        if dataloader.batch_size == 1:
            latent = [latent]

        #print(latent)
        #print("########")
        X.extend(latent)
        y.extend(label) 

    print("Latent X length: ", len(X))
    print("Latent Y length: ", len(y))
    return X, y


def predict(predictor, model, dataloader):
    print("Predicting...")
    correct = 0
    all = 0
    for input_tensor, labels in dataloader:
        input_tensor = input_tensor.to(device, non_blocking = True)

        if (isinstance(model, nn.DataParallel)):
            latent = model.module.get_latent(input_tensor).tolist()
        else:
            latent = model.get_latent(input_tensor).tolist()

        if dataloader.batch_size == 1:
            latent = [latent]
        pred = predictor.predict(latent)
        correct += sum(pred == labels)
        all += len(pred)

    print("Accuracy: ", correct / all)

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help = "Run training", action = "store_true")
    parser.add_argument("--testdata", help = "Use test data", action = "store_true")
    parser.add_argument("--teststats", help = "Use test stats data", action = "store_true")
    parser.add_argument("--stats", help = "Use stats data", action = "store_true")
    args = parser.parse_args()
    
    vec_size = 1

    if (args.testdata):
        dataset = TestDataset() 
        print("Using test data")
    elif (args.teststats):
        dataset = StatsTestDataset()
        vec_size = dataset.get_vector_len()
    elif (args.stats):
        dataset = StatsDataset("csv/perfect-stats4class.csv")
    else:
        dataset = SignalDataset("../Signals/full_dataset/", "csv/dataset-twoclasses.csv", raw = True)
        print("Using real data")
    
    print(len(dataset))
    train_size = int(0.6 * len(dataset)) + (0 if (len(dataset)%2 == 0) else 1)
    val_test_size = (len(dataset) - train_size) // 2 
    print(train_size, val_test_size)
    print(train_size, val_test_size, train_size + val_test_size * 2, len(dataset))
    print("Dataset length: ", str(train_size) + " train, " + str(val_test_size) + " test!")
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,  val_test_size, val_test_size])  

    #train_dataset = dataset 
    #validation_dataset = None
    #test_dataset = dataset

    if(args.stats):
        train_dataset = StatsSubsetDataset(train_dataset, seq2seq = False, minmax = False)
        validation_dataset = StatsSubsetDataset(validation_dataset, seq2seq = False, minmax = False)
        test_dataset = StatsSubsetDataset(test_dataset, seq2seq = False, minmax = False)
        vec_size = train_dataset.get_vector_len()

    if (args.train):
        train(train_dataset, validation_dataset, vector_size = vec_size)

    #showPlotFromFile("data.txt")

    print("Starting evaluation phase")
    dataloader = DataLoader(train_dataset, batch_size = 1, shuffle = True, pin_memory = True)
    model = torch.load("models/autoencoder.pt")

    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = True, pin_memory = True)
    train_dataloader = DataLoader(train_dataset, batch_size = 1, shuffle = True, pin_memory = True)
    print("Evaluating train")
    evaluate(train_dataloader, True)
    print("Evaluating test")
    #evaluate(test_dataloader, True)
   

    X, y = get_latent(dataloader, model)
    #print(X)
    test_X, test_y = get_latent(test_dataloader, model)
    #print(X)
    #predictor = knn(X, y, 3)
    #predict(predictor, model, test_dataloader)
    
    visualize(test_X, test_y, dataset.get_distinct_labels(), "test-tsne-auto.png")
    visualize(X, y, dataset.get_distinct_labels(), "train-tsne-auto.png")
    
    