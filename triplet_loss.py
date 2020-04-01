import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import time

from Model import TripletEncoder
from SignalDataset import SignalTripletDataset, TripletTestDataset

from util import knn, visualize, constructTripletDatasetCSV, showPlot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)
torch.cuda.manual_seed(1)

def collate(input):
    a, p, n, labels = map(list, zip(*input))
    a, len_a = get_input(a)
    p, len_p = get_input(p)
    n, len_n = get_input(n)
    
    return a, p, n, (len_a, len_p, len_n), labels

def get_input(in_batch):
    padded = pad_sequence(in_batch, batch_first = True, padding_value = 0)
    lens = [len(x) for x in in_batch]
    #out = pack_padded_sequence(padded, lens, batch_first = True, enforce_sorted = False)
    return padded, torch.IntTensor(lens)

def triplet_loss(a, p, n, margin=0.2) : 
    d = nn.PairwiseDistance(p=2)
    distance = d(a, p) - d(a, n) + margin 
    loss = torch.mean(torch.max(distance, torch.zeros_like(distance))) 
    return loss


def train(train_dataset, validation_dataset, iterations = 300, hidden_size = 127, batch_size = 4):
    print("Training...")
    print("Start time (1 hour behind)...", time.strftime("%H:%M:%S", time.localtime()))
    start_time = time.time()
    train = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, pin_memory = True, collate_fn = collate)
    validation = DataLoader(validation_dataset, batch_size = batch_size, shuffle = True, pin_memory = True, collate_fn = collate)

    encoder = TripletEncoder(1, hidden_size)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        encoder = nn.DataParallel(encoder)
    
    encoder = encoder.to(device, non_blocking = True)
    optimizer = optim.Adam(encoder.parameters())
    criterion = triplet_loss

    train_losses = []
    validation_losses = []

    for iter in range(iterations):
        loss_acc = 0
        enc_last_hidden = None
        encoder.train()
        
        for in_a, in_p, in_n, lens, l in train:  
            a, p, n = encoder(in_a.to(device, non_blocking = True), 
                in_p.to(device, non_blocking = True), 
                in_n.to(device, non_blocking = True), lens)
            enc_last_hidden = a
            
            optimizer.zero_grad()
            loss = criterion(a, p, n)
            loss_acc += loss.item()
            loss.backward(retain_graph = True)
            optimizer.step()
        
        train_losses.append(loss_acc/len(train))

        
        with torch.no_grad():
            val_loss_acc = 0
            for a, p, n, lens, l in validation:
            
                encoder.eval()
                a,p,n = encoder(a.to(device, non_blocking = True), 
                p.to(device, non_blocking = True), 
                n.to(device, non_blocking = True), lens)

                val_loss = criterion(a,p,n)
                val_loss_acc += val_loss.item()
            validation_losses.append(val_loss_acc/len(validation))

        if iter%1 == 0:
            print("Iteration:", iter, 
            " Train loss: ", "{0:.5f}".format(loss_acc/len(train)), 
            " Validation loss: ", "{0:.5f}".format(validation_losses[-1])
            )
        loss_acc = 0
        
    
    if iter%50 == 0:
            torch.save(envoder, "models/triplet.pt")
            showPlot(train_losses, validation_losses)
    
    showPlot(train_losses, validation_losses)
    torch.save(encoder, "models/triplet.pt")

def evaluate(test_dataloader):
    model = torch.load("models/triplet_encoder.pt")
    for a,p,n,l in test_dataloader:
        output = model.get_latent(a)
        print("Target")
        print()
        print("Model out")
        print(output.squeeze())
        print()

def get_latent(dataloader, model):
    print("Collecting latent vector...")
    X = []
    y = []


    for a,p,n,lens,label in dataloader:

        if (isinstance(model, nn.DataParallel)):
            hidden = model.module.get_latent(a.to(device, non_blocking = True))
        else:
            hidden = model.get_latent(a.to(device, non_blocking = True))

        vector = hidden.squeeze().tolist()
        if dataloader.batch_size == 1:
            vector = [vector]
        
        X.extend(vector)
        y.extend(label) 
    
    return X, y


def predict(predictor, model, dataloader):
    print("Predicting...")
    correct = 0
    for a, p, n, lens, l in dataloader:
        X = a.to(device, non_blocking = True)
        labels = l
        print(X.squeeze()[:15])
 
        if (isinstance(model, nn.DataParallel)):
            encoder_hidden = model.module.get_latent(a.to(device, non_blocking = True))
        else:
            encoder_hidden = model.get_latent(a.to(device, non_blocking = True))

        X = encoder_hidden.squeeze().tolist()
        print(encoder_hidden.squeeze()[:15])
        if dataloader.batch_size == 1:
            X = [X]
        pred = predictor.predict(X)
        #print("predicted", pred)
        #print("label", labels)
        
        for i in range(len(pred)):
            if pred[i] == labels[i]:
                correct += 1
        #scorrect += sum(pred == labels.item())
    
    n = len(dataloader) * dataloader.batch_size
    print(correct)
    print("Accuracy: ", correct / n)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help = "Run training", action = "store_true")
    parser.add_argument("--testdata", help = "Re-generate dataset csv", action = "store_true")
    args = parser.parse_args()

    if(args.testdata):
        dataset = TripletTestDataset()
        print("Using test data")
    else:
        dataset = SignalTripletDataset("../Signals/full_dataset/", "csv/dataset_triplet-2class.csv", raw = True)
        print("Using real data")
   
    train_size = int(0.8 * len(dataset))
    val_test_size = (len(dataset) - train_size) // 2
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,  val_test_size, val_test_size]) 

    if (args.train):
        train(train_dataset, validation_dataset)

    print("Starting evaluation phase")
    dataloader = DataLoader(train_dataset, batch_size = 1, shuffle = True, collate_fn = collate)
    model = torch.load("models/triplet.pt")

    X, y = get_latent(dataloader, model)
    predictor = knn(X, y, 3)

    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = True, collate_fn = collate)
    #evaluate(test_dataloader)
    predict(predictor, model, dataloader)

    visualize(X, y, dataset.get_distinct_labels())