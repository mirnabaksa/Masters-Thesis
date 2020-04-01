import torch
import torch.nn as nn
from torch import optim
from torch.nn import MSELoss
from torch.nn.utils.rnn import  pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from util import asMinutes, timeSince, showPlot, timeNow, constructDatasetCSV, knn, visualize
from Model import AutoEncoder
from SignalDataset import Signal, SignalDataset, TestDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)
torch.cuda.manual_seed(1)

def collate(input):
    in_batch, labels = map(list, zip(*input))
   
    padded = pad_sequence(in_batch, batch_first = True, padding_value = 0)
    lens = [len(x) for x in in_batch]
    out = pack_padded_sequence(padded, lens, batch_first = True, enforce_sorted = False)
    return out, padded.squeeze(), labels
  

def train(train_dataset, validation_dataset = None, iterations = 60, hidden_size = 5, batch_size = 16):
    print("Training...")
    train = DataLoader(train_dataset, batch_size = batch_size, collate_fn = collate, shuffle = True)
    validation = DataLoader(validation_dataset, batch_size = batch_size, collate_fn = collate, shuffle = True)

    model = AutoEncoder(1, hidden_size).to(device)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    train_losses = []
    validation_losses = []

    for iter in range(iterations):
        loss_acc = 0
        enc_last_hidden = None
        dec_last_hidden = None
        
        for input, target, label in train:  
            optimizer.zero_grad()
            output = model(input)
            
            loss = criterion(output, target)
            loss_acc += loss.item()
      
            loss.backward(retain_graph = True)
            optimizer.step()
        
        train_losses.append(loss_acc/len(train))

        
        with torch.no_grad():
            val_loss_acc = 0
            for input, target, _ in validation:
                model.eval()

                output = model(input)

                val_loss = criterion(output, target)
                val_loss_acc += val_loss.item()
            validation_losses.append(val_loss_acc)
        

        if iter%1 == 0:
            print("Iteration:", iter, 
            " Train loss: ", "{0:.5f}".format(loss_acc/len(train)), 
            " Validation loss: ", "{0:.5f}".format(validation_losses[-1])
            )
        loss_acc = 0
        
    showPlot(train_losses, validation_losses)

    torch.save(model, "models/model.pt")

def evaluate(test_dataloader):
    model = torch.load("models/model.pt")
    for input, target, label in test_dataloader:
        output = model(input)
        print("Target")
        print(target)
        print("Model out")
        print(output)
        print()

def get_latent(dataloader, model):
    print("Collecting latent vector...")
    X = []
    y = []
    for input, _, label in dataloader:
        hidden = model.get_latent(input)
        vector = hidden.squeeze().tolist()
        if dataloader.batch_size == 1:
            vector = [vector]
        
        X.extend(vector)
        y.extend(label) 
    
    return X, y


def predict(predictor, model, dataloader):
    print("Predicting...")
    correct = 0
    for X_test in dataloader:
        X = X_test[0]
        target = X_test[1]
        labels = X_test[2]
 
        encoder_hidden = model.get_latent(X)

        X = encoder_hidden.squeeze().tolist()
        if dataloader.batch_size == 1:
            X = [X]
        pred = predictor.predict(X)
        correct += sum(pred == labels)
    
    n = len(dataloader) * dataloader.batch_size
    print("Accuracy: ", correct / n)


if __name__ == '__main__':
    #constructDatasetCSV("../Signals/full_dataset/")
    #dataset = SignalDataset("../Signals/full_dataset/", "csv/dataset.csv", raw = False)
    dataset = TestDataset()
    train_size = int(0.8 * len(dataset))
    val_test_size = (len(dataset) - train_size) // 2
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,  val_test_size, val_test_size])  
 
    train(train_dataset, validation_dataset)
    
    dataloader = DataLoader(train_dataset, batch_size = 1, shuffle = True, collate_fn = collate)
    model = torch.load("models/model.pt")
    X, y = get_latent(dataloader, model)
    predictor = knn(X, y, 3)
    visualize(X,y, dataset.get_distinct_labels())

    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = True, collate_fn = collate)
    evaluate(test_dataloader)
    predict(predictor, model, test_dataloader)
    
    
    