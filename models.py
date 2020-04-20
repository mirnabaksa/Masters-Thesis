import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

import numpy as np

from torch.nn.utils.rnn import  pad_sequence

# define the NN architecture
class ConvAutoencoder(nn.Module):
    
    def __init__(self):
        super().__init__()

        kernel_size = 3
        in_channels = 1
        filters = 3
        
        '''self.conv1 = nn.Conv1d(1, 8, 3, padding = 1)
        self.mp1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(8, 1, 5, padding = 2)
        self.bn1 = nn.BatchNorm1d(32)
        self.mp2 = nn.MaxPool1d(2)
        self.t_conv2 = nn.ConvTranspose1d(1, 32, 2, stride = 2)
        self.t_conv3 = nn.ConvTranspose1d(32, 1, 2, stride = 2)
        '''

        self.conv1 = nn.Conv1d(1, 3, 3, padding=1)  
        self.conv2 = nn.Conv1d(3, 1, 3, padding=1)
        self.pool = nn.MaxPool1d(4)
        
        self.t_conv1 = nn.ConvTranspose1d(1, 3, 2, stride=4)
        self.t_conv2 = nn.ConvTranspose1d(3, 1, 2, stride=4)

        
    def forward(self, x):
        #print(x.shape)
        x = self.conv1(x)
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        x = self.pool(x)
        ##print(x.shape)
        #exit(0)
        x = self.t_conv1(x)
        x = self.t_conv2(x)  
        print(x.shape)
        exit(0)
        return x

    def get_latent(self, x):
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = self.pool(x)
        print(x.shape)
        return x.squeeze()

class DenseAutoEncoder(nn.Module):
    def __init__(self):
        super(DenseAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(True),
            nn.Linear(16, 8))

        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(True),
            nn.Linear(16, 2),
            nn.Sigmoid())

    def forward(self, x):
        #print(x)
        #exit(0)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_latent(self, x):
        latent =  self.encoder(x)
        print(latent)
        print(latent[:,-1,:])
        exit(0)
        return latent[:,-1,:]

class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers = 1, dropout = 0.1, bidirectional = False):
        super(LSTMAutoEncoder, self).__init__()
        self.vec_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout, bidirectional)
        self.decoder = Decoder(hidden_size, output_size, num_layers, dropout, bidirectional)

    def forward(self, input):
        batch_size, steps, _ = input.shape
        encoded = self.encoder(input)
        encoded = encoded.view(batch_size, 1, self.hidden_size * (2 if self.bidirectional else 1)).expand(-1, steps, -1)
        y = self.decoder(encoded)
        return y


    def get_latent(self, input):
        latent = self.encoder.get_latent(input)
        return latent


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1, drop = 0.1, bidirectional = False):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.NN = nn.LSTM(input_size, hidden_size, batch_first = True, num_layers = num_layers, bidirectional = bidirectional)

        nn.init.orthogonal_(self.NN.weight_ih_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.NN.weight_hh_l0, gain=np.sqrt(2))
     
    def forward(self, input):
        self.NN.flatten_parameters()
        batch_size, steps, features = input.shape
        output, (hidden, _) = self.NN(input)
        hidden = hidden.view(self.num_layers, 2 if self.bidirectional else 1, batch_size, self.hidden_size)
        hidden = hidden[-1]

        if self.bidirectional:
            hidden = torch.cat((hidden[0],hidden[1]), 1)

        return hidden
        

    def get_latent(self, input):
        batch_size, _, _ = input.shape
        output, (hidden, _) = self.NN(input)
        hidden = hidden.view(self.num_layers, 2 if self.bidirectional else 1, batch_size, self.hidden_size)
        hidden = hidden[-1]
        
        if self.bidirectional:
            hidden = torch.cat((hidden[0],hidden[1]), 1)

        return hidden.squeeze()

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers = 1, drop = 0.1, bidirectional = False):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.NN = nn.LSTM(hidden_size * (2 if bidirectional else 1), output_size, batch_first = True, num_layers = num_layers, bidirectional = False)

        nn.init.orthogonal_(self.NN.weight_ih_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.NN.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, input):
        self.NN.flatten_parameters()
        output, hidden = self.NN(input)
        return output

class TripletLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1, dropout = 0.1, bidirectional = False):
        super(TripletLSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.NN = nn.LSTM(input_size, hidden_size, bidirectional = bidirectional, num_layers = num_layers, batch_first = True, dropout = dropout)

        #nn.init.orthogonal_(self.NN.weight_ih_l0, gain=np.sqrt(2))
        #nn.init.orthogonal_(self.NN.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, a, p, n):
        out_a = self.output(a)
        out_p = self.output(p)
        out_n = self.output(n)
        return out_a, out_p, out_n

    def output(self, input):
        batch_size, _, _ = input.shape
        out, (hidden, _) = self.NN(input)
    
        hidden = hidden.view(self.num_layers, 2 if self.bidirectional else 1, batch_size, self.hidden_size)
        hidden = hidden[-1]

        if self.bidirectional:
            hidden = torch.cat((hidden[0], hidden[1]),1)

        return (hidden.view(batch_size, -1, self.hidden_size * (2 if self.bidirectional else 1)))


    def get_latent(self, input):
        batch_size, _, _ = input.shape
        output, (hidden, _) = self.NN(input)
        hidden = hidden.view(self.num_layers, 2 if self.bidirectional else 1, batch_size, self.hidden_size)
        hidden = hidden[-1]
        if self.bidirectional:
            hidden = torch.cat((hidden[0], hidden[1]),1)
        return hidden.view(batch_size, -1, self.hidden_size * (2 if self.bidirectional else 1)).squeeze()


class TripletConvolutionalEncoder(nn.Module):
    def __init__(self, in_size, dropout):
        super(TripletConvolutionalEncoder, self).__init__()
        filters = 16

        self.conv1 = nn.Conv1d(in_size, filters, 3, padding = 2)
        self.mp1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(filters, 1, 1, padding = 2)
        self.bn1 = nn.BatchNorm1d(1)
        self.mp2 = nn.MaxPool1d(2)

        #self.conv3 = nn.Conv1d(8, 1, 3, padding = 2)
        #self.bn2 = nn.BatchNorm1d(1)
        #self.mp3 = nn.MaxPool1d(2)

    def forward(self, a, p, n):
        enc_a = self.get_latent(a)
        enc_p = self.get_latent(p)
        enc_n = self.get_latent(n)
        return enc_a, enc_p, enc_n

    def get_latent(self, input):
        batch_size, L, features = input.shape
        input = input.view(batch_size, features, L)
        #print(input.shape)
        x = F.relu(self.conv1(input))
        #print(x.shape)
        x = self.mp1(x)
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = self.bn1(x)
        #print(x.shape)
        x = self.mp2(x)
        #print(x.shape)
        #x = F.relu(self.conv3(x))
        #x = self.bn2(x)
        #x = self.mp3(x)
        #print(x.shape)
        #exit(0)
        return x