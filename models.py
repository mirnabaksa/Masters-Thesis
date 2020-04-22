import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

import numpy as np

from torch.nn.utils.rnn import  pad_sequence

# define the NN architecture
class ConvolutionalAutoencoder(nn.Module):
    
    def __init__(self, in_size, filters):
        super().__init__()
        self.in_size = in_size
        self.filters = filters

        '''self.conv1 = nn.Conv1d(in_size, filters, 3, padding = 1)
        self.mp1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(filters, 1, 3, padding = 1)
        self.bn1 = nn.BatchNorm1d(1)
        self.mp2 = nn.MaxPool1d(2)
        
        self.t_conv1 = nn.ConvTranspose1d(1, filters, 3, stride=2)
        self.t_conv2 = nn.ConvTranspose1d(filters, in_size, 3, stride=2)'''
        
        #Encoder
        self.conv1 = nn.Conv1d(in_size, filters, 3, padding = 1)
        self.mp1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(filters, filters//3, 3, padding = 1)
        self.mp2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(filters//3, filters//6, 3, padding = 1)
        self.bn3 = nn.BatchNorm1d(filters//6)
        self.mp3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(filters//6, 1, 3, padding = 1)
        self.bn4 = nn.BatchNorm1d(1)
        self.mp4 = nn.MaxPool1d(2)


        # Decoder
        self.t_conv1 = nn.ConvTranspose1d(1, filters//6, 3, stride = 2)
        self.t_conv2 = nn.ConvTranspose1d(filters//6, filters//3, 3, stride = 2)
        self.t_conv3 = nn.ConvTranspose1d(filters//3, filters, 1, stride = 1)
        self.t_conv4 = nn.ConvTranspose1d(filters, in_size, 1, stride = 1)

        
    def forward(self, x):
        #print("in", x.shape)
        encoded = self.get_latent(x)
        reconstructed = self.get_reconstructed(encoded)
        return reconstructed

    def get_latent(self, input):
        batch_size, L, features = input.shape
        input = input.view(batch_size, features, L)
        #print(input.shape)
        x = self.conv1(input)
        #print(x.shape)
        #x = self.mp1(x)
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        #x = self.bn2(x)
        #print(x.shape)
       # x = self.mp2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.mp3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.mp4(x)
        #print(x.shape)
        return x

    def get_reconstructed(self, x):
        #print("recon in ",x.shape)
        batch_size, _, _ = x.shape
        #x = input.view(batch_size, features, L)
        #print("recon")
        #print(x.shape)
        x = self.t_conv1(x)
        #print(x.shape)
        x = self.t_conv2(x)
        #print(x.shape)
        x = self.t_conv3(x)
        #print(x.shape)
        x = self.t_conv4(x)
        #print(x.shape)
        x = x.view(batch_size, -1, self.in_size)
        #print(x.shape)
        #exit(0)
        return x

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
        #print("inforward",input.shape)
        batch_size, steps, _ = input.shape
        #print("here", steps)
        encoded = self.encoder(input)
        #print("here2")
        #print(encoded.view(batch_size, 1, self.hidden_size).expand(-1, steps, -1).shape)
        encoded = encoded.view(batch_size, 1, self.hidden_size * (2 if self.bidirectional else 1)).expand(-1, steps, -1)
        y = self.decoder(encoded)
        #print("out",y.shape)
        return y


    def get_latent(self, input):
        #print("latent",input.shape)
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
        self.bidirectional = bidirectional
        self.output_size = output_size

        self.NN = nn.LSTM(hidden_size * (2 if bidirectional else 1), output_size, batch_first = True, num_layers = num_layers, bidirectional = bidirectional)

        #nn.init.orthogonal_(self.NN.weight_ih_l0, gain=np.sqrt(2))
        #nn.init.orthogonal_(self.NN.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, input):
        batch_size, steps, _ = input.shape
        self.NN.flatten_parameters()
        output, hidden = self.NN(input)
        output = output.view(batch_size, steps, 2 if self.bidirectional else 1, self.output_size)
        
        if self.bidirectional:
            output = output[:,:,0,:] + output[:,:,1,:]

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
            hidden = torch.cat((hidden[0], hidden[1]), 1)
        return hidden.view(batch_size, -1, self.hidden_size * (2 if self.bidirectional else 1)).squeeze()


class TripletConvolutionalEncoder(nn.Module):
    def __init__(self, in_size, filters, dropout):
        super(TripletConvolutionalEncoder, self).__init__()

        ''' self.conv1 = nn.Conv1d(in_size, filters, 3, padding = 2)
        self.mp1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(filters, filters//2, 3, padding = 2)
        #self.bn1 = nn.BatchNorm1d(filters//2)
        self.mp2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(filters//2, 1, 3, padding = 2)
        self.bn2 = nn.BatchNorm1d(1)
        self.mp3 = nn.MaxPool1d(2)'''

        self.conv1 = nn.Conv1d(in_size, filters, 3, padding = 1)
        #self.mp1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(filters, filters//3, 3, padding = 1)
        #self.mp2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(filters//3, filters//6, 3, padding = 1)
        self.bn3 = nn.BatchNorm1d(filters//6)
        self.mp3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(filters//6, 1, 3, padding = 1)
        self.bn4 = nn.BatchNorm1d(1)
        self.mp4 = nn.MaxPool1d(2)

    def forward(self, a, p, n):
        enc_a = self.get_latent(a)
        enc_p = self.get_latent(p)
        enc_n = self.get_latent(n)
        return enc_a, enc_p, enc_n

    def get_latent(self, input):
        #print(input.shape)
        batch_size, L, features = input.shape
        input = input.view(batch_size, features, L)
        #print(input.shape)
        x = self.conv1(input)
        #print(x.shape)
        #x = self.mp1(x)
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        #x = self.bn1(x)
        #print(x.shape)
        #x = self.mp2(x)
        #print(x.shape)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.mp3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.mp4(x)
        #print(x.shape)
        #exit(0)
        return x