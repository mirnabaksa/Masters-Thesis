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
        #self.mp1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(filters, 1, 3, padding = 1)
        self.bn2 = nn.BatchNorm1d(1)
        self.mp2 = nn.MaxPool1d(2)
        
        self.t_conv1 = nn.ConvTranspose1d(1, filters, 3, stride=2)
        self.t_conv2 = nn.ConvTranspose1d(filters, in_size, 3, stride=1)'''
        
        #Encoder
        '''self.conv1 = nn.Conv1d(in_size, 4, 3, padding = 1)
        self.mp1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(4, 8, 3, padding = 1)
        self.bn2 = nn.BatchNorm1d(8)
        #self.mp2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(8, 16, in_size, padding = 1)
        self.bn3 = nn.BatchNorm1d(16)
        self.mp3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(16, 32, 3, padding = 1)

        self.conv5 = nn.Conv1d(32, 64, 3, padding = 1)
        self.mp5 = nn.MaxPool1d(2)

        self.conv6 = nn.Conv1d(64, 16, 3, padding = 1)

        self.conv7 = nn.Conv1d(16, 1, 3, padding = 1)
        self.mp7 = nn.MaxPool1d(2)


        # Decoder
        self.t_conv1 = nn.ConvTranspose1d(1, 16, 3, stride = 2)
        self.t_conv2 = nn.ConvTranspose1d(16, 64, 3, stride = 1)
        self.t_conv3 = nn.ConvTranspose1d(64, 32, 1, stride = 2)
        self.t_conv4 = nn.ConvTranspose1d(32, 16, 1, stride = 1)
        self.t_conv5 = nn.ConvTranspose1d(16, 8, 1, stride = 2)
        self.t_conv6 = nn.ConvTranspose1d(8, 4, 1, stride = 1)
        self.t_conv7 = nn.ConvTranspose1d(4, in_size, 1, stride = 2)
        
        self.dense = nn.Linear(81, 75)'''

        self.encoder = nn.Sequential(
            nn.Conv1d(in_size, 8, 3),
            nn.ReLU(),
            #nn.MaxPool1d(2),

            nn.Conv1d(8, 32, 4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            #nn.MaxPool1d(2),

            nn.Conv1d(32, 16, 3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            #nn.MaxPool1d(2),

            nn.Conv1d(16, 64, 4),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5),
            nn.ReLU(),
            #nn.MaxPool1d(2),

            nn.Conv1d(128, 32, 3),
            nn.ReLU(),
            nn.Conv1d(32, 1, 3),
            nn.Tanh(),

        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1, 32, 3),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 128, 3),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 5),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 16, 4),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 32, 3),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 8, 4),
            nn.ReLU(),
            nn.ConvTranspose1d(8, in_size, 3),
            nn.Tanh(),
        )

        
    def forward(self, x):
        encoded = self.get_latent(x)
        reconstructed = self.get_reconstructed(encoded)
        return reconstructed

    def get_latent(self, input):
        batch_size, L, features = input.shape
        input = input.view(batch_size, features, L)
        x = self.encoder(input)
        #print(x.shape)
        return x

    def get_reconstructed(self, x):
        batch_size, _, _ = x.shape
        x = self.decoder(x)
        x = x.view(batch_size, -1, self.in_size)
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
        self.bidirectional = False
        self.output_size = output_size

        self.NN = nn.LSTM(hidden_size * (2 if bidirectional else 1), output_size, batch_first = True, num_layers = num_layers, bidirectional = False)
        nn.init.orthogonal_(self.NN.weight_ih_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.NN.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, input):
        batch_size, steps, _ = input.shape
        self.NN.flatten_parameters()
        output, hidden = self.NN(input)
        #print("out",output.shape)
        
        #output = output.view(batch_size, steps, 2 if self.bidirectional else 1, self.output_size)
        #if self.bidirectional:
        #    output = output[:,:,0,:] + output[:,:,1,:]

        return output

class TripletLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1, dropout = 0.1, bidirectional = False):
        super(TripletLSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.NN = nn.LSTM(input_size, hidden_size, bidirectional = bidirectional, num_layers = num_layers, batch_first = True, dropout = dropout)
        self.pr = True
        #nn.init.orthogonal_(self.NN.weight_ih_l0, gain=np.sqrt(2))
        #nn.init.orthogonal_(self.NN.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, a, p, n):
        if self.pr:
            print(a.shape)
            self.pr = False

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
    def __init__(self, in_size, hidden_size, filters, dropout):
        super(TripletConvolutionalEncoder, self).__init__()

        self.pr = True
        self.hidden_size = hidden_size

        '''self.encoder = nn.Sequential(
            nn.Conv1d(in_size, filters, 3, padding = 1),
            nn.BatchNorm1d(filters),
            nn.MaxPool1d(2),

            nn.Conv1d(filters, filters // 2, 3, padding = 1),
            nn.BatchNorm1d(filters // 2),
            nn.MaxPool1d(2),

            nn.Conv1d(filters // 2, filters // 4, 3, padding = 1),
            nn.BatchNorm1d(filters // 4),

            nn.Conv1d(filters // 4, 1, 3, padding = 1),
            nn.BatchNorm1d(1),
        )

        '''
        self.encoder = nn.Sequential(
            nn.Conv1d(in_size, 16, 3,  padding = 1),
            nn.Tanh(),

            nn.Conv1d(16, 32, 3, stride = 1,  padding = 1),
            nn.Tanh(),

            nn.Conv1d(32, 1, 3, stride = 1, padding = 1),
            nn.Tanh(),
            nn.MaxPool1d(2),

            nn.Linear(25, self.hidden_size)
        )
        
        

        '''self.encoder = nn.Sequential(
            nn.Conv1d(in_size, 16, 3,  padding = 1),
            nn.Tanh(),

            nn.Conv1d(16, 32, 3, padding = 1),
            nn.Tanh(),
            nn.Dropout(0.1),

            nn.Conv1d(32, 48, 3, padding = 1),
            nn.Tanh(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(48, 1, 3, stride = 3, padding = 1),
            nn.Tanh(),
            nn.Dropout(0.5),

            nn.Linear(25, self.hidden_size)
        )'''

  
    def forward(self, a, p, n):
        enc_a = self.get_latent(a)
        enc_p = self.get_latent(p)
        enc_n = self.get_latent(n)
        return enc_a, enc_p, enc_n

    def get_latent(self, input):
        batch_size, L, features = input.shape
        input = input.view(batch_size, features, L)

        y = self.encoder(input)
        y = y.view(batch_size, self.hidden_size, 1)
        
        if self.pr:
            print(y.shape)
            self.pr = False

        return y
        