import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

import numpy as np

from torch.nn.utils.rnn import  pad_sequence

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VariationalAutoencoder, self).__init__()
        self.in_size = input_size
        self.hidden_size = hidden_size
        intermediate = 10
        
        # encoder
        self.enc_linear = nn.Linear(input_size, intermediate)
        self.mu = nn.Linear(intermediate, hidden_size)
        self.var = nn.Linear(intermediate, hidden_size)

        #decoder
        self.dec_linear = nn.Linear(hidden_size, intermediate)
        self.out = nn.Linear(intermediate, input_size)

        self.print = True


    def forward(self, input):
        batch_size, steps, _ = input.shape
        if self.print:
            print(input.shape)
            self.print = False

        #encoder
        hidden = F.relu(self.enc_linear(input))
        print(hidden.shape)
        z_mu = self.mu(hidden)
        z_var = self.var(hidden)
        print(z_mu.shape)


        #sampling
        sample = self.sampling(z_mean, z_log_sigma)
        
        hidden = F.relu(self.dec_linear(x))
        predicted = torch.sigmoid(self.out(hidden))
        print(predicted.shape)
        exit(0)
        return decoded
       
    
    def sampling(self, h_mean, h_sigma):
        std = torch.exp(h_sigma / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(h_mean)
        return x_sample

        #batch_size, hidden_size, _ = hmean.shape
        #epsilon = np.random.normal(0,1, (h_mean.shape))
        #return z_mean + torch.exp(h_sigma) * epsilon


    def get_latent(self, input):
        #out, (hidden, _) = self.enc1(input)
        #hidden = torch.cat((hidden[0], hidden[1]), 1)
        #out, (hidden, _) = self.enc2(out)
        #out, (hidden, _) = self.enc3(out)
        #hidden = hidden[-1]
        return hidden.squeeze()




class ConvolutionalAutoencoder(nn.Module):
    
    def __init__(self, in_size, filters):
        super().__init__()
        self.in_size = in_size
        self.filters = filters
        self.print = True

        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv1d(self.in_size, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, 3),
            nn.ReLU(),
            nn.Conv1d(16, 1, 7),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1, 16, 7),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 32, 3),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, self.in_size, 3, stride=2, padding=1),
            nn.Tanh(),
            nn.Linear(197, 200),
            nn.Tanh()
        )

        
    def forward(self, input):
        batch_size, L, features = input.shape
        input = input.view(batch_size, features, L)
        encoded = self.encoder(input)
        
        if self.print:
            print(encoded.shape)
            self.print = False
        reconstructed = self.get_reconstructed(encoded)
        return reconstructed

    def get_latent(self, input):
        batch_size, L, features = input.shape
        input = input.view(batch_size, features, L)
        x = self.encoder(input)
        return x.squeeze()

    def get_reconstructed(self, x):
        batch_size, _, _ = x.shape
        x = self.decoder(x)
        x = x.view(batch_size, -1, self.in_size)
        return x

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers = 1, dropout = 0.1, bidirectional = False):
        super(LSTMAutoEncoder, self).__init__()
        self.vec_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
       
        self.enc1 = nn.LSTM(input_size, self.hidden_size, num_layers = 2, batch_first = True)
        #self.enc2 = nn.LSTM(self.hidden_size*4, self.hidden_size*2, num_layers = 2, dropout = 0.5, batch_first = True) 
        #self.enc3 = nn.LSTM(self.hidden_size*2, self.hidden_size,  num_layers = 2, dropout = 0.5, batch_first = True) 

       
        self.dec1 = nn.LSTM(self.hidden_size, self.hidden_size, num_layers = 2, batch_first = True)
        self.dec2 = nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers = 2, batch_first = True)
        self.dense = nn.Linear(self.hidden_size//2, self.vec_size)
        self.tdd = TimeDistributed(self.dense, batch_first = True)
        self.print = True

    def forward(self, input):
        batch_size, steps, _ = input.shape
        if self.print:
            print(input.shape)
            self.print = False

        #print(input)
        out, (hidden, _) = self.enc1(input)
        hidden = hidden[-1]
        encoded = hidden.view(batch_size, 1, self.hidden_size).repeat(1, steps, 1)

        out, (hidden, _) = self.dec1(encoded)
        out, (hidden, _) = self.dec2(out)
        return self.tdd(out)
       
     


    def get_latent(self, input):
        out, (hidden, _) = self.enc1(input)
        #hidden = torch.cat((hidden[0], hidden[1]), 1)
        #out, (hidden, _) = self.enc2(out)
        #out, (hidden, _) = self.enc3(out)
        hidden = hidden[-1]
        return hidden.squeeze()


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1, drop = 0.1, bidirectional = False):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.NN = nn.LSTM(input_size, hidden_size, batch_first = True, num_layers = num_layers, bidirectional = bidirectional)

        self.encoder = nn.Sequential(
            nn.LSTM(input_size, 8, batch_first = True),
            nn.LSTM(8, 32,  batch_first = True) )

        self.decoder = nn.Sequential(
            nn.LSTM(32, 8, batch_first = True),
            nn.LSTM(8, input_size,  batch_first = True) )
       

        #nn.init.orthogonal_(self.NN.weight_ih_l0, gain=np.sqrt(2))
        #nn.init.orthogonal_(self.NN.weight_hh_l0, gain=np.sqrt(2))
     
    def forward(self, input):
        #print(input.shape)
        self.NN.flatten_parameters()
        batch_size, steps, features = input.shape
        output, (hidden, _) = self.encoder(input)
        print(hidden.shape)
        output, (hidden, _) = self.decoder(input)
        print()
        exit(0)
        return hidden
        

    def get_latent(self, input):
        batch_size, _, _ = input.shape
        output, (hidden, _) = self.NN(input)
        return hidden.squeeze()

        hidden = hidden.view(self.num_layers, 2 if self.bidirectional else 1, batch_size, self.hidden_size)
        hidden = hidden[-1]
        
        if self.bidirectional:
            hidden = torch.cat((hidden[0],hidden[1]), 1)

        print(hidden.shape)
        print(output.shape)
        return hidden.squeeze()

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers = 1, drop = 0.1, bidirectional = False):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = False
        self.output_size = output_size
        self.relu = nn.ReLU()

        self.NN = nn.LSTM(hidden_size * (2 if bidirectional else 1), output_size, batch_first = True, num_layers = num_layers, bidirectional = False)
        nn.init.orthogonal_(self.NN.weight_ih_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.NN.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, input):
        batch_size, steps, _ = input.shape
        self.NN.flatten_parameters()
        output, hidden = self.NN(input)
        return self.relu(output)

class TripletLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1, dropout = 0.1, bidirectional = False):
        super(TripletLSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.NN = nn.LSTM(input_size, hidden_size, bidirectional = bidirectional, num_layers = num_layers, batch_first = True, dropout = dropout)
        self.pr = True
        nn.init.orthogonal_(self.NN.weight_ih_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.NN.weight_hh_l0, gain=np.sqrt(2))

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
            hidden = torch.add(hidden[0], hidden[1])

        return hidden.view(batch_size, -1, self.hidden_size * (1 if self.bidirectional else 1))


    def get_latent(self, input):
        batch_size, _, _ = input.shape
        output, (hidden, _) = self.NN(input)
        hidden = hidden.view(self.num_layers, 2 if self.bidirectional else 1, batch_size, self.hidden_size)
        hidden = hidden[-1]
        if self.bidirectional:
            hidden = torch.add(hidden[0], hidden[1])
        return hidden.view(batch_size, -1, self.hidden_size * (1 if self.bidirectional else 1)).squeeze()


class TripletConvolutionalEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, filters, dropout):
        super(TripletConvolutionalEncoder, self).__init__()

        self.pr = True
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            #nn.Dropout(0.25),
            nn.Conv1d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            #nn.Dropout(0.25),
            nn.Conv1d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            #nn.Dropout(0.25),
            nn.Conv1d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            #nn.Dropout(0.25),
            nn.Conv1d(32, 16, 3, stride = 1),
            nn.ReLU(),
            #nn.Dropout(0.25),
            nn.Conv1d(16, 1, 7),
            nn.ReLU(),
        )

  
    def forward(self, a, p, n):
        enc_a = self.get_latent(a)
        enc_p = self.get_latent(p)
        enc_n = self.get_latent(n)
        return enc_a, enc_p, enc_n

    def get_latent(self, input):
        batch_size, L, features = input.shape
        input = input.view(batch_size, features, L)

        y = self.encoder(input)
        
        if self.pr:
            print(y.shape)
            self.pr = False

        return y.squeeze()
        