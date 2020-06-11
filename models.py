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

        intermediate = hidden_size//2
        
        self.enc = nn.Sequential(
            nn.Linear(836, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            )

        self.mu = nn.Linear(200, 64)
        self.var = nn.Linear(200, 64)

        self.dec = nn.Sequential(
            nn.Linear(64, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 836)
            )
        
        self.print = True


    def forward(self, input):
        batch_size, steps, features = input.shape
        if self.print:
            print(input.shape)
            self.print = False

        input = input.flatten(start_dim = 1)
        hidden = self.enc(input)
        
        z_mu = self.mu(hidden)
        z_var = self.var(hidden)

        #sampling
        sample = self.reparameterize(z_mu, z_var)
        out = self.dec(sample)
        out = torch.sigmoid(out).view(input.shape)
        return out, z_mu, z_var
       
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def get_latent(self, input):
        batch_size, steps, features = input.shape
        input = input.flatten(start_dim = 1)
        hidden = self.enc(input)
        return hidden

class ConvVariationalAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ConvVariationalAutoencoder, self).__init__()
        self.in_size = input_size
        self.hidden_size = hidden_size

        self.mu = nn.Linear(47, hidden_size)
        self.var = nn.Linear(47, hidden_size)

        self.enc = nn.Sequential(
            nn.Conv1d(self.in_size, 8, 3, padding=1),
            nn.PReLU(),
            nn.Conv1d(8, 16, 3, padding=1),
            nn.PReLU(),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.PReLU(),
            nn.Conv1d(32, 64, 3),
            nn.PReLU(),
            nn.Conv1d(64, 16, 3),
            nn.PReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 1, 5),
            nn.PReLU(),
            nn.MaxPool1d(2),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(1, 16, 5, stride = 2, padding=1),
            nn.PReLU(),
            nn.ConvTranspose1d(16, 32, 3, stride = 2, padding=1),
            nn.PReLU(),
            nn.ConvTranspose1d(32, self.in_size, 3, stride = 2, padding=1),
            nn.PReLU(),
            nn.Linear(81, 200),
            nn.PReLU()
        )

        
        self.print = True


    def forward(self, input):
        batch_size, steps, features = input.shape
        input = input.view(batch_size, features, steps)

        #print(input.shape)
        hidden = self.enc(input)
        #print(hidden)
        #print(hidden.shape)
        
        if self.print:
            print(input.shape)
            print(hidden.shape)
            self.print = False
        
        z_mu = self.mu(hidden)
        z_var = self.var(hidden)

        #sampling
        sample = self.sampling(z_mu, z_var)
       # sample = sample.view(batch_size, 1, -1)

        out = self.dec(sample)
        out = out.view((batch_size, steps, features))

        return out, z_mu, z_var
       
    
    def sampling(self, h_mean, h_sigma):
        std = torch.exp(h_sigma / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(h_mean)
        return x_sample

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std


    def get_latent(self, input):
        batch_size, steps, features = input.shape
        #print(input)
        input = input.view(batch_size, features, steps)
        hidden = self.enc(input)
        #print(hidden)
        #exit(0)
        return hidden.squeeze()
        #print(hidden.shape)
        
        z_mu = self.mu(hidden)
        z_var = self.var(hidden)

        #sampling
        sample = self.reparameterize(z_mu, z_var)
        return hidden.squeeze()

class LSTMVariationalAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMVariationalAutoencoder, self).__init__()
        self.in_size = input_size
        self.hidden_size = hidden_size
        intermediate = hidden_size*2
        
        # encoder
        self.enc = nn.LSTM(input_size, hidden_size, bidirectional = True, batch_first = True)
        self.mu = nn.Linear(hidden_size*2, hidden_size)
        self.var = nn.Linear(hidden_size*2, hidden_size)

        #decoder
        self.dec = nn.LSTM(hidden_size, hidden_size, bidirectional = True, batch_first = True)
        self.dense = nn.Linear(hidden_size * 2, input_size)
        self.tdd = TimeDistributed(self.dense, batch_first = True)
        self.activation = nn.Tanh()
        self.print = True


    def forward(self, input):
        batch_size, steps, features = input.shape
        if self.print:
            print(input.shape)
            self.print = False

        #encoder
        out, (hidden, _) = self.enc(input)
        hidden = hidden.view(batch_size, 1, -1)
        hidden = hidden.repeat(1,steps,1)
        z_mu = self.mu(hidden)
        z_var = self.var(hidden)

        #sampling
        sample = self.reparameterize(z_mu, z_var)
        out, _  = self.dec(sample)
        return self.activation(self.tdd(out)), z_mu, z_var
       
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std


    def get_latent(self, input):
        batch_size, steps, features = input.shape
        out, (hidden, _) = self.enc(input)
        hidden = hidden.view(batch_size, 1, -1)
        return hidden.squeeze()


class ConvolutionalAutoencoder(nn.Module):
    
    def __init__(self, in_size, filters):
        super().__init__()
        self.in_size = in_size
        self.filters = filters
        self.print = True

        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv1d(self.in_size, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            #nn.MaxPool1d(2),
            nn.Conv1d(32, 1, 3, stride=1, padding=1),
            nn.MaxPool1d(2),
            nn.ReLU(),
            #nn.Linear(52, 32),
            #nn.Conv1d(32, 16, 3),
            #nn.ReLU(),
            #nn.Conv1d(16, 1, 7),
            #nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            #nn.Linear(32, 52),
            nn.ConvTranspose1d(1, 8, 3, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 16, 3),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, self.in_size, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Linear(212, 209),
            #nn.ReLU(),
            #nn.PReLU()
        )

        
    def forward(self, input):
        batch_size, L, features = input.shape
        input = input.view(batch_size, features, L)
        encoded = self.encoder(input)
        
        if self.print:
            print(encoded.shape)
            self.print = False
        reconstructed = self.get_reconstructed(encoded)
        return reconstructed, None, None

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
       
        self.enc1 = nn.LSTM(input_size, self.hidden_size, num_layers = 1, dropout = 0.25, bidirectional = True, batch_first = True)
        #self.enc2 = nn.LSTM(self.hidden_size*4, self.hidden_size*2, num_layers = 2, dropout = 0.5, batch_first = True) 
        #self.enc3 = nn.LSTM(self.hidden_size*2, self.hidden_size,  num_layers = 2, dropout = 0.5, batch_first = True) 

       
        self.dec1 = nn.LSTM(self.hidden_size, self.hidden_size, num_layers = 1, dropout = 0.25, bidirectional = True, batch_first = True)
        self.dense = nn.Linear(self.hidden_size * 2, input_size)
        #self.dec2 = nn.LSTM(self.hidden_size, self.hidden_size//2, num_layers = 2, batch_first = True)
        #self.dense = nn.Linear(self.hidden_size, self.vec_size)
        #self.tdd = TimeDistributed(self.dense, batch_first = True)
        #self.act = nn.PReLU()
        self.print = True

    def forward(self, input):
        batch_size, steps, _ = input.shape
        if self.print:
            print(input.shape)
            self.print = False

        #print(input)
        out, (hidden, _) = self.enc1(input)
        hidden = hidden.view(1, 2, batch_size, self.hidden_size)
        hidden = hidden[-1]
        hidden = torch.add(hidden[0], hidden[1])

        encoded = hidden.view(batch_size, 1, self.hidden_size).repeat(1, steps, 1)
        out, (hidden, _) = self.dec1(encoded)
        out = self.dense(out)
        #out, (hidden, _) = self.dec2(out)
        return out, None, None
       
     
    def get_latent(self, input):
        batch_size, steps, _ = input.shape
        out, (hidden, _) = self.enc1(input)
        #hidden = torch.cat((hidden[0], hidden[1]), 1)
        #out, (hidden, _) = self.enc2(out)
        #out, (hidden, _) = self.enc3(out)
        hidden = hidden.view(1, 2, batch_size, self.hidden_size)
        hidden = hidden[-1]
        hidden = torch.add(hidden[0], hidden[1])

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
        #print(hidden.shape)
        output, (hidden, _) = self.decoder(input)
        #print()
        #exit(0)
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
    def __init__(self, input_size, hidden_size, batch_size, num_layers = 1, dropout = 0.1, bidirectional = False):
        super(TripletLSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.bidirectional = bidirectional

        self.NN = nn.LSTM(input_size, 16, bidirectional = self.bidirectional, num_layers = num_layers, batch_first = True, dropout = dropout)
        self.NN2 = nn.LSTM(32, 48, bidirectional = False, num_layers = num_layers, batch_first = True, dropout = dropout)
        #self.NN3 = nn.LSTM(hidden_size*2, hidden_size, bidirectional = self.bidirectional, num_layers = num_layers, batch_first = True, dropout = dropout)

        self.pr = True
        #
        #nn.init.orthogonal_(self.NN.weight_ih_l0, gain=np.sqrt(2))
        #nn.init.orthogonal_(self.NN.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, a, p, n, len_a, len_p, len_n):
        batch_size, _, _ = a.shape
        input_shape = a.shape

        a = torch.nn.utils.rnn.pack_padded_sequence(a, len_a, batch_first=True, enforce_sorted=False)
        p = torch.nn.utils.rnn.pack_padded_sequence(p, len_p, batch_first=True, enforce_sorted=False)
        n = torch.nn.utils.rnn.pack_padded_sequence(n, len_n, batch_first=True, enforce_sorted=False)
        

        out_a = self.output(a, batch_size)
        out_p = self.output(p, batch_size)
        out_n = self.output(n, batch_size)

        if self.pr:
            print(input_shape)
            print(out_a.shape)
            self.pr = False

        return out_a, out_p, out_n

    def output(self, input, batch_size):
        out, (hidden, c) = self.NN(input)
        out, (hidden, c) = self.NN2(out)
        #out = self.drop(out)
        #_, (hidden, _) = self.NN2(out)

        #hidden = hidden.view(self.num_layers, 2 if self.bidirectional else 1, batch_size, 24)
        #hidden = hidden[-1]

        #if self.bidirectional:
        #    hidden = torch.cat((hidden[0], hidden[1]), 1)

        return hidden.squeeze()


    def get_latent(self, input, len_in):
        batch_size, _, _ = input.shape
        return self.output(input, batch_size)


class TripletConvolutionalEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, filters, dropout):
        super(TripletConvolutionalEncoder, self).__init__()

        self.pr = True
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Conv1d(in_size, 16, 3,  padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Conv1d(16, 32, 3,  padding=1),
            nn.ReLU(),   
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            
            #nn.Dropout(0.1),

            nn.Conv1d(32, 64, 3,  padding=1),
            nn.ReLU(),  
            nn.Dropout(0.1),

            nn.Conv1d(64, 1, 5,  padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),

            #nn.Flatten(),
            #nn.Linear(3328, 100),
            #nn.PReLU(),
            nn.Linear(51, 48),
            #nn.Softmax()
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
        