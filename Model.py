import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from time import time
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers = 1, dropout = 0.1):
        super(AutoEncoder, self).__init__()
        self.vec_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.encoder = Encoder(input_size, hidden_size, n_layers, dropout)
        self.decoder = Decoder(hidden_size, output_size, n_layers, dropout)

    def forward(self, input, target_tensor, max_len, lens):
        '''batch_size, steps, features = target_tensor.shape
        _, (last_hidden, _) = self.encoder(input)
        if (self.n_layers > 1):
            last_hidden = last_hidden[-1]
        encoded = last_hidden.view(batch_size, 1, self.hidden_size).repeat(1, steps, 1)
        y, _ = self.decoder(encoded)
        return y
        '''
        #print(input)
        input_tensor = input
        batch_size = len(target_tensor)
        input_tensor = pack_padded_sequence(input_tensor, lens, batch_first = True, enforce_sorted = False)

        _, encoder_hidden = self.encoder(input_tensor, None)
        decoder_hidden = encoder_hidden
        decoder_input = target_tensor[:,0].view(batch_size, 1, self.vec_size)
        outputs = torch.cuda.FloatTensor(batch_size, max_len, self.vec_size).fill_(0)
     
        for di in range(1, max_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[:,di] = decoder_output.view(batch_size, self.vec_size)
            decoder_input = decoder_output.detach() 

        for i in range(len(lens)):
            outputs[i,lens[i].item():] = 0
        
        return outputs
        #'''

    def get_latent(self, input):
        latent = self.encoder.get_latent(input)
        if self.n_layers > 1:
            latent = latent[-1]
        return latent


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers = 1, drop = 0.1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.NN = nn.LSTM(input_size, hidden_size, num_layers = n_layers, dropout = drop, batch_first = True)
        self.relu = nn.ReLU()

    def forward(self, input, hidden = None):
        self.NN.flatten_parameters()
        output, hidden = self.NN(input, hidden)
        return output, hidden

    def get_latent(self, input):
        _, (hidden, _) = self.NN(input, None)
        return hidden.squeeze()

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers = 1, drop = 0.1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.NN = nn.LSTM(output_size, hidden_size, dropout = drop, num_layers = n_layers, batch_first = True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden = None):
        self.NN.flatten_parameters()
        output, hidden = self.NN(input, hidden)
       # output = self.sigmoid(self.out(output))
        output = self.out(output)
        return output, hidden

class TripletEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers = 1, dropout = 0.1):
        super(TripletEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.NN = nn.LSTM(input_size, hidden_size, num_layers = n_layers, batch_first = True, dropout = dropout)

    def forward(self, a, p, n, len_a, len_p, len_n):
        self.NN.flatten_parameters()
        tot_a = a.size(1)
        tot_p = p.size(1)
        tot_n = n.size(1)

        batch_size, seq_len, features = a.shape
        #print(a.shape)
        #a = pack_padded_sequence(a, len_a, batch_first = True, enforce_sorted = False)
        #p = pack_padded_sequence(p, len_p, batch_first = True, enforce_sorted = False)
        #n = pack_padded_sequence(n, len_n, batch_first = True, enforce_sorted = False)

        out_a, (hidden_a, _) = self.NN(a)
        out_p, (hidden_p, _) = self.NN(p)
        out_n, (hidden_n, _) = self.NN(n)

        if self.n_layers > 1:
            hidden_a = hidden_a[-1]
            hidden_p = hidden_p[-1]
            hidden_n = hidden_n[-1]

        #print("##")
        #print(a.shape)
        #print(out_a.shape)
        #print(hidden_a)
        hidden_a = hidden_a.view(batch_size, -1, self.hidden_size)
        hidden_p = hidden_p.view(batch_size, -1, self.hidden_size)
        hidden_n = hidden_n.view(batch_size, -1, self.hidden_size)
        #print("##")
        
        return hidden_a, hidden_p, hidden_n

    def get_latent(self, input):
        self.NN.flatten_parameters()
        _, (hidden, _) = self.NN(input)
        if (self.n_layers > 1):
            return hidden[-1]
        return hidden
