import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from time import time
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers = 1):
        super(AutoEncoder, self).__init__()
        self.vec_size = input_size
        self.encoder = Encoder(input_size, hidden_size, n_layers)
        self.decoder = Decoder(hidden_size, output_size, n_layers)

    def forward(self, input_tensor, target_tensor, max_len, lens):
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


    def get_latent(self, input):
        return self.encoder.get_latent(input)


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers = 1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.NN = nn.LSTM(input_size, hidden_size, num_layers = n_layers, batch_first = True)
        self.relu = nn.ReLU()

    def forward(self, input, hidden = None):
        self.NN.flatten_parameters()
        output, hidden = self.NN(input, hidden)
        return output, hidden

    def get_latent(self, input):
        _, hidden = self.NN(input, None)
        return hidden[-1].squeeze()

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers = 1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.NN = nn.LSTM(output_size, hidden_size, num_layers = n_layers, batch_first = True)
        self.out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden = None):
        self.NN.flatten_parameters()
        output, hidden = self.NN(input, hidden)
       # output = self.sigmoid(self.out(output))
        output = self.out(output)
        return output, hidden

class TripletEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers = 1):
        super(TripletEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.encoder = nn.GRU(input_size, hidden_size, num_layers = 1, batch_first = True)

    def forward(self, a, p, n, lens, hidden = None):
        a = pack_padded_sequence(a, lens[0], batch_first = True, enforce_sorted = False)
        p = pack_padded_sequence(p, lens[1], batch_first = True, enforce_sorted = False)
        n = pack_padded_sequence(n, lens[2], batch_first = True, enforce_sorted = False)

        _, hidden_a = self.encoder(a, hidden)
        _, hidden_p = self.encoder(p, hidden)
        _, hidden_n = self.encoder(n, hidden)
        return hidden_a, hidden_p, hidden_n

    def get_latent(self, input):
        self.encoder.flatten_parameters()
        _, hidden = self.encoder(input, None)
        return hidden
