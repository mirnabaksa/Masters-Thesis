import torch
from torch.utils.data import Dataset, DataLoader

import h5py
import pandas as pd
import os
import numpy as np
from math import sin, cos, tan, atan, sqrt, exp

from statistics import mean, median, stdev
from scipy.stats import mode, hmean, gmean, entropy

import random
from random import uniform, randint, choice

from sklearn.preprocessing import StandardScaler, minmax_scale
from sklearn import preprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_printoptions(precision=10)
SOS_token = 10.0
EOS_token = 10.0

SOS_MEAN, SOS_GMEAN,  SOS_MEDIAN, SOS_STDEV, SOS_EN, SOS_IQR = 10,11,12,13,14, 14

class GeneralDataset(self, reference_csv, )

class TripletTestDataset(Dataset):
    def __init__(self):
        n_classes = 3
        n_points = 50

        self.positive = []
        self.negative = []
        for i in range(n_points):
            test_sample = []
            x = uniform(-0.5,0.5)
            #x_len = 4
            x_len = randint(30, 60)
            #self.data.append(([min(max(x+0.1*i, -1),1) for i in range(x_len)], 1))
            self.positive.append(([sin(0.1* x+0.05*i) for i in range(x_len)], "positive"))

        for i in range(n_points):
            test_sample = []
            x = uniform(-0.5,0.5)
            #x_len = 4
            x_len = randint(30, 60)
            #self.data.append(([min(max(x-0.2*i, -1), 1) for i in range(x_len)], 2))
            self.negative.append(([cos(x-0.08*i) for i in range(x_len)], "negative"))

        self.neutral = []
        for i in range(n_points):
            test_sample = []
            x = uniform(-0.5,0.5)
            #x_len = 4
            x_len = randint(30, 60)
            #self.data.append(([min(max(x-0.2*i, -1), 1) for i in range(x_len)], 2))
            self.neutral.append(([sin(2*x+0.1*i) for i in range(x_len)], "neutral"))


        self.triplets = []
        for i in range(n_points//2):
            a = self.positive[randint(0,49)]
            p = self.positive[randint(0,49)]
            n = self.negative[randint(0,49)]
            self.triplets.append((a,p,n, 1))

        for i in range(n_points//2):
            a = self.positive[randint(0,49)]
            p = self.positive[randint(0,49)]
            n = self.neutral[randint(0,49)]
            self.triplets.append((a,p,n, 1))

        for i in range(n_points//2):
            a = self.negative[randint(0,49)]
            p = self.negative[randint(0,49)]
            n = self.positive[randint(0,49)]
            self.triplets.append((a,p,n, 2))

        for i in range(n_points//2):
            a = self.negative[randint(0,49)]
            p = self.negative[randint(0,49)]
            n = self.neutral[randint(0,49)]
            self.triplets.append((a,p,n, 2))

        for i in range(n_points//2):
            a = self.neutral[randint(0,49)]
            p = self.neutral[randint(0,49)]
            n = self.negative[randint(0,49)]
            self.triplets.append((a,p,n, 3))

        for i in range(n_points//2):
            a = self.neutral[randint(0,49)]
            p = self.neutral[randint(0,49)]
            n = self.positive[randint(0,49)]
            self.triplets.append((a,p,n, 3))

        ''' 
        np.set_printoptions(precision=5)      
        print(np.array(self.data[1][0]))
        print(np.array(self.data[151][0]))
        print(np.array(self.data[201][0]))
        print(np.array(self.data[354][0]))
        '''

        self.n = n_points * 3

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        a, p, n, l = self.triplets[idx]
        return torch.FloatTensor([[i] for i in a[0]]), torch.FloatTensor([[i] for i in p[0]]), torch.FloatTensor([[i] for i in n[0]]), l

    def get_distinct_labels(self):
        return [1,2, 3]
        
class StatsDataset(Dataset):

    def __init__(self, reference_csv, raw = True):
        self.reference = pd.read_csv(reference_csv, delimiter = ",", header = None)
        self.startVector = [SOS_MEAN, SOS_STDEV, SOS_IQR, SOS_EN, SOS_MEDIAN]

        self.df = pd.DataFrame(columns = ["data", "label"])

        for i in range(1, len(self.reference)):
            index, label, data = self.reference.loc[i]
            data = [[float(mean), float(stdev), float(iqr), float(entropy), float(med)] for mean, gmean,  med, stdev, entropy, iqr in [line.split(",") for line in data.split("$")]]
            
            #means = minmax_scale(self.column(data, 0))
            #stdevs = minmax_scale(self.column(data, 1))
            #entropies = minmax_scale(self.column(data, 2))

            #data = np.column_stack((means, stdevs))
            #data = np.insert(data, 0, self.startVector, axis = 0)
            self.df = self.df.append({"data" : data, "label" : label}, ignore_index = True)


        self.n = len(self.df)

    def column(self, matrix, i):
        return [row[i] for row in matrix]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        data, label = self.df.loc[idx]
        return torch.FloatTensor(data), label

    def get_distinct_labels(self):
        return ["bacillus_anthracis", "ecoli", "pseudomonas_koreensis"]

    def get_vector_len(self):
        return len(self.startVector)

class StatsTestDataset(Dataset):

    def __init__(self):
        n_classes = 4
        n_points = 50
        self.chunk = 100
        self.startVector = [SOS_MEAN, SOS_STDEV]

        l = 30000
        r = 40000

        self.data = []
        for i in range(n_points):
            test_sample = []
            x_len = randint(l, r)
            data = [sin(-x*x+0.02) for x in range(300, 600)]
            stats_data = self.generateStatsData(data)
            self.data.append((stats_data, 1))

        for i in range(n_points):
            test_sample = []
            x = uniform(-0.2,0.2)
            x_len = randint(l, r)
            data = [x * exp(x) for x in range(300, 600)]
            stats_data = self.generateStatsData(data)
            self.data.append((stats_data, 2))  
     
        for i in range(n_points):
            test_sample = []
            x = uniform(-0.2,0.2)
            x_len = randint(l, r)
            data = [x * x *tan(2*x+0.1) for x in range(300, 600)]
            stats_data = self.generateStatsData(data)
            self.data.append((stats_data, 3)) 

        for i in range(n_points):
            test_sample = []
            x = uniform(-0.2,0.2)
            x_len = randint(l, r)
            data = [cos(-2*sqrt(x)+0.25) for x in range(300, 600)] 
            stats_data = self.generateStatsData(data)
            self.data.append((stats_data, 4))

        self.n = n_classes * n_points

    def generateStatsData(self, data):
        data = minmax_scale(data, feature_range = (0,1)) 
        stats_data = []
        for i in range(len(data)//self.chunk+1):
                data_chunk = data[i*self.chunk : (i+1)*self.chunk]
                if len(data_chunk) < 2:
                    continue
                
                if len(data_chunk) == 1:
                    stats_data.append([data_chunk[0], data_chunk[0], 0])
                    continue

                data_mean, data_median, data_stdev, en = mean(data_chunk), median(data_chunk),  stdev(data_chunk), entropy(data_chunk)
                stats_data.append([data_mean, data_stdev])
        return stats_data

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        data, label = self.data[idx]
        copy = data.copy()
        copy.insert(0, self.startVector)
        return torch.FloatTensor(copy), label

    def get_distinct_labels(self):
        return [1,2,3,4]

    def get_vector_len(self):
        return len(self.startVector)


class TestDataset(Dataset):

    def __init__(self):
        n_classes = 4
        n_points = 50

        self.data = []
        for i in range(n_points):
            test_sample = []
            x = uniform(-0.2,0.2)
            x_len = randint(15, 20)
            #self.data.append(([min(max(x+0.1*i, -1),1) for i in range(x_len)], 1))
            #self.data.append(([0.2 for i in range(x_len)], 1))
            self.data.append(([sin(0.1* x*x+0.05*i) for i in range(x_len)],1))

        for i in range(n_points):
            test_sample = []
            x = uniform(-0.2,0.2)
            x_len = randint(15, 20)
            #self.data.append(([min(max(x-0.2*i, -1), 1) for i in range(x_len)], 2))
            #self.data.append(([0.7 for i in range(x_len)], 2))
            self.data.append(([cos(x-0.08*i) for i in range(x_len)],2))    
     
        for i in range(n_points):
            test_sample = []
            x = uniform(-0.2,0.2)
            x_len = randint(15, 20)
            #self.data.append(([min(max(2*x - 0.5*i, -1), 1) for i in range(x_len)], 3))
            #self.data.append(([-0.3 for i in range(x_len)], 3))
            self.data.append(([sin(2*x+0.1*i) for i in range(x_len)],3))   

        for i in range(n_points):
            test_sample = []
            x = uniform(-0.2,0.2)
            x_len = randint(15, 20)
            #self.data.append(([min(max(-x + 0.3*i, -1), 1) for i in range(x_len)], 4))
            #self.data.append(([-0.5 for i in range(x_len)], 4))
            self.data.append(([cos(-2*x*x+0.25*i) for i in range(x_len)],4))   

        self.n = n_classes * n_points

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        data, label = self.data[idx]
        copy = data.copy()
        copy.insert(0, SOS_token)
        return torch.FloatTensor([[i] for i in copy]), label

    def get_distinct_labels(self):
        return [1,2,3,4]

class SignalTripletDataset(Dataset):

    def __init__(self, root_dir, reference_csv, raw = True):
        self.root_dir = root_dir
        self.reference = pd.read_csv(reference_csv, delimiter = ",", header = None)
        self.raw = raw
        self.n = len(self.reference)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        anchor, positive, negative, label = self.reference.iloc[idx]
        a, p, n = Signal(anchor), Signal(positive), Signal(negative)
        data_a = a.get_raw() if self.raw else a.get_pA()
        data_p = p.get_raw() if self.raw else p.get_pA()
        data_n = n.get_raw() if self.raw else n.get_pA()
        return torch.FloatTensor([[i] for i in data_a]), torch.FloatTensor([[i] for i in data_p]), torch.FloatTensor([[i] for i in data_n]), label

    def get_distinct_labels(self):
        return ["bacillus_anthracis", "ecoli"]#, "klebsiella_pneumoniae", "pantonea_agglomerans", "pseudomonas_koreensis", "yersinia_pestis"]

class SignalDataset(Dataset):

    def __init__(self, root_dir, reference_csv, raw = True):
        self.root_dir = root_dir
        self.reference = pd.read_csv(reference_csv, delimiter = ",", header = None)
        self.raw = raw
        self.n = len(self.reference)

    def __len__(self):
        print("signal dataset: ", self.n)
        return self.n

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        read_name, label = self.reference.iloc[idx]
        signal = Signal(read_name)
        data = signal.get_raw() if self.raw else signal.get_pA()
        copy = np.insert(data, 0, SOS_token, axis=0)
        
       # labels = [label for i in range(len(data))]
        return torch.FloatTensor([[i] for i in copy]), label
        #return list(torch.cuda.FloatTensor([[i] for i in data[j]]) for j in range(len(data))), label

    def get_distinct_labels(self):
        return ["bacillus_anthracis", "ecoli", "klebsiella_pneumoniae", "pantonea_agglomerans", "pseudomonas_koreensis", "yersinia_pestis"]


class Signal():
    def __init__(self, filename):
        f = h5py.File(filename, 'r')
        self.raw = np.array(f["Raw"]["Reads"]["Read_981"]["Signal"]).astype(np.float)
        self.metadata = f["UniqueGlobalKey"]["channel_id"]
        self.offset = self.metadata.attrs['offset']
        self.chunk = 250

        range = self.metadata.attrs['range']
        quantisation = self.metadata.attrs['digitisation']
        self.scale = range/quantisation

        
    
    def get_raw(self):
       # return minmax_scale(self.data[i*self.chunk : (i+1)*self.chunk for i in range(len(self.data)//self.chunk)])
        return minmax_scale(self.raw[:100])
        #return list(minmax_scale(self.raw[i*self.chunk : (i+1)*self.chunk]) for i in range(len(self.raw)//self.chunk))

    def get_pA(self):
        return minmax_scale((self.scale * (self.raw + self.offset))[:200])
        #return list(minmax_scale(
        #    self.scale * (self.raw[i*self.chunk : (i+1)*self.chunk] + self.offset))
        #    for i in range(len(self.raw)//self.chunk)
        #    )