import torch
from torch.utils.data import Dataset

import h5py
import pandas as pd
import os
import numpy as np
from math import sin, cos, tan, atan, sqrt, exp

from statistics import mean, median, stdev
from scipy.stats import mode, hmean, gmean, entropy, iqr

import random
from random import uniform, randint

from sklearn.preprocessing import StandardScaler, minmax_scale, normalize

import pickle 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_printoptions(precision=10)

class TripletStatsDataset(Dataset):

    def __init__(self, data = None, pickled = None, num_samples = 2, test_set = False):
        print("Is test set: ", test_set)
        if data is not None:    
            self.flat_data = pd.DataFrame(columns = ["stats_data", "label"])
            for i in range(len(data)):
                stat, label, target = data[i]
                self.flat_data = self.flat_data.append({"stats_data" : stat, "label" : label}, ignore_index = True)
        elif pickled is not None:
            data = pickle.load(open(pickled, "rb"))
            seq_data = []
            for latent, label in data:
                sample = [[i] for i in latent]
                seq_data.append([sample, label])

            self.flat_data = pd.DataFrame(seq_data, columns = ["stats_data", "label"])
        else:
            print("No data provided")
            exit(0)

        self.data = pd.DataFrame(columns = ["anchor", "positive", "negative", "label"])

        for stats, label in self.flat_data.values.tolist():
            if test_set:
                self.data = self.data.append({"anchor" : stats, "positive" : stats, "negative" : stats, "label" : label}, ignore_index = True)
                continue

            for other_label in self.get_distinct_labels():
                if label == other_label:
                    continue

              
                pos =  self.flat_data.loc[self.flat_data['label'] == label]
                neg = self.flat_data.loc[self.flat_data['label'] == other_label]
                

                if pos.empty or neg.empty:
                    continue
                    
                pos = pos.sample(min(len(pos),num_samples))
                neg = neg.sample(min(len(neg),num_samples))

                for i in range(min(len(pos),len(neg))):
                    positive, labp = pos.values[i]
                    negative, labn = neg.values[i]
                    self.data = self.data.append({"anchor" : stats, "positive" : positive, "negative" : negative, "label" : label}, ignore_index = True)
           
        self.n = len(self.data)
        print("Effective length: ", self.n)

    def __getitem__(self, idx):
        anchor, positive, negative, label = self.data.iloc[idx]
        return torch.tensor(anchor), torch.tensor(positive), torch.tensor(negative), torch.tensor(label)

    def __len__(self):
        return self.n

    def get_distinct_labels(self):
        return [0,1,2,3,4,5]

class StatsSubsetDataset(Dataset):

    def __init__(self, subset, wrapped = False, seq2seq = False, minmax = False, noisy = False):
        self.subset = subset
        self.n = len(subset)
        self.n_stats = len(subset[0][0][0])
        self.seq2seq = seq2seq
        self.wrapped = wrapped
        self.noisy = noisy

        print("N: ", self.n)
        print("N stats: ", self.n_stats)

        cols = [[] for j in range(self.n_stats)]

        for point in self.subset:
            matrix, label = point
            for col in range(self.n_stats):
                column = self.column(matrix, col)
                cols[col].extend(column)
        
        col_stats = []
        for col in cols:  
            if minmax:
                col_min, col_max = min(col), max(col)
                col_stats.append((col_min, col_max))
            else:
                col_mean, stdev_mean = mean(col), stdev(col)
                col_stats.append((col_mean, stdev_mean))

                print(col_mean, stdev_mean)
        
        self.df = pd.DataFrame(columns = ["data", "label", "target"])
        for point in self.subset:
            matrix, label = point
            data = []
            for row in matrix:
                data_row = []
                for i in range(len(row)):
                    if minmax:
                        col_min, col_max = col_stats[i]
                        data_row.append((row[i] - col_min)/(col_max - col_min))
                    
                    else:
                        col_mean, col_stdev = col_stats[i]
                        data_row.append((row[i] - col_mean) / col_stdev)
                
                data.append(data_row)

            target = data
            if self.noisy:
                noise = np.random.normal(0,0.5, (len(data), len(data[0])))
                data = (np.array(data) + noise).tolist()
            self.df = self.df.append({"data" : data, "label" : self.get_label(label), "target" : target}, ignore_index = True)


    def get_label(self, label):
        if label == "Enter":
            return 0
        if label == "Staph":
            return 1
        if label == "Liste":
            return 2
        if label == "Lacto":
            return 3
        if label == "Bacil":
            return 4
        if label == "Esche":
            return 5
     
    '''def get_label(self, label):
        if label == "ecoli":
            return 0
        if label == "pseudomonas_koreensis":
            return 1
        if label == "yersinia_pestis":
            return 2
        if label == "pantonea_agglomerans":
            return 3
        if label == "bacillus_anthracis":
            return 4
        if label == "klebsiella_pneumoniae":
            return 5
        else:
            return label'''



    def column(self, matrix, i):
        return [row[i] for row in matrix]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        data, label, target = self.df.loc[idx]
        if self.wrapped:
            return data, label, target
        return torch.tensor(data), torch.tensor(label), torch.tensor(target)
    

    def get_distinct_labels(self):
        return [0,1,2,3,4,5]
        #return ["bacillus_anthracis", "ecoli", "pseudomonas_koreensis", "yersinia_pestis", "pantonea_agglomerans", "klebsiella_pneumoniae"]
    
    def get_vector_len(self):
        return self.n_stats


class StatsDataset(Dataset):

    def __init__(self, reference_csv, raw = True):
        self.reference = pd.read_csv(reference_csv, delimiter = ",", header = None)
        self.df = pd.DataFrame(columns = ["data", "label"])
        
        for i in range(1, len(self.reference)):
            index, label, data = self.reference.loc[i]
            data = [[float(mean), float(med), float(stdev), float(iqr)] for mean, med, stdev, iqr in [line.split(",") for line in data.split("$")]]
            self.df = self.df.append({"data" : data, "label" : label}, ignore_index = True)
        self.n = len(self.df)

    def column(self, matrix, i):
        return [row[i] for row in matrix]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        data, label = self.df.loc[idx]
        return data, label

    def get_distinct_labels(self):
        return ["bacillus_anthracis", "ecoli", "pseudomonas_koreensis", "pantonea_agglomerans", "yersinia_pestis", "klebsiella_pneumoniae"]


class PickledDataset(Dataset):

    def __init__(self, pickled, raw = True):
        data = pickle.load(open(pickled, "rb"))
        self.seq_data = []
        for latent, label in data:
            sample = [[i] for i in latent]
            self.seq_data.append([sample, label])
            print(sample)

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, idx):
        data, label = self.seq_data[idx]
        return data, label

    def get_distinct_labels(self):
        return ["bacillus_anthracis", "ecoli", "pseudomonas_koreensis", "pantonea_agglomerans", "yersinia_pestis", "klebsiella_pneumoniae"]

       

class StatsTestDataset(Dataset):

    def __init__(self):
        n_classes = 2
        n_points = 50
        self.chunk = 2
        x_len = 40

        self.data = []
        for i in range(n_points):
            x = uniform(-1,1)
            data = [2 * sin(j - x) for j in np.arange(0, random.randint(1,3), 0.1)]
            self.data.append((data, "Enter"))
        
        for i in range(n_points):
            x = uniform(-1,1)
            data = [ sin(2 * j + x) for j in np.arange(0, random.randint(1,3), 0.1)]
            data = self.generateStatsData(data)
            self.data.append((data, "Staph"))

        
        '''for i in range(n_points):
            x = uniform(-1,1)
            data = [sin(4*j *j -x) for j in np.arange(0, x_len, 0.1)]
            data = self.generateStatsData(data)
            
            self.data.append((data, 2))
       

        for i in range(n_points):
            x = uniform(-1,1)
            data = [2 * j * sin(-x - j) for j in np.arange(0, x_len, 0.1)]
            data = self.generateStatsData(data)
            self.data.append((data, 3))'''
        
        self.n = len(self.data)

        return 
    
    def generateStatsData(self, data):
        stats_data = []
        
        for i in range(len(data)//self.chunk+1):
                data_chunk = data[i*self.chunk : (i+1)*self.chunk]
                
                if len(data_chunk) == 0:
                    continue
                
                if len(data_chunk) == 1:
                    stats_data.append([data_chunk[0], 0])
                    continue

                data_mean, data_median, data_stdev, en, data_iqr = mean(data_chunk), median(data_chunk),  stdev(data_chunk), entropy(data_chunk), iqr(data_chunk)
                stats_data.append([data_mean, data_stdev])
        return stats_data

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        data, label = self.data[idx]
        return data, label

    def get_distinct_labels(self):
        return [1,2,3,4]

    def get_vector_len(self):
        return len(self.startVector)


class TestDataset(Dataset):

    def __init__(self):
        n_classes = 2
        n_points = 200


        self.data = []
        for i in range(n_points):
            test_sample = []
            x_len = 20
            x = uniform(0,0.5)
            data = [sin(j*j) for j in np.arange(0, x_len, 0.1)]
            self.data.append((data, 1))

        for i in range(n_points):
            test_sample = []
            x_len = 20
            x = uniform(0.5,1)
            data = [cos(j*j) for j in np.arange(0, x_len, 0.1)]
            self.data.append((data, 2))

        '''for i in range(n_points):
            test_sample = []
            x_len = 200
            x = uniform(0.5,1)
            data = [-2 * j + 4 for j in np.arange(0, 1, 0.01)]
            self.data.append((data, 3))
        '''
    
        self.n = len(self.data)
        return

        self.data = []
        for i in range(n_points):
            test_sample = []
            x = uniform(-1,1)
            x_len = randint(100, 150)
            #self.data.append(([min(max(x+0.1*i, -1),1) for i in range(x_len)], 1))
            #self.data.append(([0.2 for i in range(x_len)], 1))
            self.data.append(([0.1* x*x+0.05*i for i in range(x_len)],1))

        for i in range(n_points):
            test_sample = []
            x = uniform(-5,5)
            x_len = randint(100, 150)
            #self.data.append(([min(max(x-0.2*i, -1), 1) for i in range(x_len)], 2))
            #self.data.append(([0.7 for i in range(x_len)], 2))
            self.data.append(([x-0.08*i for i in range(x_len)],2))    
     
        for i in range(n_points):
            test_sample = []
            x = uniform(-5,5)
            x_len = randint(100, 150)
            #self.data.append(([min(max(2*x - 0.5*i, -1), 1) for i in range(x_len)], 3))
            #self.data.append(([-0.3 for i in range(x_len)], 3))
            self.data.append(([2*x+0.1*i for i in range(x_len)],3))   

        for i in range(n_points):
            test_sample = []
            x = uniform(-5,5)
            x_len = randint(100, 150)
            #self.data.append(([min(max(-x + 0.3*i, -1), 1) for i in range(x_len)], 4))
            #self.data.append(([-0.5 for i in range(x_len)], 4))
            self.data.append(([-2*x*x+0.25*i for i in range(x_len)],4))   

        self.n = n_classes * n_points

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        data, label = self.data[idx]
        return torch.FloatTensor([[i] for i in data]), torch.tensor(label), torch.FloatTensor([[i] for i in data])

    def get_distinct_labels(self):
        return [1,2,3,4]


class TripletTestDataset(Dataset):
    def __init__(self):
        n_classes = 3
        n_points = 40

        self.positive = []
        self.negative = []
        x_len = 10
        for i in range(n_points):
            test_sample = []
            x = uniform(-0.5,0.5)
            #x_len = 4
            #x_len = randint(30, 60)
            #self.data.append(([min(max(x+0.1*i, -1),1) for i in range(x_len)], 1))
            self.positive.append(([sin(0.1* x+0.05*i) for i in range(x_len)], "positive"))

        for i in range(n_points):
            test_sample = []
            x = uniform(-0.5,0.5)
            #x_len = randint(30, 60)
            #self.data.append(([min(max(x-0.2*i, -1), 1) for i in range(x_len)], 2))
            self.negative.append(([cos(x-0.08*i) for i in range(x_len)], "negative"))

        self.neutral = []
        for i in range(n_points):
            test_sample = []
            x = uniform(-0.5,0.5)
            #x_len = randint(30, 60)
            #self.data.append(([min(max(x-0.2*i, -1), 1) for i in range(x_len)], 2))
            self.neutral.append(([tan(2*x+0.1*i) for i in range(x_len)], "neutral"))


        self.triplets = []
        for i in range(n_points//2):
            a = self.positive[randint(0,n_points-1)]
            p = self.positive[randint(0,n_points-1)]
            n = self.negative[randint(0,n_points-1)]
            self.triplets.append((a,p,n, 1))

        for i in range(n_points//2):
            a = self.positive[randint(0,n_points-1)]
            p = self.positive[randint(0,n_points-1)]
            n = self.neutral[randint(0,n_points-1)]
            self.triplets.append((a,p,n, 1))

        for i in range(n_points//2):
            a = self.negative[randint(0,n_points-1)]
            p = self.negative[randint(0,n_points-1)]
            n = self.positive[randint(0,n_points-1)]
            self.triplets.append((a,p,n, 2))

        for i in range(n_points//2):
            a = self.negative[randint(0,n_points-1)]
            p = self.negative[randint(0,n_points-1)]
            n = self.neutral[randint(0,n_points-1)]
            self.triplets.append((a,p,n, 2))

        for i in range(n_points//2):
            a = self.neutral[randint(0,n_points-1)]
            p = self.neutral[randint(0,n_points-1)]
            n = self.negative[randint(0,n_points-1)]
            self.triplets.append((a,p,n, 3))

        for i in range(n_points//2):
            a = self.neutral[randint(0,n_points-1)]
            p = self.neutral[randint(0,n_points-1)]
            n = self.positive[randint(0,n_points-1)]
            self.triplets.append((a,p,n, 3))

        self.n = len(self.triplets)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        a, p, n, l = self.triplets[idx]
        return torch.FloatTensor(a[0]), torch.FloatTensor(p[0]), torch.FloatTensor(n[0]), l

    def get_distinct_labels(self):
        return [1,2,3]


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
        return ["bacillus_anthracis", "ecoli", "klebsiella_pneumoniae", "pantonea_agglomerans", "pseudomonas_koreensis", "yersinia_pestis"]

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
        
        return torch.FloatTensor([[i] for i in copy]), label

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
        return list(minmax_scale(self.raw[i*self.chunk : (i+1)*self.chunk]) for i in range(len(self.raw)//self.chunk))

    def get_pA(self):
        return minmax_scale((self.scale * (self.raw + self.offset)))