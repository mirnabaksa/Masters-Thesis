import os
from collections import OrderedDict
import pickle
from sklearn import metrics
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from pytorch_lightning import Callback
from pytorch_lightning import _logger as log
from pytorch_lightning.core import LightningModule

from SignalDataset import StatsDataset, StatsSubsetDataset, TripletStatsDataset, StatsTestDataset
from util import knn, visualize, showPlot
from models import TripletLSTMEncoder, TripletConvolutionalEncoder

def collate(input):
    a, p, n, labels = map(list, zip(*input))

    a, len_a = get_input(a)
    p, len_p = get_input(p)
    n, len_n = get_input(n)

    return a, p, n, len_a,len_p, len_n, torch.tensor(labels)

def get_input(in_batch):
    padded = pad_sequence(in_batch, batch_first = True, padding_value = 0)
    lens = [len(x) for x in in_batch]
    return padded, torch.IntTensor(lens)

class TripletModel(LightningModule):

    def __init__(self, hparams):
    
        super().__init__()
        self.hparams = hparams
        self.batch_size = hparams.batch_size
        self.__build_model()


    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        log.info('Build model called.')
        if self.hparams.type == "lstm":
            self.model = TripletLSTMEncoder(self.hparams.features, 
                self.hparams.hidden_size, 
                self.hparams.batch_size,
                bidirectional = self.hparams.bidirectional, 
                num_layers = self.hparams.num_layers, 
                dropout = self.hparams.drop_prob)
        else:
            self.model = TripletConvolutionalEncoder(self.hparams.features,
                hidden_size = self.hparams.hidden_size,
                filters = self.hparams.filters,
                dropout = self.hparams.drop_prob)
        

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, a, p, n, len_a, len_p, len_n):
        return self.model(a, p, n, len_a, len_p, len_n)
        

    def get_latent(self, input, len_in):
        return self.model.get_latent(input, len_in)

    def loss(self, a, p, n): 
        margin = self.hparams.margin
        d = nn.PairwiseDistance(p=2)
        distance = d(a, p) - d(a, n) + margin 
        loss = torch.mean(torch.max(distance, torch.zeros_like(distance))) 
        return loss
  
    def training_step(self, batch, batch_idx):
        in_a, in_p, in_n, len_a, len_p, len_n, l = batch
        a, p, n = self(in_a, in_p, in_n, len_a, len_p, len_n)
        loss_val = self.loss(a, p, n)
        
        tqdm_dict = {'train_loss': loss_val, 'step' : self.current_epoch}
        output = OrderedDict({
            'latent': a.squeeze().tolist(),
            'label' : l.tolist(),
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        return output

    def training_epoch_end(self, outputs):
        if self.logger and (self.current_epoch % self.hparams.plot_every == 0 or self.current_epoch == self.hparams.epochs - 1):
            latents = []
            labels = []
            for output in outputs:
                latents.extend(output['latent'])
                labels.extend(output['label'])

            image = visualize(latents, 
                labels,  
                self.hparams.threeD,
                False, 
                self.hparams.model + " " + self.hparams.type)

            self.logger.experiment.add_image('train', image, self.current_epoch)
            
        return {}

    def validation_step(self, batch, batch_idx):
        in_a, in_p, in_n, len_a, len_p, len_n, l = batch
        a, p, n = self(in_a, in_p, in_n, len_a, len_p, len_n)
        loss_val = self.loss(a, p, n)

        output = OrderedDict({
            'latent': a.squeeze().tolist(),
            'label' : l.tolist(),
            'val_loss': loss_val
        })

        return output
      

    def validation_epoch_end(self, outputs):
        if self.logger and (self.current_epoch % self.hparams.plot_every == 0 or self.current_epoch == self.hparams.epochs - 1):
            latents = []
            labels = []
            for output in outputs:
                latents.extend(output['latent'])
                labels.extend(output['label'])
            
            image = visualize(latents, 
                labels, 
                self.hparams.threeD,
                False,
                self.hparams.model + " " + self.hparams.type)

            self.logger.experiment.add_image('validation', image, self.current_epoch)
            
        val_loss_mean = 0
        for output in outputs:
            val_loss = output['val_loss']

            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

        val_loss_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean,'step' : self.current_epoch}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
        return result

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer

   
    def __dataloader(self, set):
        dataset = self.train_dataset
        if set == "validation":
            dataset = self.validation_dataset
        elif set == "test":
            dataset = self.test_dataset

        loader = DataLoader(
            dataset=dataset,
            collate_fn = collate,
            shuffle = True,
            batch_size=self.hparams.batch_size)

        return loader


    def prepare_data(self):
        num_classes = self.hparams.num_classes
        print("In prepare data")

        ## Zymo data
        '''if num_classes == 2:
            filename = "csv/loman/2-class-4000.csv"
        elif num_classes == 4:
            filename = "csv/loman/4-class-4000.csv"
        else:
            filename = "csv/loman/6-class.csv"

        dataset = StatsDataset(filename)

        train_size = int(0.8 * len(dataset)) 
        val_test_size = int((len(dataset) - train_size) * 0.5)
        test_size = len(dataset) - train_size - val_test_size

        if((train_size + val_test_size + test_size) != len(dataset)):
            train_size += (len(dataset) - val_test_size - test_size)
        
        train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,  val_test_size, test_size]) 

        train_dataset = TripletStatsDataset(StatsSubsetDataset(train_dataset, wrapped = True, minmax = self.hparams.min_max))
        validation_dataset = TripletStatsDataset(StatsSubsetDataset(validation_dataset, wrapped = True, minmax = self.hparams.min_max))
        test_dataset = TripletStatsDataset(StatsSubsetDataset(test_dataset, wrapped = True, minmax = self.hparams.min_max), test_set = True)

        pickle.dump(train_dataset, open("data/loman/train-" + str(num_classes) + ".p", "wb"))
        pickle.dump(validation_dataset,  open("data/loman/val-" + str(num_classes) + ".p", "wb"))
        pickle.dump(test_dataset,  open("data/loman/test-" + str(num_classes) + ".p", "wb"))'''
        

        #self.train_dataset = pickle.load(open("data/loman/train-" + str(num_classes) + ".p", "rb"))
        #self.validation_dataset = pickle.load(open("data/loman/val-" + str(num_classes) + ".p", "rb"))
        #self.test_dataset = pickle.load(open("data/loman/test-" + str(num_classes) + ".p", "rb"))
        

        ## Artificial Data
        '''if num_classes == 2:
            filename = "csv/parsed/stats_dataset-2class-400.csv"
        elif num_classes == 4:
            filename = "csv/parsed/stats_dataset-4class-400.csv"
        else:
            filename = "csv/parsed/stats_dataset-6class-400.csv"

        dataset = StatsDataset(filename)
        print("Here")

        train_size = int(0.8 * len(dataset)) #int(0.8 * len(dataset))
        val_test_size = int((len(dataset) - train_size) * 0.5)
        test_size = len(dataset) - train_size - val_test_size

        if((train_size + val_test_size + test_size) != len(dataset)):
            train_size += (len(dataset) - val_test_size - test_size)
        
        print(train_size, val_test_size, test_size)
        train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,  val_test_size, test_size]) 

        train_dataset = TripletStatsDataset(StatsSubsetDataset(train_dataset, wrapped = True, minmax = self.hparams.min_max))
        validation_dataset = TripletStatsDataset(StatsSubsetDataset(validation_dataset, wrapped = True, minmax = self.hparams.min_max))
        test_dataset = TripletStatsDataset(StatsSubsetDataset(test_dataset, wrapped = True, minmax = self.hparams.min_max), test_set = True)

        pickle.dump(train_dataset, open("data/parsed/pickles/true/train-" + str(num_classes) + ".p", "wb"))
        pickle.dump(validation_dataset,  open("data/parsed/pickles/true/val-" + str(num_classes) + ".p", "wb"))
        pickle.dump(test_dataset,  open("data/parsed/pickles/true/test-" + str(num_classes) + ".p", "wb"))
        exit(0)
        '''
    
        self.train_dataset = pickle.load(open("data/parsed/pickles/true/train-" + str(num_classes) + ".p", "rb"))
        self.validation_dataset = pickle.load(open("data/parsed/pickles/true/val-" + str(num_classes) + ".p", "rb"))
        self.test_dataset = pickle.load(open("data/parsed/pickles/true/test-" + str(num_classes) + ".p", "rb"))


        ## Test Data
        '''dataset = StatsTestDataset()
        train_size = int(0.8 * len(dataset)) #int(0.8 * len(dataset))
        val_test_size = int((len(dataset) - train_size) * 0.5)
        test_size = len(dataset) - train_size - val_test_size

        if((train_size + val_test_size + test_size) != len(dataset)):
            train_size += (len(dataset) - val_test_size - test_size)
        
        print(train_size, val_test_size, test_size)
        train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,  val_test_size, test_size]) 

        self.train_dataset = TripletStatsDataset(StatsSubsetDataset(train_dataset, wrapped = True, minmax = self.hparams.min_max))
        self.validation_dataset = TripletStatsDataset(StatsSubsetDataset(validation_dataset, wrapped = True, minmax = self.hparams.min_max))
        self.test_dataset = TripletStatsDataset(StatsSubsetDataset(test_dataset, wrapped = True, minmax = self.hparams.min_max), test_set = True)
        '''
        
     
        string = str(self.model)
        if self.hparams.type == "conv":
            string = string.replace("))", "))<br>")
            string = string.replace("True)", "True)<br>")
            string = string.replace("False)", "False)<br>")
        self.logger.experiment.add_text("model", string)
      

    def train_dataloader(self):
        return self.__dataloader(set="train")

    def val_dataloader(self):
        return self.__dataloader(set="validation")

    def test_dataloader(self):
        return self.__dataloader(set="test")

    def test_step(self, batch, batch_idx):
        in_a, _, _, len_a, _, _, l = batch
        a = self.get_latent(in_a, len_a)

        output = OrderedDict({
            'latent': a.squeeze().tolist(),
            'labels': l.tolist()
        })

        return output

    def test_epoch_end(self, outputs):
        log.info('Fitting predictor...')
        latents = []
        labels = []

        for output in outputs:
            latents.extend(output['latent'])
            labels.extend(output['labels'])

        train_size = int(0.8 * len(labels))
        test_size = len(labels) - train_size
        print(train_size, test_size)

        train_latents, test_latents = latents[:train_size], latents[train_size:]
        train_labels, test_labels = labels[:train_size], labels[train_size:]
        k = int(sqrt(len(latents)))
        if k%2 == 0:
            k += 1

        print("Using k... ", k)
        self.logger.experiment.add_text("K", str(k))
        predictor = knn(train_latents, train_labels, k)
        predicted = predictor.predict(test_latents)

        # Metrics
        acc = metrics.accuracy_score(test_labels, predicted)
        precision = metrics.precision_score(test_labels, predicted, average='macro')
        recall = metrics.recall_score(test_labels, predicted, average='macro')
        f1 = metrics.f1_score(test_labels, predicted, average='macro')

        print("Calculated metrics: ", acc, recall, precision, f1)
        
        if self.logger:
            self.logger.experiment.add_scalar('info/test_acc', acc)
            self.logger.experiment.add_scalar('info/test_recall', recall)
            self.logger.experiment.add_scalar('info/test_precision', precision)
            self.logger.experiment.add_scalar('info/test_f1', f1)

        image = visualize(latents, 
            labels, 
            three_d = self.hparams.threeD,
            test = True,
            subtitle = self.hparams.model + " " + self.hparams.type)  

        if self.logger:
            self.logger.experiment.add_image('test', image, self.current_epoch)

        tqdm_dict = {'test_acc': acc}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'test_acc': acc}
        return result


