import os
from argparse import ArgumentParser
from collections import OrderedDict
import pickle
import numpy as np
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from pytorch_lightning import Callback
from pytorch_lightning import _logger as log
from pytorch_lightning.core import LightningModule

import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

from SignalDataset import StatsDataset, StatsSubsetDataset, TripletStatsDataset, TestDataset, StatsTestDataset
from util import knn, visualize, showPlot, plotOutput
from models import LSTMAutoEncoder, ConvolutionalAutoencoder, VariationalAutoencoder

from sklearn.cluster import KMeans


class AutoencoderModel(LightningModule):

    def __init__(self, hparams):
        """
        Pass in hyperparameters as a `argparse.Namespace` or a `dict` to the model.
        """
    
        super().__init__()
        self.hparams = hparams
        self.batch_size = hparams.batch_size

        # if you specify an example input, the summary will show input/output for each layer
        #self.example_input_array = torch.rand(5, 10, 3)
        #print(self.example_input_array.size(-1))
        #print(self.example_input_array.shape)

        # build model
        self.__build_model()
        self.in_vec = None
        self.in_vec_train = None
        


    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        #log.info('Build model called.')

        if self.hparams.type == "lstm":
            self.model = LSTMAutoEncoder(self.hparams.features, 
                self.hparams.hidden_size, 
                self.hparams.features,
                bidirectional = self.hparams.bidirectional, 
                num_layers = self.hparams.num_layers, 
                dropout = self.hparams.drop_prob)
        elif self.hparams.type == "conv":
            self.model = ConvolutionalAutoencoder(self.hparams.features, filters = self.hparams.filters)
        elif self.hparams.type == "vae":
            self.model = VariationalAutoencoder(self.hparams.features, self.hparams.hidden_size, self.hparams.features)
       

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, input):
        #log.info('Forward called.')
        return self.model(input)
        

    def get_latent(self, input):
        return self.model.get_latent(input)

    def loss(self, input, target, latents, labels):
        return  F.l1_loss(input, target) 
  
    def training_step(self, batch, batch_idx):
        input, l, target = batch
        output = self(input)
        latent = self.get_latent(input).squeeze()
        loss_val = self.loss(output, target, latent, l)

        if self.in_vec_train == None:
            self.in_vec_train = input[0]
            self.target_vec_train = target[0]

        
        tqdm_dict = {'train_loss': loss_val, 'step' : self.current_epoch}
        output = OrderedDict({
            'latent': latent.tolist(),
            'label' : l.tolist(),
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        return output

    ## method not called on last epoch!
    def training_epoch_end(self, outputs):
        if self.current_epoch % 1 == 0:
            in_vec = self.in_vec_train.view(1, -1, self.hparams.features)
            target_vec = self.target_vec_train.view(1, -1, self.hparams.features)
            out_vec = self(in_vec)
            image = plotOutput(in_vec[0].tolist(), out_vec[0].tolist(), target_vec[0].tolist())
            self.logger.experiment.add_image('output-train', image, self.current_epoch)

        if self.logger and (self.current_epoch % self.hparams.plot_every == 0 or  self.current_epoch == self.hparams.epochs - 1):
            latents = []
            labels = []
            for output in outputs:
                latent = output['latent']
                if self.hparams.batch_size == 1:
                    latent = [latent]
                latents.extend(latent)
                labels.extend(output['label'])

            image = visualize(latents, 
                labels,  
                self.hparams.threeD,
                False, 
                self.hparams.model + " " + self.hparams.type)

            if self.current_epoch == self.hparams.epochs - 1:
                log.info("Pickling...")
                pickle.dump(zip(latents, labels), open("autoencoder/train.p", "wb"))

            self.logger.experiment.add_image('train', image, self.current_epoch)

        return {}

    def validation_step(self, batch, batch_idx):
        input, l, target = batch
        output = self(input)
        
        latent = self.get_latent(input).squeeze()
        loss_val = self.loss(output, target, latent, l)

        
        if self.in_vec == None:
            self.in_vec = input[0]
            self.target_vec = target[0]


        output = OrderedDict({
            'latent' : latent.tolist(),
            'label' : l.tolist(),
            'val_loss': loss_val
        })
        # can also return just a scalar instead of a dict (return loss_val)
        return output
      

    def validation_epoch_end(self, outputs):
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        if self.current_epoch % 1 == 0:
            in_vec = self.in_vec.view(1, -1, self.hparams.features)
            target_vec = self.target_vec.view(1, -1, self.hparams.features)
            out_vec = self(in_vec)
            image = plotOutput(in_vec[0].tolist(), out_vec[0].tolist(), target_vec[0].tolist())
            self.logger.experiment.add_image('output', image, self.current_epoch)

  
        if outputs and self.logger and (self.current_epoch % self.hparams.plot_every == 0 or self.current_epoch == self.hparams.epochs - 1):
            latents = []
            labels = []
            for output in outputs:
                latent = output['latent']
                if self.hparams.batch_size == 1:
                    latent = [latent]
                latents.extend(latent)
                labels.extend(output['label'])

            if self.current_epoch == self.hparams.epochs - 1:
                log.info("Pickling...")
                pickle.dump(zip(latents, labels), open("autoencoder/val.p", "wb"))

            
            image = visualize(latents, 
                labels, 
                self.hparams.threeD,
                False,
                self.hparams.model + " " + self.hparams.type)

            self.logger.experiment.add_image('validation', image, self.current_epoch)


        val_loss_mean = 0
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

        val_loss_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean, 'step' : self.current_epoch}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
        return result

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        #scheduler = None
        return [optimizer], [scheduler]

    def __dataloader(self, set):
 
        dataset = self.train_dataset
        if set == "validation":
            dataset = self.validation_dataset
        elif set == "test":
            dataset = self.test_dataset

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers = 0, 
            drop_last = True)

        return loader



    def prepare_data(self):
        num_classes = self.hparams.num_classes
        
        print("Num classes: ", self.hparams.num_classes)
        '''if num_classes == 2:
            filename = "csv/parsed/stats_dataset-2class-400.csv"
        elif num_classes == 4:
            filename = "csv/parsed/stats_dataset-4class-400.csv"
        else:
            filename = "csv/parsed/stats_dataset-6class-400.csv"

        dataset = StatsDataset(filename)

        train_size = int(0.8 * len(dataset)) #int(0.8 * len(dataset))
        val_test_size = int((len(dataset) - train_size) * 0.5)
        test_size = len(dataset) - train_size - val_test_size

        if((train_size + val_test_size + test_size) != len(dataset)):
            train_size += (len(dataset) - val_test_size - test_size)
        
        print(train_size, val_test_size, test_size)
        train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,  val_test_size, test_size]) 

        train_dataset = StatsSubsetDataset(train_dataset, wrapped = False, minmax = self.hparams.min_max)
        validation_dataset = StatsSubsetDataset(validation_dataset, wrapped = False, minmax = self.hparams.min_max)
        test_dataset = StatsSubsetDataset(test_dataset, wrapped = False, minmax = self.hparams.min_max)

        pickle.dump(train_dataset, open("data/parsed/pickles/true/train-" + str(num_classes) + "-auto.p", "wb"))
        pickle.dump(test_dataset,  open("data/parsed/pickles/true/val-" + str(num_classes) + "-auto.p", "wb"))
        pickle.dump(validation_dataset,  open("data/parsed/pickles/true/test-" + str(num_classes) + "-auto.p", "wb"))
        #exit(0)
        '''
        
        
        dataset = StatsTestDataset()
        train_size = int(0.8 * len(dataset)) #int(0.8 * len(dataset))
        val_test_size = int((len(dataset) - train_size) * 0.5)
        test_size = len(dataset) - train_size - val_test_size

        if((train_size + val_test_size + test_size) != len(dataset)):
            train_size += (len(dataset) - val_test_size - test_size)
        
        print(train_size, val_test_size, test_size)
        self.train_dataset, self.validation_dataset, self.test_dataset = torch.utils.data.random_split(dataset, [train_size,  val_test_size, test_size]) 
        return
        self.train_dataset = StatsSubsetDataset(train_dataset, wrapped = False, minmax = self.hparams.min_max)
        self.validation_dataset = StatsSubsetDataset(validation_dataset, wrapped = False, minmax = self.hparams.min_max)
        self.test_dataset = StatsSubsetDataset(test_dataset, wrapped = False, minmax = self.hparams.min_max)
        return
        
        
        

        self.train_dataset = pickle.load(open("data/parsed/pickles/true/train-" + str(num_classes) + "-auto.p", "rb"))
        self.validation_dataset = pickle.load(open("data/parsed/pickles/true/val-" + str(num_classes) + "-auto.p", "rb"))
        self.test_dataset = pickle.load(open("data/parsed/pickles/true/test-" + str(num_classes) + "-auto.p", "rb"))
     
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


    ### TESTING

    def test_step(self, batch, batch_idx):
        input, l, target = batch
        latent = self.get_latent(input)

        output = OrderedDict({
            'latent': latent.squeeze().tolist(),
            'labels': l.tolist()
        })

        return output

    def test_epoch_end(self, outputs):
        log.info('Fitting predictor...')
        latents = []
        labels = []

        for output in outputs:
            latent = output['latent']
            
            if self.hparams.batch_size == 1:
                latent = [latent]
            latents.extend(output['latent'])
            labels.extend(output['labels'])

        log.info("Pickling...")
        pickle.dump(zip(latents, labels), open("autoencoder/test.p", "wb"))

        train_size = int(0.8 * len(latents))
        test_size = len(latents) - train_size
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
        correct = (test_labels == predicted).sum()
        all = len(test_labels)
        test_acc = correct/all

        #k_means_lab = []
        #for label in test_labels:
        #    if label == 3:
        #        k_means_lab.append(0)
        #    else:
        #        k_means_lab.append(1)

        '''km = KMeans(self.hparams.num_classes).fit(train_latents)
        predicted = km.predict(test_latents)
        correct = (labels == predicted).sum()
        print(k_means_lab)
        print(predicted)
        all = len(test_labels)
        test_acc_means = correct/all
        print("kmeans accuracy ", test_acc_means)'''

        if self.logger:
            self.logger.experiment.add_scalar('test_acc', test_acc)

        image = visualize(latents, 
            labels, 
            three_d = self.hparams.threeD,
            test = True,
            subtitle = self.hparams.model + " " + self.hparams.type)    

        '''km_image = visualize(latents, 
            km.predict(latents), 
            three_d = self.hparams.threeD,
            test = True,
            subtitle = self.hparams.model + " " + self.hparams.type)   '''
        
        if self.logger:
            self.logger.experiment.add_image('test', image, self.current_epoch)
            #self.logger.experiment.add_image('test_kmeans', km_image, self.current_epoch)


            # reduce manually when using dp
            #if self.trainer.use_dp or self.trainer.use_ddp2:
            #    test_loss = torch.mean(test_loss)

    
        tqdm_dict = {'test_acc': test_acc, 'step': self.current_epoch}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'test_acc': test_acc}
        return result
