import os
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
from models import LSTMAutoEncoder, ConvolutionalAutoencoder, LSTMVariationalAutoencoder, VariationalAutoencoder, ConvVariationalAutoencoder

from sklearn.cluster import KMeans


class AutoencoderModel(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.batch_size = hparams.batch_size

        # build model
        self.__build_model()
        self.in_vec = None
        self.in_vec_train = None
        


    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
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
            self.model = LSTMVariationalAutoencoder(self.hparams.features, self.hparams.hidden_size, self.hparams.features)
       

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, input):
        return self.model(input)
        

    def get_latent(self, input):
        return self.model.get_latent(input)

    def loss(self, output, target):
        if self.hparams.type == "vae":
            return self.vae_loss(output, target)
        else:
            return F.l1_loss(output[0], target) 

    def vae_loss(self, output, target):
        model_out, mu, logvar = output
        BCE = F.mse_loss(model_out, target, reduction = "sum") 
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
  
    def training_step(self, batch, batch_idx):
        input, l, target = batch
        output = self(input)
        latent = self.get_latent(input).squeeze()
        loss_val = self.loss(output, target)

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


    def training_epoch_end(self, outputs):
        if self.current_epoch % 1 == 0:
            in_vec = self.in_vec_train.view(1, -1, self.hparams.features)
            target_vec = self.target_vec_train.view(1, -1, self.hparams.features)
            out_vec,_,_ = self(in_vec)
            out_vec = out_vec.view(1,-1,self.hparams.features)
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

            self.logger.experiment.add_image('train', image, self.current_epoch)

        return {}

    def validation_step(self, batch, batch_idx):
        input, l, target = batch
        output = self(input)
        
        latent = self.get_latent(input).squeeze()
        loss_val = self.loss(output, target)

        
        if self.in_vec == None:
            self.in_vec = input[0]
            self.target_vec = target[0]


        output = OrderedDict({
            'latent' : latent.tolist(),
            'label' : l.tolist(),
            'val_loss': loss_val
        })

        return output
      

    def validation_epoch_end(self, outputs):
        if self.current_epoch % 1 == 0:
            in_vec = self.in_vec.view(1, -1, self.hparams.features)
            target_vec = self.target_vec.view(1, -1, self.hparams.features)
            out_vec,_,_ = self(in_vec)
            out_vec = out_vec.view(1,-1,self.hparams.features)
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
        tqdm_dict = {'val_loss': val_loss_mean, 'step' : self.current_epoch}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
        return result

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
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

        train_dataset = StatsSubsetDataset(train_dataset, wrapped = False, minmax = self.hparams.min_max, noisy = False)
        validation_dataset = StatsSubsetDataset(validation_dataset, wrapped = False, minmax = self.hparams.min_max, noisy  =False)
        test_dataset = StatsSubsetDataset(test_dataset, wrapped = False, minmax = self.hparams.min_max, noisy = False)

        pickle.dump(train_dataset, open("data/parsed/pickles/true/train-" + str(num_classes) + "-auto.p", "wb"))
        pickle.dump(validation_dataset,  open("data/parsed/pickles/true/val-" + str(num_classes) + "-auto.p", "wb"))
        pickle.dump(test_dataset,  open("data/parsed/pickles/true/test-" + str(num_classes) + "-auto.p", "wb"))
        #exit(0)'''
        
        
        '''
        dataset = StatsTestDataset()
        train_size = int(0.8 * len(dataset)) #int(0.8 * len(dataset))
        val_test_size = int((len(dataset) - train_size) * 0.5)
        test_size = len(dataset) - train_size - val_test_size

        if((train_size + val_test_size + test_size) != len(dataset)):
            train_size += (len(dataset) - val_test_size - test_size)
        
        print(train_size, val_test_size, test_size)
        train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,  val_test_size, test_size]) 
        #return

        self.train_dataset = StatsSubsetDataset(train_dataset, wrapped = False, minmax = self.hparams.min_max, noisy = True)
        self.validation_dataset = StatsSubsetDataset(validation_dataset, wrapped = False, minmax = self.hparams.min_max, noisy = True)
        self.test_dataset = StatsSubsetDataset(test_dataset, wrapped = False, minmax = self.hparams.min_max, noisy = True)
        return
        '''
        

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


        if self.logger:
            self.logger.experiment.add_scalar('test_acc', test_acc)

        image = visualize(latents, 
            labels, 
            three_d = self.hparams.threeD,
            test = True,
            subtitle = self.hparams.model + " " + self.hparams.type)    


        if self.logger:
            self.logger.experiment.add_image('test', image, self.current_epoch)

        tqdm_dict = {'test_acc': test_acc, 'step': self.current_epoch}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'test_acc': test_acc}
        return result
