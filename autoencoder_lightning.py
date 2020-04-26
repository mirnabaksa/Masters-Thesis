import os
from argparse import ArgumentParser
from collections import OrderedDict
import pickle
import numpy as np

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
from util import knn, visualize, showPlot
from models import LSTMAutoEncoder, ConvolutionalAutoencoder

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
        else:
            self.model = ConvolutionalAutoencoder(self.hparams.features, filters = self.hparams.filters)
        

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, input):
        #log.info('Forward called.')
        return self.model(input)
        

    def get_latent(self, input):
        return self.model.get_latent(input)

    def loss(self, input, target, latents, labels):
        #d = nn.PairwiseDistance(p=2)
        #print(labels)
        #latents = torch.tensor([[1, 0.0], [0, 2], [0.0, 3], [0.0, 0.0], [0,0], [0,0], [0,0], [0,0]])
        '''distance_error_pos = 0
        distance_error_neg = 0
        for label in labels:
            indices = [i for i, x in enumerate(labels) if x == label]
            other = [i for i, x in enumerate(labels) if x != label]
            #indices = [0,1,2]
            x = latents[indices]
            differences = x.unsqueeze(1) - x.unsqueeze(0)
            distances = torch.sum(differences * differences, -1).sum().div(2)
            distance_error_pos += distances / len(x)

            x = latents[other]
            differences = x.unsqueeze(1) - x.unsqueeze(0)
            distances = torch.sum(differences * differences, -1).sum().div(2)
            distance_error_neg += distances / len(x)
                
        #print(distance_error)  
        #exit(0)
        distance = distance_error_pos - distance_error_neg + 0.2
        distance = torch.mean(torch.max(distance, torch.zeros_like(distance))) '''
        #print(input.shape)
        #print(target.shape)
        return  F.l1_loss(input, target) 
  
    def training_step(self, batch, batch_idx):
        input, l, target = batch
        output = self(input)
        latent = self.get_latent(input).squeeze()
        loss_val = self.loss(output, target, latent, l)

        #if self.current_epoch == self.hparams.epochs - 1:
        #    print("####")
        #    print(input[0])
        #    print(output[0])
        #    print("#####")
        
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
        if self.logger and (self.current_epoch % self.hparams.plot_every == 0 or self.current_epoch == self.hparams.epochs - 1):
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
                "train-tsne.png",
                self.hparams.model + " " + self.hparams.type)

            if self.current_epoch == self.hparams.epochs - 1:
                log.info("Pickling...")
                pickle.dump(latents, open("latents-auto.p", "wb"))
                pickle.dump(labels, open("labels-auto.p", "wb"))
            self.logger.experiment.add_image('train', image, self.current_epoch)

        return {}

    def validation_step(self, batch, batch_idx):
        input, l, target = batch
        output = self(input)
        
        latent = self.get_latent(input).squeeze()
        loss_val = self.loss(output, target, latent, l)

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
  
        if outputs and self.logger and (self.current_epoch == 0 or self.current_epoch % self.hparams.plot_every == 0 or self.current_epoch == self.hparams.epochs - 1):
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
                "val-tsne.png", 
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
        if num_classes == 2:
            filename = "csv/perfect-stats2class.csv"
        elif num_classes == 4:
            filename = "csv/perfect-stats4class.csv"
        else:
            filename = "csv/perfect-stats.csv"


        dataset = StatsDataset(filename)
        #dataset = StatsTestDataset()
        train_size = int(0.7 * len(dataset)) 
        val_test_size = (len(dataset) - train_size) // 2
        #log.info("Dataset sizes: " + str(train_size) + " " + str(val_test_size))

        if((train_size + 2 * val_test_size) != len(dataset)):
            train_size += 1
        train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,  val_test_size, val_test_size]) 
        
        #self.train_dataset = train_dataset
        #self.validation_dataset = validation_dataset
        #self.test_dataset = test_dataset
        #return
        
        self.train_dataset = StatsSubsetDataset(train_dataset, wrapped = False, minmax = False)
        self.validation_dataset = StatsSubsetDataset(validation_dataset, wrapped = False, minmax = False)
        self.test_dataset = StatsSubsetDataset(test_dataset, wrapped = False, minmax = False)


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
        latents = pickle.load(open("latents-auto.p", "rb"))
        labels = pickle.load(open("labels-auto.p", "rb"))
        predictor = knn(latents, labels, 3)

        latents = []
        labels = []

        for output in outputs:
            latent = output['latent']
            if self.hparams.batch_size == 1:
                latent = [latent]
            latents.extend(output['latent'])
            labels.extend(output['labels'])

        predicted = predictor.predict(latents)
        correct = (labels == predicted).sum()
        all = len(labels)

        test_acc = correct/all
        if self.logger:
            self.logger.experiment.add_scalar('test_acc', test_acc)

        image = visualize(latents, 
            labels, 
            "tsne-test.png",
            self.hparams.model + " " + self.hparams.type)   
        
        if self.logger:
            self.logger.experiment.add_image('test', image, self.current_epoch)


            # reduce manually when using dp
            #if self.trainer.use_dp or self.trainer.use_ddp2:
            #    test_loss = torch.mean(test_loss)

        tqdm_dict = {'test_acc': test_acc, 'step': self.current_epoch}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'test_acc': test_acc}
        return result

