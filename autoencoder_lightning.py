import os
from argparse import ArgumentParser
from collections import OrderedDict

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

from SignalDataset import StatsDataset, StatsSubsetDataset, TripletStatsDataset
from util import knn, visualize, showPlot
from models import LSTMAutoEncoder

class AutoencoderModel(LightningModule):

    def __init__(self, hparams):
        """
        Pass in hyperparameters as a `argparse.Namespace` or a `dict` to the model.
        """
    
        super().__init__()
        self.hparams = hparams
        self.batch_size = hparams.batch_size

        # if you specify an example input, the summary will show input/output for each layer
        #self.example_input_array = torch.rand(3, 75, 16)
        #print(self.example_input_array.shape)

        # build model
        self.__build_model()


    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        #log.info('Build model called.')
        '''self.NN = nn.LSTM(self.hparams.features, 
            self.hparams.hidden_size, 
            bidirectional = self.hparams.bidirectional, 
            num_layers = self.hparams.n_layers, 
            dropout = self.hparams.drop_prob,
            batch_first = True)
        '''
        #self.model = nn.Linear(self.hparams.features, self.hparams.hidden_size)

        if self.hparams.type == "lstm":
            self.model = LSTMAutoEncoder(self.hparams.features, 
                self.hparams.hidden_size, 
                self.hparams.features,
                bidirectional = self.hparams.bidirectional, 
                num_layers = self.hparams.num_layers, 
                dropout = self.hparams.drop_prob)
        

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, input):
        #log.info('Forward called.')
        return self.model(input)
        

    def get_latent(self, input):
        return self.model.get_latent(input)

    def loss(self, input, target):
        return F.l1_loss(input, target)

  
    def training_step(self, batch, batch_idx):
        input, l = batch
        output = self(input)
        loss_val = self.loss(output, input)

        latent = self.get_latent(input)
        
        tqdm_dict = {'train_loss': loss_val}
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
        log.info("training epoch end")
        if self.current_epoch % self.hparams.plot_every == 0 or self.current_epoch == self.hparams.epochs - 1:
            self.latents = []
            self.labels = []
            for output in outputs:
                self.latents.extend(output['latent'])
                self.labels.extend(output['label'])

            image = visualize(self.latents, 
                self.labels, 
                self.train_dataset.get_distinct_labels(), 
                "train-tsne.png",
                self.hparams.model + " " + self.hparams.type)

            self.logger.experiment.add_image('train', image, self.current_epoch)

        return {}

    def validation_step(self, batch, batch_idx):
        input, l = batch
        output = self(input)
        
        loss_val = self.loss(output, input)
        latent = self.get_latent(input)

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
        if self.current_epoch == 0 or self.current_epoch % self.hparams.plot_every == 0 or self.current_epoch == self.hparams.epochs - 1:
            self.latents = []
            self.labels = []
            for output in outputs:
                #print(output['latent'].shape)
                #exit(0)
                self.latents.extend(output['latent'])
                self.labels.extend(output['label'])

            image = visualize(self.latents, 
                self.labels, 
                self.train_dataset.get_distinct_labels(), 
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
        tqdm_dict = {'val_loss': val_loss_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
        return result

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        scheduler = None
        return [optimizer] #, [scheduler]

    def __dataloader(self, set):
        # when using multi-node (ddp) we need to add the  datasampler
 
        transform = transforms.Compose([transforms.ToTensor()])

        dataset = self.train_dataset
        if set == "validation":
            dataset = self.validation_dataset
        elif set == "test":
            dataset = self.test_dataset

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers = 0)

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

        train_size = int(0.8 * len(dataset)) 
        val_test_size = (len(dataset) - train_size) // 2
        if((train_size + 2 * val_test_size) != len(dataset)):
            train_size += 1
        train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,  val_test_size, val_test_size]) 
        
        
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
        input, l = batch
        latent = self.get_latent(input)

        output = OrderedDict({
            'latent': latent.squeeze().tolist(),
            'labels': l.tolist()
        })

        return output

    def test_epoch_end(self, outputs):
        log.info('Fitting predictor...')
        return 
        predictor = knn(self.latents, self.labels, 3)

        latents = []
        labels = []

        for output in outputs:
            latents.extend(output['latent'].squeeze().tolist())
            labels.extend(output['labels'])

        predicted = predictor.predict(latents)
        correct = (labels == predicted).sum()
        all = len(labels)

        test_acc = correct/all
        self.logger.experiment.add_scalar('test_acc', test_acc)

        image = visualize(latents, 
            labels, 
            self.train_dataset.get_distinct_labels(), 
            "tsne-test.png",
            self.hparams.model + " " + self.hparams.type)   
        self.logger.experiment.add_image('test', image, self.current_epoch)


            # reduce manually when using dp
            #if self.trainer.use_dp or self.trainer.use_ddp2:
            #    test_loss = torch.mean(test_loss)

        tqdm_dict = {'test_acc': test_acc}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'test_acc': test_acc}
        return result

