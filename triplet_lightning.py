import os
from argparse import ArgumentParser
from collections import OrderedDict
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader


from pytorch_lightning import Callback
from pytorch_lightning import _logger as log
from pytorch_lightning.core import LightningModule


from SignalDataset import StatsDataset, StatsSubsetDataset, TripletStatsDataset
from util import knn, visualize, showPlot
from models import TripletLSTMEncoder, TripletConvolutionalEncoder

class TripletModel(LightningModule):

    def __init__(self, hparams):
        """
        Pass in hyperparameters as a `argparse.Namespace` or a `dict` to the model.
        """
    
        super().__init__()
        self.hparams = hparams
        self.batch_size = hparams.batch_size
        # if you specify an example input, the summary will show input/output for each layer
        #self.example_input_array = (torch.rand(64, 75, 3), torch.rand(64, 75, 3), torch.rand(64, 75, 3))
        #print(self.example_input_array.shape)

        # build model
        self.__build_model()



    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        log.info('Build model called.')
        if self.hparams.type == "lstm":
            self.model = TripletLSTMEncoder(self.hparams.features, 
                self.hparams.hidden_size, 
                bidirectional = self.hparams.bidirectional, 
                num_layers = self.hparams.num_layers, 
                dropout = self.hparams.drop_prob)
        else:
            self.model = TripletConvolutionalEncoder(self.hparams.features,
                filters = self.hparams.filters,
                dropout = self.hparams.drop_prob)
        

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, a, p, n):
        return self.model(a, p, n)
        

    def get_latent(self, input):
        return self.model.get_latent(input)

    def loss(self, a, p, n, margin = 0.2):
        d = nn.PairwiseDistance(p=2)
        distance = d(a, p) - d(a, n) + margin 
        loss = torch.mean(torch.max(distance, torch.zeros_like(distance))) 
        return loss

  
    def training_step(self, batch, batch_idx):
        in_a, in_p, in_n, l = batch
        a, p, n = self(in_a, in_p, in_n)
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

    ## method not called on last epoch!
    def training_epoch_end(self, outputs):
        if self.logger and (self.current_epoch == 0 or self.current_epoch % self.hparams.plot_every == 0 or self.current_epoch == self.hparams.epochs - 1):
            latents = []
            labels = []
            for output in outputs:
                latents.extend(output['latent'])
                labels.extend(output['label'])

            image = visualize(latents, 
                labels,  
                "train-tsne.png", 
                self.hparams.model + " " + self.hparams.type)

            if self.current_epoch == self.hparams.epochs - 1:
                log.info("Pickling...")
                pickle.dump(latents, open("latents-triplet.p", "wb"))
                pickle.dump(labels, open("labels-triplet.p", "wb"))
            self.logger.experiment.add_image('train', image, self.current_epoch)
            

        return {}

    def validation_step(self, batch, batch_idx):
        in_a, in_p, in_n, l = batch
        a, p, n = self(in_a, in_p, in_n)
        loss_val = self.loss(a, p, n)

        output = OrderedDict({
            'latent': a.squeeze().tolist(),
            'label' : l.tolist(),
            'val_loss': loss_val
        })

        return output
      

    def validation_epoch_end(self, outputs):
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()
        if self.logger and (self.current_epoch == 0 or self.current_epoch % self.hparams.plot_every == 0 or self.current_epoch == self.hparams.epochs - 1):
            latents = []
            labels = []
            for output in outputs:
                latents.extend(output['latent'])
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
        return [optimizer] , [scheduler]

    def __dataloader(self, set):
        # when using multi-node (ddp) we need to add the  datasampler
 
        dataset = self.train_dataset
        if set == "validation":
            dataset = self.validation_dataset
        elif set == "test":
            dataset = self.test_dataset

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers = 3,
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

        train_size = int(0.7 * len(dataset)) #int(0.8 * len(dataset))
        val_test_size = (len(dataset) - train_size) // 2
        if((train_size + 2 * val_test_size) != len(dataset)):
            train_size += 1
        train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,  val_test_size, val_test_size]) 
        
        self.train_dataset = TripletStatsDataset(StatsSubsetDataset(train_dataset, wrapped = True, minmax = False))
        self.validation_dataset = TripletStatsDataset(StatsSubsetDataset(validation_dataset, wrapped = True, minmax = False))
        self.test_dataset = TripletStatsDataset(StatsSubsetDataset(test_dataset, wrapped = True, minmax = False))


    def train_dataloader(self):
        return self.__dataloader(set="train")

    def val_dataloader(self):
        return self.__dataloader(set="validation")

    def test_dataloader(self):
        return self.__dataloader(set="test")

    def test_step(self, batch, batch_idx):
        in_a, _, _, l = batch
        a = self.get_latent(in_a)

        output = OrderedDict({
            'latent': a.squeeze().tolist(),
            'labels': l.tolist()
        })

        return output

    def test_epoch_end(self, outputs):
        log.info('Fitting predictor...')
        latents = pickle.load(open("latents-triplet.p", "rb"))
        labels = pickle.load(open("labels-triplet.p", "rb"))
        predictor = knn(latents, labels, 3)

        latents = []
        labels = []

        for output in outputs:
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

        tqdm_dict = {'test_acc': test_acc}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'test_acc': test_acc}
        return result


    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  
        parser = ArgumentParser(parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)

        # network params
        parser.add_argument('--model', default="lstm", type=str)
        parser.add_argument('--features', default=3, type=int)
        parser.add_argument('--num_layers', default=1, type=int)
        parser.add_argument('--hidden_size', default=10, type=int)
        parser.add_argument('--drop_prob', default=0.2, type=float)
        #parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--bidirectional', default = False, type=bool)

        # training params (opt)
        parser.add_argument('--epochs', default=100, type=int)
        parser.add_argument('--optimizer_name', default='adam', type=str)
        parser.add_argument('--batch_size', default=64, type=int)

        parser.add_argument('--plot_every', default=5, type=int)

        parser.add_argument('--num_classes', default=2, type=int)
        return parser


