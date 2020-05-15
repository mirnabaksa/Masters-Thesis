import os
from argparse import ArgumentParser
from collections import OrderedDict
import pickle
from sklearn import metrics
from math import sqrt
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


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
                hidden_size = self.hparams.hidden_size,
                filters = self.hparams.filters,
                dropout = self.hparams.drop_prob)
        

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, a, p, n):
        return self.model(a, p, n)
        

    def get_latent(self, input):
        return self.model.get_latent(input)

    def loss(self, a, p, n): 
        margin = self.hparams.margin
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

            #if self.current_epoch == self.hparams.epochs - 1:
            #    print(latents)

            image = visualize(latents, 
                labels,  
                self.hparams.threeD,
                False, 
                self.hparams.model + " " + self.hparams.type)

            #if self.current_epoch == self.hparams.epochs - 1:
            #    log.info("Pickling...")
            #    pickle.dump(latents, open("latents-triplet.p", "wb"))
            #    pickle.dump(labels, open("labels-triplet.p", "wb"))
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
            
            #print(latents)

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
        tqdm_dict = {'val_loss': val_loss_mean,'step' : self.current_epoch}
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

        '''def collate(input):
            a, p, n, labels = map(list, zip(*input))
            a, l_a = get_input(a)
            p, l_p = get_input(p)
            n, l_n = get_input(n)
            print("in collate")
    
            return a, p, n, l_a, l_p, l_n, labels

        def get_input(in_batch):
            padded = pad_sequence(in_batch, batch_first = True, padding_value = 0)
            lens = [len(x) for x in in_batch]
            return padded, torch.IntTensor(lens)'''

 
        dataset = self.train_dataset
        if set == "validation":
            dataset = self.validation_dataset
        elif set == "test":
            dataset = self.test_dataset
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers = 3)

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
        
        #torch.save(train_dataset, "data/parsed/train-4-4-200.p")
        #torch.save(validation_dataset, "data/parsed/val-4-4-200.p")
        #torch.save(test_dataset, "data/parsed/test-4-4-200.p")
        print(len(dataset))
        
        
        #train_dataset, validation_dataset, test_dataset = torch.load("data/parsed/train-" + str(num_classes) +  "-" + str(self.hparams.features) + "-200.p"), \
        #    torch.load("data/parsed/val-" + str(num_classes) + "-" + str(self.hparams.features) +  "-200.p"), \
        #    torch.load("data/parsed/test-" + str(num_classes) +  "-" + str(self.hparams.features)  + "-200.p")

        train_dataset = TripletStatsDataset(StatsSubsetDataset(train_dataset, wrapped = True, minmax = self.hparams.min_max))
        validation_dataset = TripletStatsDataset(StatsSubsetDataset(validation_dataset, wrapped = True, minmax = self.hparams.min_max))
        test_dataset = TripletStatsDataset(StatsSubsetDataset(test_dataset, wrapped = True, minmax = self.hparams.min_max))

        pickle.dump(train_dataset, open("data/parsed/pickles/true/train-" + str(num_classes) + ".p", "wb"))
        pickle.dump(test_dataset,  open("data/parsed/pickles/true/val-" + str(num_classes) + ".p", "wb"))
        pickle.dump(validation_dataset,  open("data/parsed/pickles/true/test-" + str(num_classes) + ".p", "wb"))
        #exit(0)
        '''

        self.train_dataset = pickle.load(open("data/parsed/pickles/true/train-" + str(num_classes) + ".p", "rb"))
        self.validation_dataset = pickle.load(open("data/parsed/pickles/true/val-" + str(num_classes) + ".p", "rb"))
        self.test_dataset = pickle.load(open("data/parsed/pickles/true/test-" + str(num_classes) + ".p", "rb"))

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
        in_a, _, _, l = batch
        a = self.get_latent(in_a)

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

        #print(len(latents))

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
        prob = predictor.predict_proba(test_latents)
        #for i in range(len(prob)):
        #    print(prob[i], test_labels[i]) 

        acc = metrics.accuracy_score(test_labels, predicted)
        print("calculated acc: ", acc)
        
        correct = 0
        for i in range(len(predicted)):
            if test_labels[i] == predicted[i]:
                correct += 1
    
        all = len(test_labels)
        test_acc = correct/all
        print(str(correct) + " of " + str(all))
        
        if self.logger:
            self.logger.experiment.add_scalar('test_acc', test_acc)

        image = visualize(latents, 
            labels, 
            three_d = self.hparams.threeD,
            test = True,
            subtitle = self.hparams.model + " " + self.hparams.type)  

        #all_image = visualize(latents, 
        #    labels, 
        #    three_d = self.hparams.threeD,
        #    test = False,
        #    subtitle = self.hparams.model + " " + self.hparams.type)    

        if self.logger:
            self.logger.experiment.add_image('test', image, self.current_epoch)
            #self.logger.experiment.add_image('test-all', all_image, self.current_epoch)


            # reduce manually when using dp
        #if self.trainer.use_dp or self.trainer.use_ddp2:
        #    test_loss = torch.mean(test_loss)

        tqdm_dict = {'test_acc': test_acc}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'test_acc': test_acc}
        return result


