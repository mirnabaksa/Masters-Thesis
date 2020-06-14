import os
from argparse import ArgumentParser

import numpy as np
import torch
import random

import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from autoencoder_lightning import AutoencoderModel
from triplet_lightning import TripletModel

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def main(hparams):
    print(hparams)
    ckpt = "./longer/triplet-lstm/6-classes/version_16/checkpoints/epoch=31.ckpt"

    save_dir = "test" if hparams.no_log else  "longer/" + hparams.model + "-" + hparams.type + "/"
    model = TripletModel.load_from_checkpoint(ckpt)

    logger = TestTubeLogger(save_dir = save_dir, name = str(hparams.num_classes) + "-classes")
    trainer = pl.Trainer(
        resume_from_checkpoint=ckpt,
        logger = logger,
        gpus=1,
        #distributed_backend="ddp",
        max_epochs=16
    )
    #trainer.fit(model)
    trainer.test(model)
  
    

if __name__ == '__main__':


    root_dir = os.path.dirname(os.path.realpath(__file__))
    parser = ArgumentParser(add_help=False)

    parser.add_argument('--model', default="triplet", type=str, help = 'one of: auto, triplet')

    # network params
    parser.add_argument('--type', default="lstm", type=str)
    parser.add_argument('--features', default=4, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--hidden_size', default=10, type=int)
    parser.add_argument('--drop_prob', default=0.0, type=float)
    parser.add_argument('--bidirectional', default = False, action='store_true')
    parser.add_argument('--min_max', default = False, action='store_true')
    parser.add_argument('--filters', default=16, type=int)
    parser.add_argument('--margin', default=0.2, type=float)

    # training params (opt)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--optimizer_name', default='adam', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--plot_every', default=10, type=int)
    parser.add_argument('--threeD', default = False, action='store_true')
    parser.add_argument('--num_classes', default=2, type=int)
    

    # gpu args
    '''parser.add_argument(
        '--gpus',
        type=int,
        default=2,
        help='how many gpus'
    )

    parser.add_argument(
        '--use_16bit',
        dest='use_16bit',
        action='store_true',
        help='if true uses 16 bit precision'
    )

    parser.add_argument(
        '--distributed_backend',
        dest='distributed_backend',
        default="ddp",
        type=str,
        help='dp, ddp'
    )'''

    parser.add_argument(
        '--no_log',
        dest='no_log',
        action='store_true',
        help='turn off logging'
    )

    hyperparams = parser.parse_args()
    main(hyperparams)