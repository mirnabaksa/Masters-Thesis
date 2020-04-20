import os
from argparse import ArgumentParser

import numpy as np
import torch

import pytorch_lightning as pl
from autoencoder_lightning import AutoencoderModel
from triplet_lightning import TripletModel

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)

def main(hparams):
    model = AutoencoderModel(hparams) if hparams.model == "auto" else TripletModel(hparams)

    # distributed backend has to be ddp!
    trainer = pl.Trainer(
        gpus = hparams.gpus,
        distributed_backend="ddp",
        max_epochs=hparams.epochs,
        precision=16 if hparams.use_16bit else 32,
        default_save_path = "logs/" + hparams.model + "-" + hparams.type
    )


    trainer.fit(model)
    trainer.test(model)


if __name__ == '__main__':

    root_dir = os.path.dirname(os.path.realpath(__file__))
    parser = ArgumentParser(add_help=False)

    parser.add_argument('--model', default="triplet", type=str, help = 'one of: auto, triplet')

    # network params
    parser.add_argument('--type', default="lstm", type=str)
    parser.add_argument('--features', default=3, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--hidden_size', default=10, type=int)
    parser.add_argument('--drop_prob', default=0.2, type=float)
    #parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--bidirectional', default = False, action='store_true')

    # training params (opt)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--optimizer_name', default='adam', type=str)
    parser.add_argument('--batch_size', default=64, type=int)

    parser.add_argument('--plot_every', default=10, type=int)

    parser.add_argument('--num_classes', default=2, type=int)
    

    # gpu args
    parser.add_argument(
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
        '--test_only',
        dest='test_only',
        action='store_true',
        help='if true runs only test'
    )

    hyperparams = parser.parse_args()
    main(hyperparams)