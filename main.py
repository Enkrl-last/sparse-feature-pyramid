from sparse_feature_pyramid.model import SparseFeaturePyramidAutoencoder
from sparse_feature_pyramid.data import SevenScenesDataModule
from sparse_feature_pyramid.utils import UniversalFactory
from sparse_feature_pyramid.utils.clearml_figure_reporter import ClearmlFigureReporter

from clearml import Task, Logger

from argparse import ArgumentParser
import sys
import pytorch_lightning as pl
import os
from pytorch_lightning.utilities.parsing import AttributeDict
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import albumentations
import torchvision.transforms as transforms
import torchvision
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

if __name__ == "__main__":
    parser = ArgumentParser(description='Launch experiment sparse feature pyramid')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-image_size', type=int, default=128, help='image size')
    parser.add_argument('-scenes', default=1, type=int, help='number_of_scenes (1 - fire only, 7 - all scenes)')
    parser.add_argument('-root_dataset_path', default="/home/ikalinov/sparce_feature_pyramid/dataset", help='path to dataset (from root)')
    parser.add_argument('-feature_dimensions', default=[8, 16, 32, 64, 128], nargs="+", type=int, help='array for each layer')
    parser.add_argument('-max_epochs', type=int, default=100, help='Number of epoch')
    parser.add_argument('-checkpoint_every_n_val_epochs', type=int, default=10, help='Chekpoint each N epoch')
    args = parser.parse_args()

    if args.scenes == 1:
        scenes=['fire']
    elif args.scenes == 7:
        scenes=["fire", "chess", "pumpkin", "stairs", "heads", "office", "redkitchen"]
    else:
        print('Unexpected scenes, choose 1 or 7')
        sys.exit() 

    factory = UniversalFactory([SparseFeaturePyramidAutoencoder])
    task = Task.init(project_name="sparse-feature-pyramid", task_name="Sparse feature pyramid on server",
                     auto_connect_frameworks={'matplotlib': False, 'tensorflow': True, 'tensorboard': True,
                                              'pytorch': True, 'xgboost': True, 'scikit': True, 'fastai': True,
                                              'lightgbm': True, 'hydra': True})
    data_module_parameters = {
        "batch_size": args.batch_size,
        "num_workers": 4,
        "image_size": args.image_size,
        "scenes": scenes,
        "center_crop": True,
        "random_jitter": True,
        "random_rotation": True,
        "root_dataset_path": args.root_dataset_path
    }

    task.connect(data_module_parameters)
    scene = data_module_parameters["scenes"][0]
    data_module = SevenScenesDataModule(**data_module_parameters)

    model_parameters = AttributeDict(
        name="SparseFeaturePyramidAutoencoder",
        optimizer=AttributeDict(),
        feature_dimensions=args.feature_dimensions,
        size_loss_koef=(args.image_size*args.image_size*3) * (1 / 500000.),
        input_dimension=3
    )
    task.connect(model_parameters)
    model = factory.make_from_parameters(model_parameters)
    model.set_figure_reporter(ClearmlFigureReporter())

    logger_path = os.path.join(os.path.dirname(task.cache_dir), "lightning_logs", "sparse_feature_pyramid")
    trainer_parameters = {
        "max_epochs": args.max_epochs,
        "checkpoint_every_n_val_epochs": args.checkpoint_every_n_val_epochs,
        "gpus": 1,
        "check_val_every_n_epoch": 2
    }
    task.connect(trainer_parameters)
    model_checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                    every_n_val_epochs=trainer_parameters[
                                                        "checkpoint_every_n_val_epochs"])
    trainer = factory.kwargs_function(pl.Trainer)(
        logger=TensorBoardLogger(logger_path, name=scene),
        callbacks=[model_checkpoint],
        **trainer_parameters
    )

    trainer.fit(model, data_module)

    task.close()
