from sparse_feature_pyramid.model import SparseFeaturePyramidAutoencoder
from sparse_feature_pyramid.data import SevenScenesDataModule
from sparse_feature_pyramid.utils import UniversalFactory
from sparse_feature_pyramid.utils.clearml_figure_reporter import ClearmlFigureReporter

from clearml import Task, Logger

import argparse
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
    factory = UniversalFactory([SparseFeaturePyramidAutoencoder])

    task = Task.init(project_name="sparse-feature-pyramid", task_name="Sparse feature pyramid on local machine",
                     auto_connect_frameworks={'matplotlib': False, 'tensorflow': True, 'tensorboard': True,
                                              'pytorch': True, 'xgboost': True, 'scikit': True, 'fastai': True,
                                              'lightgbm': True, 'hydra': True})
    data_module_parameters = {
        "batch_size": 64,
        "num_workers": 4,
        "image_size": 128,
        "scenes": ["fire"],  # , "chess", "pumpkin", "stairs", "heads", "office", "redkitchen"],
        "center_crop": True,
        "random_jitter": True,
        "random_rotation": True,
        "root_dataset_path": "/home/andrei/media/7scenes"
    }

    task.connect(data_module_parameters)
    scene = data_module_parameters["scenes"][0]
    data_module = SevenScenesDataModule(**data_module_parameters)

    model_parameters = AttributeDict(
        name="SparseFeaturePyramidAutoencoder",
        optimizer=AttributeDict(),
        feature_dimensions=[8, 16, 32, 64, 128],
        size_loss_koef=1 / 500000.,
        input_dimension=3,
        kl_loss_coefficient=0.5
    )
    task.connect(model_parameters)
    model = factory.make_from_parameters(model_parameters)
    model.set_figure_reporter(ClearmlFigureReporter())

    logger_path = os.path.join(os.path.dirname(task.cache_dir), "lightning_logs", "sparse_feature_pyramid")
    trainer_parameters = {
        "max_epochs": 100,
        "checkpoint_every_n_val_epochs": 10,
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
