import torch
import unittest
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.parsing import AttributeDict

from sparse_feature_pyramid.model import SparseFeaturePyramidAutoencoder
from sparse_feature_pyramid.data import SevenScenesDataModule
from sparse_feature_pyramid.utils import UniversalFactory


# noinspection PyTypeChecker
class TestBinarizationPointNet(unittest.TestCase):
    def setUp(self) -> None:
        torch.autograd.set_detect_anomaly(True)
        dataset_folder = "/media/mikhail/Data3T/7scenes"
        self._data_module = SevenScenesDataModule("chess", dataset_folder, 32, 4, image_size=128, center_crop=True)
        parameters = AttributeDict(
            name="SparseFeaturePyramidAutoencoder",
            optimizer=AttributeDict(),
            feature_dimensions=[8, 16, 32, 64, 128],
            size_loss_koef=1 / 1000.,
            input_dimension=3,
        )
        self._trainer = pl.Trainer(logger=TensorBoardLogger("lightning_logs"), max_epochs=1, gpus=1,
                                   limit_train_batches=1, limit_val_batches=1, limit_test_batches=1)
        factory = UniversalFactory([SparseFeaturePyramidAutoencoder])
        self._model = factory.make_from_parameters(parameters)

    def test_training(self):
        self._trainer.fit(self._model, self._data_module)

    def test_testing(self):
        self._trainer.test(self._model, self._data_module.test_dataloader())