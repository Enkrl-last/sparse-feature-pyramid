import unittest
import os
from sparse_feature_pyramid.data import SevenScenesDataModule
import torch


class TestSevenScenesDataModule(unittest.TestCase):
    def setUp(self) -> None:
        dataset_folder = "/media/mikhail/Data3T/7scenes"
        self._data_module = SevenScenesDataModule("chess", dataset_folder, 32, 4, image_size=128, center_crop=True)

    def test_load(self):
        self.assertEqual(len(self._data_module._train_dataset), 4000)
        self.assertEqual(len(self._data_module._test_dataset), 2000)
        batches = self._data_module.train_dataloader()
        for batch in batches:
            self.assertEqual(batch["image"].shape, torch.Size([32, 3, 128, 128]))
            self.assertEqual(batch["position"].shape, torch.Size([32, 4, 4]))
            break
