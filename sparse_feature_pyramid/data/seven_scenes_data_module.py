import pytorch_lightning as pl
import torch.utils.data
import albumentations
import albumentations.pytorch
import cv2
from .seven_scenes import SevenScenes


class SevenScenesDataModule(pl.LightningDataModule):
    def __init__(self, scene, root_dataset_path, batch_size=128, num_workers=4, seed=0,
                 image_size=256, random_jitter=False, random_rotation=False, center_crop=False):
        super().__init__()
        torch.random.manual_seed(seed)
        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]
        self._image_size = image_size
        self._image_shape = (image_size, int(image_size * 640 // 480))
        self._random_jitter = random_jitter
        self._random_rotation = random_rotation
        self._center_crop = center_crop
        self._batch_size = batch_size
        self._num_workers = num_workers

        train_image_transform = self.make_train_image_transform()
        test_image_transform = self.make_test_image_transform()
        self._train_dataset = SevenScenes(scene, root_dataset_path, train=True, image_transform=train_image_transform)
        self._test_dataset = SevenScenes(scene, root_dataset_path, train=False, image_transform=test_image_transform)
        print(f"[ToyDataModule] - train subset size {len(self._train_dataset)}")
        print(f"[ToyDataModule] - validation dataset size {len(self._test_dataset)}")

    def make_train_image_transform(self):
        transform_list = [albumentations.Resize(self._image_shape[0], self._image_shape[1],
                                                interpolation=cv2.INTER_AREA)]
        if self._center_crop:
            transform_list.append(albumentations.RandomCrop(self._image_size, self._image_size))
        if self._random_jitter:
            transform_list.append(albumentations.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2))
        transform_list.append(albumentations.Normalize(mean=self._mean, std=self._std))
        if self._random_rotation:
            transform_list.append(albumentations.Affine(rotate=(-180, 180), translate_percent=(0.3, 0.3),
                                                        scale=(0.5, 2.), p=0.8))
        transform_list.append(albumentations.pytorch.ToTensorV2())
        return albumentations.Compose(transform_list)

    def make_test_image_transform(self):
        transform_list = [albumentations.Resize(self._image_shape[0], self._image_shape[1],
                                                interpolation=cv2.INTER_AREA)]
        if self._center_crop:
            transform_list.append(albumentations.CenterCrop(self._image_size, self._image_size))
        transform_list.append(albumentations.Normalize(mean=self._mean, std=self._std))
        transform_list.append(albumentations.pytorch.ToTensorV2())
        return albumentations.Compose(transform_list)

    def train_dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(self._train_dataset, self._batch_size, shuffle=True, pin_memory=False,
                                           num_workers=self._num_workers)

    def val_dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(self._test_dataset, self._batch_size, shuffle=False, pin_memory=False,
                                           num_workers=self._num_workers)

    def test_dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(self._test_dataset, self._batch_size, shuffle=False, pin_memory=False,
                                           num_workers=self._num_workers)
