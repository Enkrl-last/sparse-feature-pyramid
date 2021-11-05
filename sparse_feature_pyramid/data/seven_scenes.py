import os
import os.path as osp

import numpy as np
from torch.utils import data

from .utils import load_image


class SevenScenes(data.Dataset):
    def __init__(self, scene, dataset_path, train, image_transform):
        """
        :param scene: scene name ['chess', 'pumpkin', ...]
        :param dataset_path: root 7scenes data directory.
        :param train: if True, return the training images. If False, returns the testing images
        :param image_transform: transform to apply to the images
        """
        self._image_transform = image_transform
        base_directory = osp.join(osp.expanduser(dataset_path), scene)

        # decide which sequences to use
        if train:
            split_file = osp.join(base_directory, 'TrainSplit.txt')
        else:
            split_file = osp.join(base_directory, 'TestSplit.txt')
        with open(split_file, 'r') as fd:
            sequences = [int(x.split('sequence')[-1]) for x in fd if not x.startswith('#')]

        self._color_images = []
        self._positions = []

        try:
            import google.colab
            in_colab = True
        except ImportError:
            in_colab = False

        for sequence in sequences:
            sequence_directory = osp.join(base_directory, 'seq-{:02d}'.format(sequence))
            if in_colab:
                extract_path = osp.join("/content/dataset/7scenes/", scene)
                sequence_directory = osp.join(extract_path, 'seq-{:02d}'.format(sequence))
                self.unzip_sequence(base_directory, extract_path, sequence)

            elif not osp.isdir(sequence_directory):
                print("[SevenScenes] - don't find sequence directory for sequence {} in directory {}".format(sequence,
                      sequence_directory))
                print("[SevenScenes] - trying to unzip")
                self.unzip_sequence(base_directory, sequence_directory, sequence)

            pose_filenames = [x for x in os.listdir(osp.join(sequence_directory)) if x.find('pose') >= 0]
            frame_indexes = np.arange(len(pose_filenames), dtype=np.int64)
            positions = [np.loadtxt(osp.join(sequence_directory, 'frame-{:06d}.pose.txt'.
                                             format(i))) for i in frame_indexes]
            color_images = [osp.join(sequence_directory, 'frame-{:06d}.color.png'.format(i))
                            for i in frame_indexes]
            self._color_images.extend(color_images)
            self._positions.extend(positions)
        self._positions = np.array(self._positions)

    @staticmethod
    def unzip_sequence(base_directory, base_sequence_path, sequence):
        import zipfile
        zip_file = osp.join(base_directory, 'seq-{:02d}.zip'.format(sequence))
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            print("[SevenScenes] - unziping file {}".format(zip_file))
            try:
                zip_ref.extractall(base_sequence_path)
            except zipfile.BadZipFile:
                print("[WARN][SevenScenes] - bad zip file, but I will continue")

    def __getitem__(self, index):
        position = self._positions[index]
        image = np.array(load_image(self._color_images[index]))
        mask = np.ones_like(image)[:, :, 0]
        transformed_data = self._image_transform(image=image, mask=mask)
        return {"image": transformed_data["image"],
                "mask": transformed_data["mask"],
                "position": position}

    def __len__(self):
        return self._positions.shape[0]
