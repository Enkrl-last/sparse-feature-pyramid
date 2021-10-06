from torch.utils import data
from .seven_scenes import SevenScenes


class ConcatSevenScenes(data.ConcatDataset):
    def __init__(self, scenes, dataset_path, train, image_transform):
        datasets = [SevenScenes(scene, dataset_path, train, image_transform) for scene in scenes]
        super().__init__(datasets)
