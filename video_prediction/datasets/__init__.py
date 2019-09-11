from .base_dataset import BaseVideoDataset
from .slide_dataset import SlideDataset
from .wall_dataset import WallDataset
from .collision_dataset import CollisionDataset


def get_dataset_class(dataset):
    dataset_mappings = {
        'slide': 'SlideDataset',
        'wall': 'WallDataset',
        'collision': 'CollisionDataset'
    }
    dataset_class = dataset_mappings.get(dataset, dataset)
    dataset_class = globals().get(dataset_class)
    if dataset_class is None or not issubclass(dataset_class, BaseVideoDataset):
        raise ValueError('Invalid dataset %s' % dataset)
    return dataset_class
