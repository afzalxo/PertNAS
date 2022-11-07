from typing import List

import torch
import torchvision

from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze

def get_ffcv_loaders(train_path, bsize_train, bsize_val, device):
    # Note that statistics are wrt to uin8 range, [0,255].
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]

    loaders = {}
    for name in ['train', 'test']:
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(torch.device(f'cuda:{device}'), non_blocking=True), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
        
        # Add image transforms and normalization
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=2),
                Cutout(8, tuple(map(int, CIFAR_MEAN))), # Note Cutout is done before normalization.
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice(torch.device(f'cuda:{device}'), non_blocking=True),
            ToTorchImage(),
            Convert(torch.float16),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        bsize = bsize_train if name == 'train' else bsize_val
        # Create loaders
        loaders[name] = Loader(train_path+f'/cifar_{name}.beton',
                                batch_size=bsize,
                                num_workers=8,
                                order=OrderOption.RANDOM,
                                drop_last=(name == 'train'),
                                pipelines={'image': image_pipeline,
                                           'label': label_pipeline})
    return loaders['train'], loaders['test']
