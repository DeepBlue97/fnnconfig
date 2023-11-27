import os

from fnnconfig import *

from fnnaug.augment.base import AUGMENTATION_TRANSFORMS
from fnnaug.transform.base import DEFAULT_TRANSFORMS
from fnnfunctor.loss import YOLOv3Loss


learning_rate=0.0001
weight_decay=0.0005
num_classes = 12

dataset_root = '/mnt/d/Share/datasets/coco'
output_dir = '/mnt/d/Share/datasets/coco/output_fnn_yolox'

# stride = [32, 16, 8]

train_dataloader = dict(
    type='torch.utils.data.DataLoader',
    dataset=dict(
        type='fnndataset.coco.COCOYOLOXDataset',
        data_dir=dataset_root,
        json_file="train_ADAS.json",
        name="train2017",
        img_size=(416, 416),
        preproc=dict(
            type='fnnaug.augment.yolox.TrainTransform',
            max_labels=50,
            flip_prob=0.5,
            hsv_prob=1.0,
        ),
        cache=False,
        cache_type="ram",
    ),
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)


val_dataloader = dict(
    type='torch.utils.data.DataLoader',
    dataset=dict(
        type='fnndataset.coco.COCOYOLOXDataset',
        data_dir=dataset_root,
        json_file="val_ADAS.json",
        name="val2017",
        img_size=(416, 416),
        preproc=dict(
            type='fnnaug.augment.yolox.ValTransform',
            legacy=False,
        ),
    ),
    batch_size=2,
    shuffle=False,
    num_workers=2,
    pin_memory=False,
)


model = dict(
    type='fnnmodel.yolo.YOLOX',
    module=dict(
        type='fnnmodule.model.YOLOX',
        num_classes=num_classes,
        width_mult=1.,
        mode='train'
    ),
    # loss=dict(
    #     type='fnnmodule.loss.MovenetLoss',
    #     num_classes=num_classes,
    # ),
    # optimizer=dict(
    #     type='torch.optim.Adam',
    #     lr=learning_rate,
    #     #betas=(momentum, 0.999),
    #     weight_decay=weight_decay,
    # ),
    device='cuda',
    weights='/mnt/d/Share/datasets/hall_pallet_imgs/hall_pallet_6/croped/output_fnn_noCenterWeight/epoch_300.pth',
    num_classes=num_classes,
)


schedule = dict(
    max_epoch=10
)
