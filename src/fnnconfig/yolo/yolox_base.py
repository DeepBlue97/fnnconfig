# import os

# from fnnconfig import *

# from fnnaug.augment.base import AUGMENTATION_TRANSFORMS
# from fnnaug.transform.base import DEFAULT_TRANSFORMS
# from fnnfunctor.loss import YOLOv3Loss

# dataset_root = '/mnt/d/Share/datasets/coco'
# output_dir = '/mnt/d/Share/datasets/coco/output_fnn_yolox'
# train_annotation = "annotations/train_ADAS.json"
# train_img_folder = "train2017"
# val_annotation = "annotations/val_ADAS.json"
# val_img_folder = "val2017"
# num_classes = 12

dataset_root = '/mnt/d/Share/datasets/hall_pallet_imgs/hall_pallet_6/croped'
train_annotation = "annotations/train.json"
train_img_folder = "imgs"
val_annotation = "annotations/train.json"
val_img_folder = "imgs"
num_classes = 3
output_dir = '/mnt/d/Share/datasets/hall_pallet_imgs/hall_pallet_6/croped/output_fnn_yolox'
weight = output_dir + '/epoch_10.pth'

batch_size = 8
max_epoch = 100
save_interval = 1

# stride = [32, 16, 8]

model = dict(
    type='fnnmodel.yolo.YOLOX',
    module=dict(
        type='fnnmodule.model.YOLOX',
        num_classes=num_classes,
        # width_mult=1.,
        # mode='train'
    ),
    output_dir = output_dir,

    start_epoch = 0,
    max_epoch = max_epoch,
    warmup_epochs = 5,
    warmup_lr = 0.00001,

    log_interval = 10,
    save_interval = save_interval,
    weight = weight,

    weight_decay = 5e-4,

    basic_lr_per_img = 0.01 / 64.0,
    batch_size = batch_size,

    momentum = 0.9,
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
    # device='cpu',
    weights=weight,
    # num_classes=num_classes,

    dataloader = dict(
        train=dict(
            type='torch.utils.data.DataLoader',
            dataset=dict(
                type='fnndataset.coco.COCOYOLOXDataset',
                data_dir=dataset_root,
                json_file=train_annotation,
                name=train_img_folder,
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
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        ),
        val=dict(
            type='torch.utils.data.DataLoader',
            dataset=dict(
                type='fnndataset.coco.COCOYOLOXDataset',
                data_dir=dataset_root,
                json_file=val_annotation,
                name=val_img_folder,
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
        ),
    ),
    
    # schedule = dict(
    #     max_epoch=10
    # )
)
