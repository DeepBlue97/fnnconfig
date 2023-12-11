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

# input/output
dataset_root = '/mnt/d/Share/datasets/hall_pallet_imgs/hall_pallet_6/croped'
# dataset_root = '/datasets/hall_pallet_imgs/hall_pallet_6/croped'
train_annotation = "annotations/train.json"
train_img_folder = "imgs"
val_annotation = "annotations/train.json"
val_img_folder = "imgs"
num_classes = 3
output_dir = '/mnt/d/Share/datasets/hall_pallet_imgs/hall_pallet_6/croped/output_fnn_yolox'
# output_dir = '/datasets/hall_pallet_imgs/hall_pallet_6/croped/output_fnn_yolox'
weight = output_dir + '/epoch_100.pth'
# weight = ''

# schedule
batch_size = 8
max_epoch = 100
save_interval = 1

# module setup
depth = 1.0
width = 1.0
act = 'silu'
features = ("dark3", "dark4", "dark5")
in_channels = [256, 512, 1024]
depthwise = False

# is_qat = True
is_qat = False

model = dict(
    type='fnnmodel.yolo.YOLOX',
    module=dict(
        type='fnnmodule.model.YOLOX',
        # mode='train'
        backbone=dict(
            type='fnnmodule.backbone.darknet.CSPDarknet',
            dep_mul=depth,
            wid_mul=width,
            out_features=features,
            depthwise=depthwise,
            act=act,
            is_qat=is_qat,
        ),
        neck=dict(
            type='fnnmodule.neck.yolox_pafpn.YOLOPAFPN',
            depth=depth,
            width=width,
            in_features=features,
            in_channels=in_channels,
            depthwise=depthwise,
            act=act,
            is_qat=is_qat,
        ),
        head=dict(
            type='fnnmodule.head.yolox.YOLOXHead',
            num_classes=num_classes,
            width=width,
            strides=[8, 16, 32],
            in_channels=in_channels,
            act=act,
            depthwise=depthwise,
            use_l1=True,
            is_qat=is_qat,
        ),
    ),
    output_dir = output_dir,

    start_epoch = 0,
    max_epoch = max_epoch,
    warmup_epochs = 5,
    warmup_lr = 0.00001,

    # schedule
    log_interval = 10,
    save_interval = save_interval,
    weight = weight,

    # optmizer
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
                    flip_prob=0.,
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
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=False,
        ),
    ),
    
    # schedule = dict(
    #     max_epoch=10
    # )
)
