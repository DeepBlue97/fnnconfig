import os

learning_rate=0.0001
weight_decay=0.0005
num_classes=17

dataset_root = '/mnt/d/Share/datasets/coco/croped'
output_dir = '/mnt/d/Share/datasets/coco/croped/output_fnn'
center_weight_origin = '/home/peter/workspace/scratch/fnn/fnnmodel/src/fnnmodel/movenet/assert/center_weight_origin.npy'

train_dataloader = dict(
    type='torch.utils.data.DataLoader',
    dataset=dict(
        type='fnndataset.coco.CocoKeypointDataset',
        label_path=os.path.join(dataset_root, "train2017.json"),
        img_dir=os.path.join(dataset_root, "imgs"),
        img_size=192,
        data_aug=None
    ),
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

val_dataloader = dict(
    type='torch.utils.data.DataLoader',
    dataset=dict(
        type='fnndataset.coco.CocoKeypointDataset',
        label_path=os.path.join(dataset_root, "val2017.json"),
        img_dir=os.path.join(dataset_root, "imgs"),
        img_size=192,
        data_aug=None
    ),
    batch_size=2,
    shuffle=False,
    num_workers=2,
    pin_memory=False,
)

test_dataloader = dict(
    type='torch.utils.data.DataLoader',
    dataset=dict(
        type='fnndataset.coco.CocoKeypointTestDataset',
        # label_path="/mnt/d/Share/datasets/hall_pallet_imgs/hall_pallet_6/croped/val2017.json",
        img_dir=os.path.join(dataset_root, "imgs"),
        img_size=192,
        data_aug=None
    ),
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=False,
)

model = dict(
    type='fnnmodel.movenet.MoveNet',
    module=dict(
        type='fnnmodule.model.MoveNet',
        num_classes=num_classes,
        width_mult=1.,
        mode='train'
    ),
    loss=dict(
        type='fnnmodule.loss.MovenetLoss',
        center_weight_path=center_weight_origin,
        num_classes=num_classes,
    ),
    optimizer=dict(
        type='torch.optim.Adam',
        lr=learning_rate,
        #betas=(momentum, 0.999),
        weight_decay=weight_decay,
    ),
    device='cuda',
    # weights='/mnt/d/Share/datasets/hall_pallet_imgs/hall_pallet_6/croped/output_fnn/epoch_9.pth',
    center_weight = center_weight_origin,
    center_weight_size = [48, 48],
    num_classes=num_classes,
)

schedule = dict(
    max_epoch=10,
    log_interval=10,
    val_interval=1,
    save_interval=1,
)
