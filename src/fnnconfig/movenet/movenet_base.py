learning_rate=0.0001
weight_decay=0.0005
num_classes=12


train_dataloader = dict(
    type='torch.utils.data.DataLoader',
    dataset=dict(
        type='fnndataset.coco.CocoKeypointDataset',
        label_path="/mnt/d/Share/datasets/coco/croped/train2017.json",
        img_dir='/mnt/d/Share/datasets/coco/croped/imgs',
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
        label_path="/mnt/d/Share/datasets/coco/croped/val2017.json",
        img_dir='/mnt/d/Share/datasets/coco/croped/imgs',
        img_size=192,
        data_aug=None
    ),
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
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
        center_weight_path='/home/peter/workspace/scratch/fnn/fnnmodel/src/fnnmodel/movenet/assert/center_weight_origin.npy',
        num_classes=num_classes,
    ),
    optimizer=dict(
        type='torch.optim.Adam',
        lr=learning_rate,
        #betas=(momentum, 0.999),
        weight_decay=weight_decay,
    ),
    device='cuda',
)

schedule = dict(
    max_epoch=10,
    log_interval=10,
    val_interval=1,
)
