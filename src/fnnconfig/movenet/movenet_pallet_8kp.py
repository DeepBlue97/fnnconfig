import os

learning_rate=0.0001
weight_decay=0.0005
num_classes=8

dataset_root = '/datasets/prj7_wood2_road_8kp_to0118_crop'
train_label = dataset_root+'/annotations/train.json'
val_label = dataset_root+'/annotations/train.json'
train_img_folder = dataset_root+'/imgs'
val_img_folder = dataset_root+'/imgs'

center_weight_origin = '/workspaces/fnn/fnnmodel/src/fnnmodel/movenet/assert/center_weight_origin.npy'
output_dir = '/output'

is_qat = False
test_size = 192
weight = ''

start_epoch = 0
max_epoch = 10
log_interval = 10
save_interval = 10

model = dict(
    type='fnnmodel.movenet.MoveNet',
    output_dir=output_dir,
    quant_dir=output_dir+'/quant',
    is_qat=is_qat,
    test_size=test_size,
    weight=weight,
    start_epoch=start_epoch,
    max_epoch=max_epoch,
    log_interval=log_interval,
    save_interval=save_interval,

    mqtt = dict(
        hostname = 'host.docker.internal',
        port = 1883,
        topic = '',
        client = '',
    ),
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
    dataloader = dict(
        train=dict(
            type='torch.utils.data.DataLoader',
            dataset=dict(
                type='fnndataset.coco.CocoKeypointDataset',
                label_path=train_label,
                img_dir=train_img_folder,
                img_size=192,
                data_aug=None
            ),
            batch_size=32,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        ),
        val=dict(
            type='torch.utils.data.DataLoader',
            dataset=dict(
                type='fnndataset.coco.CocoKeypointDataset',
                label_path=val_label,
                img_dir=val_img_folder,
                img_size=192,
                data_aug=None
            ),
            batch_size=2,
            shuffle=False,
            num_workers=2,
            pin_memory=False,
        ),
    ),
)
