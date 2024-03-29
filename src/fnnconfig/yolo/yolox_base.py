import os


# 读取环境变量
FNN_MODE = os.environ.get('FNN_MODE')
assert FNN_MODE is not None, f'FNN_MODE is {FNN_MODE}'

# input/output
datasets = '/mnt/d/Share/datasets'
datasets = '/datasets'
dataset_root = datasets+'/hall_pallet_6_feet'
train_annotation = "annotations/train.json"
train_img_folder = "imgs"
val_annotation = "annotations/train.json"
val_img_folder = "imgs"
num_classes = 3
output_dir = datasets+'/hall_pallet_6_feet/output_fnn_yolox'

# Set weight file path
# weight = output_dir + '/epoch_100.pth'
weight = ''
if FNN_MODE == 'deploy':
    weight = '/weight.pth'
    print(f'deploy mode, change weight to {weight}')
if weight:
    assert os.path.exists(weight), f'not exists weights: {weight}'

# schedule
batch_size = 8
max_epoch = 100
save_interval = 10

# module setup
# depth = 1.0
# width = 1.0

MODEL_SCALE = 'nano'

img_size=(416, 416)
act = 'relu'
features = ("dark3", "dark4", "dark5")
in_channels = [256, 512, 1024]
depthwise = False

# is_qat = True
is_qat = False

device = 'cuda:1'
if FNN_MODE == 'deploy':
    device = 'cpu'

if MODEL_SCALE == 'nano':
    # nano
    depth = 0.33
    width = 0.25
    depthwise = True
    

model = dict(
    type='fnnmodel.yolo.YOLOX',
    module=dict(
        type='fnnmodule.model.YOLOX',
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
            device=device,
        ),
        is_qat=is_qat,
    ),
    output_dir = output_dir,
    quant_dir = output_dir+'/quant',

    start_epoch = 0,
    max_epoch = max_epoch,
    warmup_epochs = 5,
    warmup_lr = 0.00001,

    img_size=img_size,

    # schedule
    log_interval = 10,
    save_interval = save_interval,
    weight = weight,

    # MQTT参数
    mqtt = dict(
        hostname = 'host.docker.internal',
        port = 1883,
        topic = '',
        client = '',
    ),

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
    device=device,

    dataloader = dict(
        train=dict(
            type='torch.utils.data.DataLoader',
            dataset=dict(
                type='fnndataset.coco.COCOYOLOXDataset',
                data_dir=dataset_root,
                json_file=train_annotation,
                name=train_img_folder,
                img_size=img_size,
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
                img_size=img_size,
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

    evaluator=dict(
        type='fnnmodel.evaluator.coco_evaluator.COCOEvaluator',
        dataloader='val',
        img_size=img_size,
        confthre=0.6,
        nmsthre=0.5,
        num_classes=num_classes,
        testdev=False,
        per_class_AP=True,
        per_class_AR=True,
        show_folder=output_dir+'/show',
    )
)
