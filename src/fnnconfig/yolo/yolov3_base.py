from fnnconfig import *

from fnnaug.augment.base import AUGMENTATION_TRANSFORMS
from fnnaug.transform.base import DEFAULT_TRANSFORMS
from fnnfunctor.loss import YOLOv3Loss

learning_rate=0.0001
weight_decay=0.0005

stride = [32, 16, 8]
yolo_layer_anchors = [
    torch.tensor([[116.,  90.],
                [156., 198.],
                [373., 326.]]) / torch.tensor(stride[0]),
    torch.tensor([[ 30.,  61.],
                [ 62.,  45.],
                [ 59., 119.]]) / torch.tensor(stride[1]),
    torch.tensor([[10., 13.],
                [16., 30.],
                [33., 23.]]) / torch.tensor(stride[2]),
]




dataset = dict(
    train = dict(
        # type='CocoDatasetDet',
        # kwargs=dict(
        #     root_dir='/mnt/d/Share/datasets/coco', 
        #     ann_file='annotations/person_train.json',
        #     img_path='train2017',
        #     transform=AUGMENTATION_TRANSFORMS, 
        #     names_file='annotations/names_person.txt'
        # )
        type='CocoDetListDataset',
        args=['/mnt/d/Share/datasets/custom_detection/train2017_yolov3.txt',],
        kwargs=dict(
            img_size=416, 
            multiscale=True, 
            transform=AUGMENTATION_TRANSFORMS,
        )
    ),
    val = dict(
        # type='CocoDatasetDet',
        # kwargs=dict(
        #     root_dir='/mnt/d/Share/datasets/coco', 
        #     ann_file='annotations/person_val.json',
        #     img_path='val2017',
        #     transform=DEFAULT_TRANSFORMS,
        #     names_file='annotations/names_person.txt'
        # )
        type='CocoDetListDataset',
        args=['/mnt/d/Share/datasets/custom_detection/val2017_yolov3.txt',],
        kwargs=dict(
            img_size=416, 
            multiscale=True,
            transform=DEFAULT_TRANSFORMS,
        )
    )
)


model = dict(
    type='YOLOv3',
    module=dict(
        type='YOLOv3',
        kwargs=dict(
            num_cls=1,
            img_w=416,
            img_h=416,
            stride=stride,
        )
    ),
    # loss = dict(
    #     type='YOLOv3Loss',
    # ),
    loss=YOLOv3Loss,
    optimizer=dict(
        type='Adam',
        kwargs=dict(
            lr=learning_rate,
            #betas=(momentum, 0.999),
            weight_decay=weight_decay
        )
    ),
    anchors=yolo_layer_anchors,
)

schedule = dict(
    max_epoch=10
)
