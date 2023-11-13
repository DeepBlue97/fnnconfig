from fnnconfig import *

from fnnaug.augment.base import AUGMENTATION_TRANSFORMS
from fnnaug.transform.base import DEFAULT_TRANSFORMS

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
        type='CocoDatasetDet',
        root_dir='/mnt/d/Share/datasets/coco', 
        ann_file='annotations/person_train.json',
        img_path='train2017',
        transform=AUGMENTATION_TRANSFORMS, 
        names_file='annotations/names_person.txt'
    ),
    val = dict(
        type='CocoDatasetDet',
        root_dir='/mnt/d/Share/datasets/coco', 
        ann_file='annotations/person_val.json',
        img_path='val2017',
        transform=DEFAULT_TRANSFORMS,
        names_file='annotations/names_person.txt'
    )
)


model = dict(

)

loss = dict(

)

schedule = dict(

)
