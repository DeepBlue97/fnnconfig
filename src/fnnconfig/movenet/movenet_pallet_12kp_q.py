import os

from .movenet_pallet_12kp import *


model = dict(
    type='fnnmodel.movenet.MoveNet',
    module=dict(
        type='fnnmodule.model.MoveNetQ',
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
    # weights='/mnt/d/Share/datasets/hall_pallet_imgs/hall_pallet_6/croped/output_fnn_noCenterWeight/epoch_300.pth',
    weights=os.path.join(output_dir, 'epoch_290.pth'),
    center_weight = center_weight_origin,
    center_weight_size = [48, 48],
    num_classes=num_classes,
    is_qat=True,
    test_size=[192,192],
)
