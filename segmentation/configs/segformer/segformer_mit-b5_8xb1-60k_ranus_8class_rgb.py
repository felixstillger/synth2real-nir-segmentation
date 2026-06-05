_base_ = [
    '../../mmsegmentation/configs/_base_/models/segformer_mit-b0.py',
    '../../_base_/datasets/ranus_8class.py',
    '../../mmsegmentation/configs/_base_/default_runtime.py', '../../mmsegmentation/configs/_base_/schedules/schedule_40k.py'
]
crop_size = (1024, 1024)
data_preprocessor = dict(size=crop_size)
checkpoint = '../../pretrained/mit_b5.pth' #'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 6, 40, 3]),
        # test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))
    decode_head=dict(in_channels=[64, 128, 320, 512],
                     num_classes=8))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.) 
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=60000,
        by_epoch=False,
    )
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=60000, val_interval=5000)

work_dir = './work_dirs/trainings/ranus/segformer_mit-b5_8xb1-60k_ranus_8class_rgb'

custom_imports = dict(imports=['mmseg_custom.datasets', 'mmseg_custom.evaluation'], allow_failed_imports=False)
