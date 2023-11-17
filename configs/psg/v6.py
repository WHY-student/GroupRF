find_unused_parameters=True

num_relation = 56
num_things_classes = 80
num_stuff_classes = 53
num_classes = num_things_classes + num_stuff_classes
model = dict(
    type='Mask2FormerVit3',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    panoptic_head=dict(
        type='Mask2FormerVitHead3',
        in_channels=[256, 512, 1024, 2048],  # pass to pixel_decoder inside
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=2048,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0)
    ),
    relationship_head=dict(
        type='rlnGroupToken',
        mlp_ratio=4.,
        embed_dim=256,
        embed_factors=[1, 1, 1],
        num_heads=[8, 8, 8],
        num_group_tokens=[64, 8, 0],
        num_output_groups=[64,8],
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        depths=[6, 3, 3],
        use_checkpoint=False,
        hard_assignment=True,
        feed_forward=256,
    ),
    panoptic_fusion_head=dict(
        type='MaskFormerFusionHead2',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,
        init_cfg=None),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='MaskHungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=2.0),
            mask_cost=dict(
                type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
            dice_cost=dict(
                type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
        sampler=dict(type='MaskPseudoSampler')),
    test_cfg=dict(
        panoptic_on=True,
        # For now, the dataset does not support
        # evaluating semantic segmentation metric.
        semantic_on=False,
        instance_on=True,
        # max_per_image is for instance segmentation.
        max_per_image=100,
        iou_thr=0.8,
        object_mask_thr=0.3,
        # In Mask2Former's panoptic postprocessing,
        # it will filter mask area where score is less than 0.5 .
        filter_low_score=True),
    init_cfg=None
    )

dataset_type = 'PanopticSceneGraphDataset'
ann_file = '/root/autodl-tmp/dataset/psg/psg.json'
coco_root = '/root/autodl-tmp/dataset/coco'
# dataset settings
dataset_type = 'CocoPanopticDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
image_size = (1024//2, 1024//2)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    # dict(type='PhotoMetricDistortion'),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=True,
        with_mask=True,
        with_seg=True,
        with_rela=True,
    ),
    dict(type='RandomFlip', flip_ratio=0.5),
    # large scale jittering
    # dict(
    #     type='Resize',
    #     img_scale=image_size,
    #     ratio_range=(0.1, 2.0),
    #     multiscale_mode='range',
    #     keep_ratio=True),
    dict(
        type='Resize',
        img_scale=[(1600//2, 400//2), (1600//2, 1400//2)],
        # img_scale=[(960, 540), (640, 180)],
        multiscale_mode='range',
        keep_ratio=True),
    # dict(
    #     type='RandomCrop',
    #     crop_size=image_size,
    #     crop_type='absolute',
    #     recompute_bbox=True,
    #     allow_negative_crop=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle', img_to_float=True),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip','flip_direction', 'img_norm_cfg', 'masks', 'gt_relationship'),

    ),
        
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333//2, 800//2),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

val_dataloader = dict(
    batch_size=1,
    dataset = 'PanopticSceneGraphDataset'
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file='/root/autodl-tmp/dataset/mfpsg/psg_tra.json',
        img_prefix=coco_root,
        seg_prefix=coco_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='/root/autodl-tmp/dataset/mfpsg/psg_val.json',
        img_prefix=coco_root,
        seg_prefix=coco_root,
        ins_ann_file='/root/autodl-tmp/dataset/mfpsg/psg_instance_val.json',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/root/autodl-tmp/dataset/mfpsg/psg_val.json',
        img_prefix=coco_root,
        seg_prefix=coco_root,
        ins_ann_file='/root/autodl-tmp/dataset/mfpsg/psg_instance_val.json',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm', 'pq'], classwise=True)
# evaluation = dict(
#     interval=1,
#     metric='sgdet',
#     relation_mode=True,
#     classwise=True,
#     iou_thrs=0.5,
#     detection_method='pan_seg',
# )

embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))


runner = dict(type='EpochBasedRunner', max_epochs=12)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[6, 10])
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'



# max_iters = 368750
# runner = dict(type='IterBasedRunner', max_iters=max_iters)
# learning policy
# lr_config = dict(
#     policy='step',
#     gamma=0.1,
#     by_epoch=False,
#     step=[327778, 355092],
#     warmup='linear',
#     warmup_by_epoch=False,
#     warmup_ratio=1.0,  # no warmup
#     warmup_iters=10)
# log_config = dict(
#     interval=50,
#     hooks=[
#         dict(type='TextLoggerHook', by_epoch=False),
#         dict(type='TensorboardLoggerHook', by_epoch=False)
#     ])
# custom_hooks = [dict(type='NumClassCheckHook')]
# dist_params = dict(backend='nccl')
# log_level = 'INFO'
# opencv_num_threads = 0
# mp_start_method = 'fork'
# auto_scale_lr = dict(enable=False, base_batch_size=12)
# interval = 100
# workflow = [('train', interval)]
# checkpoint_config = dict(
#     by_epoch=False, interval=interval, save_last=True, max_keep_ckpts=3)
# dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
# evaluation = dict(
#     interval=interval,
#     dynamic_intervals=dynamic_intervals,
#     metric=['PQ', 'bbox', 'segm'])


load_from = '/root/autodl-tmp/psg/mfpsg/checkpoints/mask2former_r50_lsj_8x2_50e_coco-panoptic_20220326_224516-11a44721.pth'
# resume_from = '/root/autodl-tmp/psg/mfpsg/output/v6/latest.pth'
resume_from = None
work_dir = '/root/autodl-tmp/psg/mfpsg/output/v6'
