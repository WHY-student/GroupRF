_base_ = ['./v0.py']

model = dict(
    type='Mask2FormerVit',
    relationship_head=dict(
        type='rlnGroupTokenMultiHead',
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
        with_token=False,
    )
)

embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            # 'panoptic_head': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0))

load_from = './checkpoints/mask2former_r50_lsj_8x2_50e_coco-panoptic_20220326_224516-11a44721.pth'
# resume_from = './output/v0_1/latest.pth'
resume_from = None
work_dir = './output/v0_prediction_token'
