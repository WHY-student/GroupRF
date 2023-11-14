_base_ = ['./v6.py']

model = dict(
    relationship_head=dict(
        num_group_tokens=[8, 1, 0],
        num_output_groups=[8, 1],
    )
)


load_from = '/root/autodl-tmp/psg/mfpsg/checkpoints/mask2former_r50_lsj_8x2_50e_coco-panoptic_20220326_224516-11a44721.pth'
resume_from = None
work_dir = '/root/autodl-tmp/psg/mfpsg/output/v6_token_ablation_8_1'