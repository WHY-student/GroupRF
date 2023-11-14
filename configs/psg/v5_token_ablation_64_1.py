_base_ = ['./v5.py']

model = dict(
    type='Mask2FormerVit2',
    num_group_tokens=[64, 1, 0],
    num_output_groups=[64, 1],
)

runner = dict(type='EpochBasedRunner', max_epochs=12)

load_from = '/root/autodl-tmp/psg/mfpsg/checkpoints/mask2former_r50_lsj_8x2_50e_coco-panoptic_20220326_224516-11a44721.pth'
resume_from = '/root/autodl-tmp/psg/mfpsg/output/v5_token_ablation_64_1/latest.pth'
# resume_from = None
work_dir = '/root/autodl-tmp/psg/mfpsg/output/v5_token_ablation_64_1'