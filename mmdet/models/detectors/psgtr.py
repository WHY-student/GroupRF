# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from detectron2.utils.visualizer import VisImage, Visualizer
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from mmdet.models import DETECTORS, SingleStageDetector


class Result(object):
    """ little container class for holding the detection result
        od: object detector, rm: rel model"""
    def __init__(
        self,
        bboxes=None,  # gt bboxes / OD: det bboxes
        dists=None,  # OD: predicted dists
        labels=None,  # gt labels / OD: det labels
        masks=None,  # gt masks  / OD: predicted masks
        formatted_masks=None,  # OD: Transform the masks for object detection evaluation
        points=None,  # gt points / OD: predicted points
        rels=None,  # gt rel triplets / OD: sampled triplets (training) with target rel labels
        key_rels=None,  # gt key rels
        relmaps=None,  # gt relmaps
        refine_bboxes=None,  # RM: refined object bboxes (score is changed)
        formatted_bboxes=None,  # OD: Transform the refine_bboxes for object detection evaluation
        refine_scores=None,  # RM: refined object scores (before softmax)
        refine_dists=None,  # RM: refined object dists (after softmax)
        refine_labels=None,  # RM: refined object labels
        target_labels=None,  # RM: assigned object labels for training the relation module.
        rel_scores=None,  # RM: predicted relation scores (before softmax)
        rel_dists=None,  # RM: predicted relation prob (after softmax)
        triplet_scores=None,  # RM: predicted triplet scores (the multiplication of sub-obj-rel scores)
        ranking_scores=None,  # RM: predicted ranking scores for rank the triplet
        rel_pair_idxes=None,  # gt rel_pair_idxes / RM: training/testing sampled rel_pair_idxes
        rel_labels=None,  # gt rel_labels / RM: predicted rel labels
        target_rel_labels=None,  # RM: assigned target rel labels
        target_key_rel_labels=None,  # RM: assigned target key rel labels
        saliency_maps=None,  # SAL: predicted or gt saliency map
        attrs=None,  # gt attr
        rel_cap_inputs=None,  # gt relational caption inputs
        rel_cap_targets=None,  # gt relational caption targets
        rel_ipts=None,  # gt relational importance scores
        tgt_rel_cap_inputs=None,  # RM: assigned target relational caption inputs
        tgt_rel_cap_targets=None,  # RM: assigned target relational caption targets
        tgt_rel_ipts=None,  # RM: assigned target relational importance scores
        rel_cap_scores=None,  # RM: predicted relational caption scores
        rel_cap_seqs=None,  # RM: predicted relational seqs
        rel_cap_sents=None,  # RM: predicted relational decoded captions
        rel_ipt_scores=None,  # RM: predicted relational caption ipt scores
        cap_inputs=None,
        cap_targets=None,
        cap_scores=None,
        cap_scores_from_triplet=None,
        alphas=None,
        rel_distribution=None,
        obj_distribution=None,
        word_obj_distribution=None,
        cap_seqs=None,
        cap_sents=None,
        img_shape=None,
        scenes=None,  # gt scene labels
        target_scenes=None,  # target_scene labels
        add_losses=None,  # For Recording the loss except for final object loss and rel loss, e.g.,
        # use in causal head or VCTree, for recording auxiliary loss
        head_spec_losses=None,  # For method-specific loss
        pan_results=None,
    ):
        self.__dict__.update(locals())
        del self.__dict__['self']

    def is_none(self):
        return all(
            [v is None for k, v in self.__dict__.items() if k != 'self'])

    # HACK: To turn this object into an iterable
    def __len__(self):
        return 1

    # HACK:
    def __getitem__(self, i):
        return self

    # HACK:
    def __iter__(self):
        yield self



def triplet2Result(triplets, use_mask, eval_pan_rels=True):
    if use_mask:
        bboxes, labels, rel_pairs, masks, pan_rel_pairs, pan_seg, complete_r_labels, complete_r_dists, \
            r_labels, r_dists, pan_masks, rels, pan_labels \
            = triplets
        if isinstance(bboxes, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            bboxes = bboxes.detach().cpu().numpy()
            rel_pairs = rel_pairs.detach().cpu().numpy()
            complete_r_labels = complete_r_labels.detach().cpu().numpy()
            complete_r_dists = complete_r_dists.detach().cpu().numpy()
            r_labels = r_labels.detach().cpu().numpy()
            r_dists = r_dists.detach().cpu().numpy()
        if isinstance(pan_seg, torch.Tensor):
            pan_seg = pan_seg.detach().cpu().numpy()
            pan_rel_pairs = pan_rel_pairs.detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            pan_masks = pan_masks.detach().cpu().numpy()
            rels = rels.detach().cpu().numpy()
            pan_labels = pan_labels.detach().cpu().numpy()
        if eval_pan_rels:
            return Result(refine_bboxes=bboxes,
                        labels=pan_labels+1,
                        formatted_masks=dict(pan_results=pan_seg),
                        rel_pair_idxes=pan_rel_pairs,# elif not pan: rel_pairs,
                        rel_dists=r_dists,
                        rel_labels=r_labels,
                        pan_results=pan_seg,
                        masks=pan_masks,
                        rels=rels)
        else:
            return Result(refine_bboxes=bboxes,
                        labels=labels,
                        formatted_masks=dict(pan_results=pan_seg),
                        rel_pair_idxes=rel_pairs,
                        rel_dists=complete_r_dists,
                        rel_labels=complete_r_labels,
                        pan_results=pan_seg,
                        masks=masks)
    else:
        bboxes, labels, rel_pairs, r_labels, r_dists = triplets
        labels = labels.detach().cpu().numpy()
        bboxes = bboxes.detach().cpu().numpy()
        rel_pairs = rel_pairs.detach().cpu().numpy()
        r_labels = r_labels.detach().cpu().numpy()
        r_dists = r_dists.detach().cpu().numpy()
        return Result(
            refine_bboxes=bboxes,
            labels=labels,
            formatted_masks=dict(pan_results=None),
            rel_pair_idxes=rel_pairs,
            rel_dists=r_dists,
            rel_labels=r_labels,
            pan_results=None,
        )


@DETECTORS.register_module()
class PSGTr(SingleStageDetector):
    def __init__(self,
                 backbone,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(PSGTr, self).__init__(backbone, None, bbox_head, train_cfg,
                                    test_cfg, pretrained, init_cfg)
        self.CLASSES = self.bbox_head.object_classes
        self.PREDICATES = self.bbox_head.predicate_classes
        self.num_classes = self.bbox_head.num_classes

    # over-write `forward_dummy` because:
    # the forward of bbox_head requires img_metas
    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        warnings.warn('Warning! MultiheadAttention in DETR does not '
                      'support flops computation! Do not use the '
                      'results in your papers!')

        batch_size, _, height, width = img.shape
        dummy_img_metas = [
            dict(batch_input_shape=(height, width),
                 img_shape=(height, width, 3)) for _ in range(batch_size)
        ]
        x = self.extract_feat(img)
        outs = self.bbox_head(x, dummy_img_metas)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_rels,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_bboxes_ignore=None):
        super(SingleStageDetector, self).forward_train(img, img_metas)

        x = self.extract_feat(img)
        if self.bbox_head.use_mask:
            BS, C, H, W = img.shape
            new_gt_masks = []
            for each in gt_masks:
                mask = torch.tensor(each.to_ndarray(), device=x[0].device)
                _, h, w = mask.shape
                padding = (0, W - w, 0, H - h)
                mask = F.interpolate(F.pad(mask, padding).unsqueeze(1),
                                     size=(H // 2, W // 2),
                                     mode='nearest').squeeze(1)
                # mask = F.pad(mask, padding)
                new_gt_masks.append(mask)

            gt_masks = new_gt_masks

        losses = self.bbox_head.forward_train(x, img_metas, gt_rels, gt_bboxes,
                                              gt_labels, gt_masks,
                                              gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):

        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(feat,
                                                  img_metas,
                                                  rescale=rescale)
        sg_results = [
            triplet2Result(triplets, self.bbox_head.use_mask)
            for triplets in results_list
        ]
        # print(time.time() - s)
        return sg_results
