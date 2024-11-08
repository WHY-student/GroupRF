# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from mmdet.core.evaluation.panoptic_utils import INSTANCE_OFFSET
from mmdet.core.mask import mask2bbox
from mmdet.models.builder import HEADS
from .base_panoptic_fusion_head import BasePanopticFusionHead

from IPython import embed
import cv2
import numpy as np

@HEADS.register_module()
class MaskFormerFusionHead(BasePanopticFusionHead):

    def __init__(self,
                 num_things_classes=80,
                 num_stuff_classes=53,
                 test_cfg=None,
                 loss_panoptic=None,
                 init_cfg=None,
                 **kwargs):
        super().__init__(num_things_classes, num_stuff_classes, test_cfg,
                         loss_panoptic, init_cfg, **kwargs)

    def forward_train(self, **kwargs):
        """MaskFormerFusionHead has no training loss."""
        return dict()

    def panoptic_postprocess(self, mask_cls, mask_pred):
        """Panoptic segmengation inference.

        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_queries, cls_out_channels) for a image.
                Note `cls_out_channels` should includes
                background.
            mask_pred (Tensor): Mask outputs of shape
                (num_queries, h, w) for a image.

        Returns:
            Tensor: Panoptic segment result of shape \
                (h, w), each element in Tensor means: \
                ``segment_id = _cls + instance_id * INSTANCE_OFFSET``.
        """
        object_mask_thr = self.test_cfg.get('object_mask_thr', 0.8)
        iou_thr = self.test_cfg.get('iou_thr', 0.8)
        filter_low_score = self.test_cfg.get('filter_low_score', False)

        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.num_classes) & (scores > object_mask_thr)
        # scores (n,1) , labels (n,1)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        # cur_masks (n, h, w)
        cur_masks = mask_pred[keep]

        # 计算出所有保留下来的mask的分数和mask和标签
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.full((h, w),
                                  self.num_classes,
                                  dtype=torch.int32,
                                  device=cur_masks.device)
        entityid_list = []
        entity_score_list = []

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            pass
        else:
            mode = 'raw'
            if mode == 'area':
                areas = (cur_masks>=0.5).sum(dim=[1,2])
                area_tensor, idx_tensor = torch.sort(areas)
                instance_id = 1
                for area, idx_cur in zip(area_tensor.flip(dims=[0]), idx_tensor.flip(dims=[0])):
                    if area <= 0:
                        continue
                    pred_class = cur_classes[idx_cur].to(torch.long)
                    if pred_class == 133:
                        continue
                    pred_mask = cur_masks[idx_cur] >= 0.5
                    pred_score = cur_prob_masks[idx_cur][pred_mask].mean()

                    isthing = pred_class < self.num_things_classes
                    if not isthing:
                        panoptic_seg[pred_mask] = panoptic_seg[pred_mask] * 0 + pred_class
                        eid = pred_class
                    else:
                        panoptic_seg[pred_mask] = panoptic_seg[pred_mask] * 0 + (pred_class + instance_id * INSTANCE_OFFSET)
                        eid = pred_class + instance_id * INSTANCE_OFFSET
                        instance_id += 1
    
                    entityid_list.append(eid)
                    entity_score_list.append(pred_score)
            elif mode == 'raw':
                # cur_mask_ids = cur_prob_masks.argmax(0)
                cur_mask_score, cur_mask_ids = cur_prob_masks.max(dim=0)
                instance_id = 1
                for k in range(cur_classes.shape[0]):
                    # pred_class = int(cur_classes[k].item())
                    pred_class = cur_classes[k].to(torch.long)
                    isthing = pred_class < self.num_things_classes
                    mask = cur_mask_ids == k
                    mask_area = mask.sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()

                    score = cur_mask_score[mask].mean()

                    if filter_low_score:
                        mask = mask & (cur_masks[k] >= 0.5)
                        # mask_area = mask.sum().item()

                    if mask_area > 0 and original_area > 0:
                        if mask_area / original_area < iou_thr:
                            continue
                        if not isthing:
                            # different stuff regions of same class will be
                            # merged here, and stuff share the instance_id 0.
                            panoptic_seg[mask] = panoptic_seg[mask] * 0 + pred_class
                            # entityid_list.append(pred_class)
                            # entity_score_list.append(score)
                        else:
                            panoptic_seg[mask] = panoptic_seg[mask] * 0 + (pred_class + instance_id * INSTANCE_OFFSET)
                            # entityid_list.append(pred_class + instance_id * INSTANCE_OFFSET)
                            # entity_score_list.append(score)
                            instance_id += 1
                
                for eid in torch.unique(panoptic_seg):
                    if eid == 133:
                        continue
                    mask = panoptic_seg == eid
                    score = cur_mask_score[mask].mean()
                    entityid_list.append(eid)
                    entity_score_list.append(score)

        # print([eid.item() for eid in entityid_list])
        # return panoptic_seg
        return panoptic_seg, entityid_list, entity_score_list

    def semantic_postprocess(self, mask_cls, mask_pred):
        """Semantic segmengation postprocess.

        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_queries, cls_out_channels) for a image.
                Note `cls_out_channels` should includes
                background.
            mask_pred (Tensor): Mask outputs of shape
                (num_queries, h, w) for a image.

        Returns:
            Tensor: Semantic segment result of shape \
                (cls_out_channels, h, w).
        """
        # TODO add semantic segmentation result
        raise NotImplementedError

    def instance_postprocess(self, mask_cls, mask_pred):
        """Instance segmengation postprocess.

        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_queries, cls_out_channels) for a image.
                Note `cls_out_channels` should includes
                background.
            mask_pred (Tensor): Mask outputs of shape
                (num_queries, h, w) for a image.

        Returns:
            tuple[Tensor]: Instance segmentation results.

            - labels_per_image (Tensor): Predicted labels,\
                shape (n, ).
            - bboxes (Tensor): Bboxes and scores with shape (n, 5) of \
                positive region in binary mask, the last column is scores.
            - mask_pred_binary (Tensor): Instance masks of \
                shape (n, h, w).
        """
        max_per_image = self.test_cfg.get('max_per_image', 100)
        num_queries = mask_cls.shape[0]
        # shape (num_queries, num_class)
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        # shape (num_queries * num_class, )
        labels = torch.arange(self.num_classes, device=mask_cls.device).\
            unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
        scores_per_image, top_indices = scores.flatten(0, 1).topk(
            max_per_image, sorted=False)
        labels_per_image = labels[top_indices]

        query_indices = top_indices // self.num_classes
        mask_pred = mask_pred[query_indices]

        # extract things
        is_thing = labels_per_image < self.num_things_classes
        scores_per_image = scores_per_image[is_thing]
        labels_per_image = labels_per_image[is_thing]
        mask_pred = mask_pred[is_thing]

        mask_pred_binary = (mask_pred > 0).float()
        mask_scores_per_image = (mask_pred.sigmoid() *
                                 mask_pred_binary).flatten(1).sum(1) / (
                                     mask_pred_binary.flatten(1).sum(1) + 1e-6)
        det_scores = scores_per_image * mask_scores_per_image
        mask_pred_binary = mask_pred_binary.bool()
        bboxes = mask2bbox(mask_pred_binary)
        bboxes = torch.cat([bboxes, det_scores[:, None]], dim=-1)

        return labels_per_image, bboxes, mask_pred_binary

    def simple_test(self,
                    mask_cls_results,
                    mask_pred_results,
                    img_metas,
                    rescale=False,
                    **kwargs):
        """Test segment without test-time aumengtation.

        Only the output of last decoder layers was used.

        Args:
            mask_cls_results (Tensor): Mask classification logits,
                shape (batch_size, num_queries, cls_out_channels).
                Note `cls_out_channels` should includes background.
            mask_pred_results (Tensor): Mask logits, shape
                (batch_size, num_queries, h, w).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): If True, return boxes in
                original image space. Default False.

        Returns:
            list[dict[str, Tensor | tuple[Tensor]]]: Semantic segmentation \
                results and panoptic segmentation results for each \
                image.

            .. code-block:: none

                [
                    {
                        'pan_results': Tensor, # shape = [h, w]
                        'ins_results': tuple[Tensor],
                        # semantic segmentation results are not supported yet
                        'sem_results': Tensor
                    },
                    ...
                ]
        """
        panoptic_on = self.test_cfg.get('panoptic_on', True)
        semantic_on = self.test_cfg.get('semantic_on', False)
        instance_on = self.test_cfg.get('instance_on', False)
        assert not semantic_on, 'segmantic segmentation '\
            'results are not supported yet.'

        results = []
        for mask_cls_result, mask_pred_result, meta in zip(
                mask_cls_results, mask_pred_results, img_metas):
            # remove padding
            img_height, img_width = meta['img_shape'][:2]
            mask_pred_result = mask_pred_result[:, :img_height, :img_width]

            if rescale:
                # return result in original resolution
                ori_height, ori_width = meta['ori_shape'][:2]
                mask_pred_result = F.interpolate(
                    mask_pred_result[:, None],
                    size=(ori_height, ori_width),
                    mode='bilinear',
                    align_corners=False)[:, 0]

            result = dict()
            if panoptic_on:
                pan_results, entityid_list, entity_score_list = self.panoptic_postprocess(
                    mask_cls_result, mask_pred_result)
                result['pan_results'] = pan_results
                result['entityid_list'] = entityid_list
                result['entity_score_list'] = entity_score_list

            if instance_on:
                ins_results = self.instance_postprocess(
                    mask_cls_result, mask_pred_result)
                result['ins_results'] = ins_results

            if semantic_on:
                sem_results = self.semantic_postprocess(
                    mask_cls_result, mask_pred_result)
                result['sem_results'] = sem_results

            results.append(result)

        return results


@HEADS.register_module()
class MaskFormerFusionHead2(MaskFormerFusionHead):
    def panoptic_postprocess(self, mask_cls, mask_pred):
        """Panoptic segmengation inference.

        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_queries, cls_out_channels) for a image.
                Note `cls_out_channels` should includes
                background.
            mask_pred (Tensor): Mask outputs of shape
                (num_queries, h, w) for a image.

        Returns:
            Tensor: Panoptic segment result of shape \
                (h, w), each element in Tensor means: \
                ``segment_id = _cls + instance_id * INSTANCE_OFFSET``.
            
        """
        object_mask_thr = self.test_cfg.get('object_mask_thr', 0.8)
        iou_thr = self.test_cfg.get('iou_thr', 0.8)
        filter_low_score = self.test_cfg.get('filter_low_score', False)

        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.num_classes) & (scores > object_mask_thr)
        keep_index = torch.nonzero(keep)
        # scores (n,1) , labels (n,1)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        # cur_masks (n, h, w)
        cur_masks = mask_pred[keep]

        # 计算出所有保留下来的mask的分数和mask和标签
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.full((h, w),
                                  self.num_classes,
                                  dtype=torch.int32,
                                  device=cur_masks.device)
        entityid_list = []
        entity_score_list = []

        # 找到每个mask对应的query的index
        target_keep = []
        target_dict = {}

        # 对应的mask
        pre_mask = []

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            pass
        else:
            mode = 'raw'
            if mode == 'area':
                areas = (cur_masks>=0.5).sum(dim=[1,2])
                area_tensor, idx_tensor = torch.sort(areas)
                instance_id = 1
                for area, idx_cur in zip(area_tensor.flip(dims=[0]), idx_tensor.flip(dims=[0])):
                    if area <= 0:
                        continue
                    pred_class = cur_classes[idx_cur].to(torch.long)
                    if pred_class == 133:
                        continue
                    pred_mask = cur_masks[idx_cur] >= 0.5
                    pred_score = cur_prob_masks[idx_cur][pred_mask].mean()

                    isthing = pred_class < self.num_things_classes
                    if not isthing:
                        panoptic_seg[pred_mask] = panoptic_seg[pred_mask] * 0 + pred_class
                        eid = pred_class
                    else:
                        panoptic_seg[pred_mask] = panoptic_seg[pred_mask] * 0 + (pred_class + instance_id * INSTANCE_OFFSET)
                        eid = pred_class + instance_id * INSTANCE_OFFSET
                        instance_id += 1
    
                    entityid_list.append(eid)
                    entity_score_list.append(pred_score)
            elif mode == 'raw':
                # cur_mask_ids = cur_prob_masks.argmax(0)
                cur_mask_score, cur_mask_ids = cur_prob_masks.max(dim=0)
                instance_id = 1
                for k in range(cur_classes.shape[0]):
                    # pred_class = int(cur_classes[k].item())
                    pred_class = cur_classes[k].to(torch.long)
                    isthing = pred_class < self.num_things_classes
                    mask = cur_mask_ids == k
                    mask_area = mask.sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()

                    score = cur_mask_score[mask].mean()

                    if filter_low_score:
                        mask = mask & (cur_masks[k] >= 0.5)
                        # mask_area = mask.sum().item()

                    if mask_area > 0 and original_area > 0:
                        if mask_area / original_area < iou_thr:
                            continue
                        # if not isthing:
                        #     # different stuff regions of same class will be
                        #     # merged here, and stuff share the instance_id 0.
                        #     panoptic_seg[mask] = panoptic_seg[mask] * 0 + pred_class
                        #     # entityid_list.append(pred_class)
                        #     # entity_score_list.append(score)
                        # else:
                        panoptic_seg[mask] = panoptic_seg[mask] * 0 + (pred_class + instance_id * INSTANCE_OFFSET)
                        target_dict[pred_class.item() + instance_id * INSTANCE_OFFSET] = keep_index[k]
                        instance_id += 1
                
                for eid in torch.unique(panoptic_seg):
                    eid = eid.item()
                    if eid == 133:
                        continue
                    mask = panoptic_seg == eid
                    target_keep.append(target_dict[eid])
                    pre_mask.append(mask)
                    score = cur_mask_score[mask].mean()
                    entityid_list.append(eid)
                    entity_score_list.append(score)

        # print([eid.item() for eid in entityid_list])
        # return panoptic_seg
        return panoptic_seg, entityid_list, entity_score_list, target_keep, pre_mask

    def simple_test(self,
                    mask_cls_results,
                    mask_pred_results,
                    img_metas,
                    rescale=False,
                    **kwargs):
        """Test segment without test-time aumengtation.

        Only the output of last decoder layers was used.

        Args:
            mask_cls_results (Tensor): Mask classification logits,
                shape (batch_size, num_queries, cls_out_channels).
                Note `cls_out_channels` should includes background.
            mask_pred_results (Tensor): Mask logits, shape
                (batch_size, num_queries, h, w).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): If True, return boxes in
                original image space. Default False.

        Returns:
            list[dict[str, Tensor | tuple[Tensor]]]: Semantic segmentation \
                results and panoptic segmentation results for each \
                image.

            .. code-block:: none

                [
                    {
                        'pan_results': Tensor, # shape = [h, w]
                        'ins_results': tuple[Tensor],
                        # semantic segmentation results are not supported yet
                        'sem_results': Tensor
                    },
                    ...
                ]
        """
        panoptic_on = self.test_cfg.get('panoptic_on', True)
        semantic_on = self.test_cfg.get('semantic_on', False)
        instance_on = self.test_cfg.get('instance_on', False)
        assert not semantic_on, 'segmantic segmentation '\
            'results are not supported yet.'

        results = []
        for mask_cls_result, mask_pred_result, meta in zip(
                mask_cls_results, mask_pred_results, img_metas):
            # remove padding
            img_height, img_width = meta['img_shape'][:2]
            mask_pred_result = mask_pred_result[:, :img_height, :img_width]

            if rescale:
                # return result in original resolution
                ori_height, ori_width = meta['ori_shape'][:2]
                mask_pred_result = F.interpolate(
                    mask_pred_result[:, None],
                    size=(ori_height, ori_width),
                    mode='bilinear',
                    align_corners=False)[:, 0]

            result = dict()
            if panoptic_on:
                pan_results, entityid_list, entity_score_list, target_keep, pre_mask = self.panoptic_postprocess(
                    mask_cls_result, mask_pred_result)
                result['pan_results'] = pan_results
                result['entityid_list'] = entityid_list
                result['entity_score_list'] = entity_score_list
                result['target_keep'] = target_keep
                result['object_mask'] = pre_mask

            if instance_on:
                ins_results = self.instance_postprocess(
                    mask_cls_result, mask_pred_result)
                result['ins_results'] = ins_results

            if semantic_on:
                sem_results = self.semantic_postprocess(
                    mask_cls_result, mask_pred_result)
                result['sem_results'] = sem_results

            results.append(result)

        return results
    
    def batch_test(self,
                    mask_cls_results,
                    mask_pred_results,
                    **kwargs):
        """Test segment without test-time aumengtation.

        Only the output of last decoder layers was used.

        Args:
            mask_cls_results (Tensor): Mask classification logits,
                shape (batch_size, num_queries, cls_out_channels).
                Note `cls_out_channels` should includes background.
            mask_pred_results (Tensor): Mask logits, shape
                (batch_size, num_queries, h, w).

        Returns:

        """
        results = []
        for mask_cls_result, mask_pred_result in zip(mask_cls_results, mask_pred_results):

            result = dict()
            pan_results, entityid_list, entity_score_list, target_keep, pre_mask = self.panoptic_postprocess(
                mask_cls_result, mask_pred_result)
            # result['pan_results'] = pan_results
            result['entityid_list'] = entityid_list
            result['entity_score_list'] = entity_score_list
            result['target_keep'] = target_keep
            # result['object_mask'] = pre_mask

            results.append(result)

        return results

