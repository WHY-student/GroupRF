# Copyright (c) OpenMMLab. All rights reserved.
import copy

import mmcv
import numpy as np

from mmdet.core import INSTANCE_OFFSET, bbox2result
from mmdet.core.visualization import imshow_det_bboxes
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .mask2formerrelation import Mask2FormerRelation
from .single_stage import SingleStageDetector

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from IPython import embed
from mmdet.utils import AvoidCUDAOOM
import random
import math
from ..utils.group_token import Mlp

@DETECTORS.register_module()
class Mask2FormerVit(Mask2FormerRelation):
    r"""Implementation of `Masked-attention Mask
    Transformer for Universal Image Segmentation
    <https://arxiv.org/pdf/2112.01527>`_."""

    def __init__(self,
                 backbone,
                 neck=None,
                 panoptic_head=None,
                 panoptic_fusion_head=None,
                 relationship_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 ):
        super().__init__(
            backbone,
            neck=neck,
            panoptic_head=panoptic_head,
            panoptic_fusion_head=panoptic_fusion_head,
            relationship_head=None,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        self.relationship_head = build_head(relationship_head)
        
        
    
    # @AvoidCUDAOOM.retry_if_cuda_oom
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_semantic_seg=None,
                      gt_bboxes_ignore=None,
                      **kargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[Dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            gt_masks (list[BitmapMasks]): true segmentation masks for each box
                used if the architecture supports a segmentation task.
            gt_semantic_seg (list[tensor]): semantic segmentation mask for
                images for panoptic segmentation.
                Defaults to None for instance segmentation.
            gt_bboxes_ignore (list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
                Defaults to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # add batch_input_shape in img_metas
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)

        losses, query_feat, pos_inds_list, pos_assigned_gt_inds_list  = self.panoptic_head.forward_train(x, img_metas, gt_bboxes,
                                                  gt_labels, gt_masks,
                                                  gt_semantic_seg,
                                                  gt_bboxes_ignore)
        
        relation_pred, all_edge_lbl, bs_size = self.relationship_head.forward_train(query_feat, pos_inds_list, pos_assigned_gt_inds_list, img_metas)

        losses_relation = {}
        # print(relation_pred.shape,all_edge_lbl.shape)

        # exit(0)
        losses_relation['loss_relationship'] = self.relationship_head.loss(relation_pred, all_edge_lbl)
        
        recall = self.get_recall_N(relation_pred, all_edge_lbl, bs_size, img_metas)
        losses_relation['rela.recall@20'] = recall

        losses.update(losses_relation)

        # torch.cuda.empty_cache()
        return losses

    def simple_test(self, imgs, img_metas, **kwargs):
        feats = self.extract_feat(imgs)
        mask_cls_results, mask_pred_results, mask_features, query_feat = self.panoptic_head.simple_test(feats, img_metas, **kwargs)
        results = self.panoptic_fusion_head.simple_test(mask_cls_results, mask_pred_results, img_metas, **kwargs)
        
        device = mask_features.device
        dtype = mask_features.dtype

        res = results[0]
        # pan_results = res['pan_results']
        entityid_list = res['entityid_list']
        # entity_score_list = res['entity_score_list']
        target_keep = res['target_keep']
        # pre_mask = res['object_mask']

        entity_embedding = query_feat[0][target_keep,:]
        # print(entity_embedding.shape)
        # entity_res = self.get_entity_embedding(
        #     pan_result=pan_results,
        #     entity_id_list=entityid_list,
        #     entity_score_list=entity_score_list,
        #     feature_map=mask_features,
        #     meta=img_metas[0]
        # )


        relation_res = []
        if len(entityid_list) != 0:
            relation_pred, neg_idx = self.relationship_head.simple_test(query_feat, entity_embedding)
            # print(relation_pred.shape)
            relation_pred = torch.softmax(relation_pred, dim=-1)
            # 去除预测为空关系标签的影响
            relation_pred = relation_pred[:,1:]
            relation_pred[neg_idx,:] = -9999
            try:
                _, topk_indices = torch.topk(relation_pred.reshape([-1,]), k=100)
            except:
                topk_indices = torch.tensor(range(0,len(relation_pred.reshape([-1,]))))
            # subject, object, cls
            for index in topk_indices:
                pred_cls = index % relation_pred.shape[1]
                index_subject_object = index // relation_pred.shape[1]
                pred_subject = index_subject_object // entity_embedding.shape[0]
                pred_object = index_subject_object % entity_embedding.shape[0]
                pred = [pred_subject.item(), pred_object.item(), pred_cls.item()]
                relation_res.append(pred)
     
        rl = dict(
            entityid_list=entityid_list,
            relation=relation_res,
        )

        res['rela_results'] = rl
        res['pan_results'] = res['pan_results'].detach().cpu().numpy()

        return [res]

    def forward_dummy(self, imgs, img_metas=None):
        """Used for computing network flops. See
        `mmdetection/tools/analysis_tools/get_flops.py`

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[Dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
        """
        if img_metas==None:
            img_metas = [{'batch_input_shape': img.shape,'img_shape':img.shape, 'ori_shape': img.shape} for img in imgs]
        feats = self.extract_feat(imgs)
        mask_cls_results, mask_pred_results, mask_features, query_feat = self.panoptic_head.simple_test(feats, img_metas)
        results = self.panoptic_fusion_head.simple_test(mask_cls_results, mask_pred_results, img_metas)
        
        device = mask_features.device
        dtype = mask_features.dtype

        res = results[0]
        # pan_results = res['pan_results']
        entityid_list = res['entityid_list']
        # entity_score_list = res['entity_score_list']
        target_keep = res['target_keep']
        # pre_mask = res['object_mask']

        entity_embedding = query_feat[0][target_keep,:]
        # print(entity_embedding.shape)
        # entity_res = self.get_entity_embedding(
        #     pan_result=pan_results,
        #     entity_id_list=entityid_list,
        #     entity_score_list=entity_score_list,
        #     feature_map=mask_features,
        #     meta=img_metas[0]
        # )


        relation_res = []
        if len(entityid_list) != 0:
            relation_pred, neg_idx = self.relationship_head.simple_test(query_feat, entity_embedding)
            # print(relation_pred.shape)
            relation_pred = torch.softmax(relation_pred, dim=-1)
            # 去除预测为空关系标签的影响
            relation_pred = relation_pred[:,1:]
            relation_pred[neg_idx,:] = -9999
            try:
                _, topk_indices = torch.topk(relation_pred.reshape([-1,]), k=100)
            except:
                topk_indices = torch.tensor(range(0,len(relation_pred.reshape([-1,]))))
            # subject, object, cls
            for index in topk_indices:
                pred_cls = index % relation_pred.shape[1]
                index_subject_object = index // relation_pred.shape[1]
                pred_subject = index_subject_object // entity_embedding.shape[0]
                pred_object = index_subject_object % entity_embedding.shape[0]
                pred = [pred_subject.item(), pred_object.item(), pred_cls.item()]
                relation_res.append(pred)
     
        rl = dict(
            entityid_list=entityid_list,
            relation=relation_res,
        )

        res['rela_results'] = rl
        res['pan_results'] = res['pan_results'].detach().cpu().numpy()

        return [res]

  
    def get_recall_N(self, y_pred, y_true, bs_size, img_metas, N=20):
        # y_pred     [n,57]
        # y_true     [n]
        # bs_size   [bs]
        device = y_pred.device

        if len(y_true) == 0:
            return 0
        idxs = torch.nonzero(y_true)
        y_true_label = y_true[idxs]
        y_pred_label_scores = y_pred[idxs,:]

        index = 0
        recall = []
        for idx, i in enumerate(bs_size):
            correct = 0
            bs_true_label = y_true_label[index:index+i]
            bs_pre_label_scores = y_pred_label_scores[index:index+i,:]
            if i==0:
                continue
            try:
                _, top_k_indices = torch.topk(bs_pre_label_scores.view(-1), k=20, largest=True)
            except:
                print(i)
                print(bs_size)
                print(bs_pre_label_scores.shape)
            for indices in top_k_indices:
                # print(indices)
                # print(bs_true_label[indices // 57])
                if bs_true_label[indices // 57] == indices % 57:
                    correct += 1
            all_num = len(img_metas[idx]['gt_relationship'][0])
            recall.append(correct/all_num)
            index += i
            


        # if y_true.numel() == 0:
        #     return [torch.zeros([], device=y_pred.device)]
        # maxk = max(topk)
        # batch_size = target.size(0)

        # _, pred = output.topk(maxk, 1, True, True)
        # pred = pred.t()
        # correct = pred.eq(target.view(1, -1).expand_as(pred))

        # res = []
        # for k in topk:
        #     correct_k = correct[:k].view(-1).float().sum(0)
        #     res.append(correct_k.mul_(100.0 / batch_size))
        # return mean(recall)

       
        mean_recall = torch.tensor(recall).to(device).mean() * 100
        return mean_recall


@DETECTORS.register_module()
class Mask2FormerVitForinfer(Mask2FormerVit):
    def get_entity_embedding(self, pan_result, entity_id_list, entity_score_list, feature_map, meta):
        device = feature_map.device
        dtype = feature_map.dtype

        ori_height, ori_width = meta['ori_shape'][:2]
        resize_height, resize_width = meta['img_shape'][:2]
        pad_height, pad_width = meta['pad_shape'][:2]

        mask_list = []
        class_mask_list = []
        instance_id_all = entity_id_list
        for idx_instance, instance_id in enumerate(instance_id_all):
            if instance_id == 133:
                continue
            mask = pan_result == instance_id
            class_mask = instance_id % INSTANCE_OFFSET
            # class_score = entity_score_list[idx_instance]
            mask_list.append(mask)
            class_mask_list.append(class_mask)

        if len(mask_list) == 0:
            return None
        
        class_mask_tensor = torch.tensor(class_mask_list).to(device).to(torch.long)[None]
        cls_entity_embedding = self.rela_cls_embed(class_mask_tensor)

        mask_tensor = torch.stack(mask_list)[None]
        mask_tensor = (mask_tensor * 1).to(dtype)
        h_img, w_img = resize_height, resize_width
        mask_tensor = F.interpolate(mask_tensor, size=(h_img, w_img))
        h_pad, w_pad = pad_height, pad_width
        mask_tensor = F.pad(mask_tensor, (0, w_pad-w_img, 0, h_pad-h_img))
        h_feature, w_feature = feature_map.shape[-2:]
        mask_tensor = F.interpolate(mask_tensor, size=(h_feature, w_feature))
        mask_tensor = mask_tensor[0][:, None]

        # feature_map [bs, 256, h, w]
        # mask_tensor [n, 1, h, w]
        if self.entity_length > 1:
            entity_embedding_list = []
            for idx in range(len(mask_list)):
                # embedding [self.entity_length, 256]
                embedding = self._mask_pooling(feature_map[0], mask_tensor[idx], self.entity_length)
                embedding = embedding + cls_entity_embedding[0, idx:idx+1]

                if self.add_postional_encoding:
                    # [1, h, w]
                    pos_embed_zeros = feature_map[0].new_zeros((1, ) + feature_map[0].shape[-2:])
                    # [1, 256, h, w]
                    pos_embed = self.relationship_head.postional_encoding_layer(pos_embed_zeros)
                    pos_embed_mask_pooling = self._mask_pooling(pos_embed[0], mask_tensor[idx], output_size=self.entity_length)
                    embedding = embedding + pos_embed_mask_pooling


                if self.use_background_feature:
                    background_embedding = self._mask_pooling(feature_map[0], 1 - mask_tensor[idx], self.entity_length)
                    embedding = embedding + background_embedding

                
                entity_embedding_list.append(embedding[None])

            # embedding [1, n*self.entity_length, 256]
            embedding = torch.cat(entity_embedding_list, dim=1)
            # entity_embedding [1, n, 256]
            entity_embedding = self._entity_encode(embedding)

        else:
            entity_embedding = (feature_map * mask_tensor).sum(dim=[2, 3]) / (mask_tensor.sum(dim=[2, 3]) + 1e-8)
            entity_embedding = entity_embedding[None]
            if self.cls_embedding_mode == 'cat':
                entity_embedding = torch.cat([entity_embedding, cls_entity_embedding], dim=-1)
            elif self.cls_embedding_mode == 'add':
                entity_embedding = entity_embedding + cls_entity_embedding


            if self.add_postional_encoding:
                pos_embed_zeros = feature_map[0].new_zeros((1, ) + feature_map[0].shape[-2:])
                pos_embed = self.relationship_head.postional_encoding_layer(pos_embed_zeros)
                for idx in range(entity_embedding.shape[1]):
                    pos_embed_mask_pooling = self._mask_pooling(pos_embed[0], mask_tensor[idx], output_size=self.entity_length)
                    entity_embedding[0, idx] = entity_embedding[0, idx] + pos_embed_mask_pooling


            if self.use_background_feature:
                background_mask = 1 - mask_tensor
                background_feature = (feature_map * background_mask).sum(dim=[2, 3]) / (background_mask.sum(dim=[2, 3]) + 1e-8)
                background_feature = background_feature[None]
                # entity_embedding [1, n, 256]
                # entity_embedding = entity_embedding + background_feature
                entity_embedding = self.relu(entity_embedding + background_feature) - (entity_embedding - background_feature)**2

        # entity_embedding [1, n, 256]
        return entity_embedding, entity_id_list, entity_score_list

    def simple_test(self, imgs, img_metas=None, **kwargs):
        if img_metas==None:
            img_metas = [{'batch_input_shape': img.shape,'img_shape':img.shape, 'ori_shape': img.shape} for img in imgs]
        feats = self.extract_feat(imgs)
        mask_cls_results, mask_pred_results, mask_features, query_feat = self.panoptic_head.simple_test(feats, img_metas, **kwargs)
        results = self.panoptic_fusion_head.simple_test(mask_cls_results, mask_pred_results, img_metas, **kwargs)
        
        device = mask_features.device
        dtype = mask_features.dtype

        res = results[0]
        # pan_results = res['pan_results']
        entityid_list = res['entityid_list']
        entity_score_list = res['entity_score_list']
        target_keep = res['target_keep']
        # pre_mask = res['object_mask']

        # print(entity_embedding.shape)
        # entity_res = self.get_entity_embedding(
        #     pan_result=pan_results,
        #     entity_id_list=entityid_list,
        #     entity_score_list=entity_score_list,
        #     feature_map=mask_features,
        #     meta=img_metas[0]
        # )

        entity_num = len(entityid_list)

        relation_res = []
        if entity_num != 0:
            relation_pred, neg_idx = self.relationship_head.simple_test(query_feat, target_keep)
            # print(relation_pred.shape)
            # relation_pred = torch.softmax(relation_pred, dim=-1)
            # 去除预测为空关系标签的影响
            # tokens_scores = tokens_scores[:,:,1:]
            # tokens_scores = torch.exp(tokens_scores)

            relation_pred = relation_pred[:,1:]
            relation_pred[neg_idx,:] = -9999
            relation_pred = torch.exp(relation_pred)
            # relation_pred = torch.softmax(relation_pred, dim=-1)

            entity_score_tensor = torch.tensor(entity_score_list, device=device, dtype=dtype).unsqueeze(0)
            entity_score_tensor = torch.matmul(entity_score_tensor.t(), entity_score_tensor).reshape(1,-1).expand(56, -1)
            # print(torch.ones((16,3)))
            relation_pred = torch.mul(entity_score_tensor.t(), relation_pred)
            
            try:
                _, topk_indices = torch.topk(relation_pred.reshape([-1,]), k=100)
            except:
                topk_indices = torch.tensor(range(0,len(relation_pred.reshape([-1,]))))
            # subject, object, cls
            for index in topk_indices:
                pred_cls = index % relation_pred.shape[1]
                index_subject_object = index // relation_pred.shape[1]
                pred_subject = index_subject_object // entity_num
                pred_object = index_subject_object % entity_num
                pred = [pred_subject.item(), pred_object.item(), pred_cls.item()]
                relation_res.append(pred)
     
        rl = dict(
            entityid_list=entityid_list,
            relation=relation_res,
        )

        res['rela_results'] = rl
        res['pan_results'] = res['pan_results'].detach().cpu().numpy()

        return [res]


def concat_relation_features(object_features, relation_tokens, target_edges):

    rel_labels = [t[:,2] if len(t)>0 else torch.zeros((0,1), dtype=torch.long).to(relation_tokens.device) for t in target_edges]
    target_edges = [t[:,:2] if len(t)>0 else torch.zeros((0,2), dtype=torch.long).to(relation_tokens.device) for t in target_edges]

    all_edge_lbl = []
    relation_features = []

    total_edge = 0
    total_fg = 0

    bs_size = []
    # loop through each of batch to collect the edge and node
    for b_id, (filtered_edge, rel_label, object_feature, relation_token) in enumerate(zip(target_edges, rel_labels, object_features , relation_tokens)):
        if len(filtered_edge) == 0:
            bs_size.append(0)
            continue
        # find the -ve edges for training
        full_adj = torch.ones((object_feature.shape[0],object_feature.shape[0])) - torch.diag(torch.ones(object_feature.shape[0]))
        full_adj[filtered_edge[:,0], filtered_edge[:,1]] = 0
        neg_edges = torch.nonzero(full_adj).to(filtered_edge.device)
        # neg_edges all relation that shouldn't be predicted

        # restrict unbalance in the +ve/-ve edge
        if filtered_edge.shape[0]>30:
            idx_ = torch.randperm(filtered_edge.shape[0])[:30]
            filtered_edge = filtered_edge[idx_,:]
            rel_label = rel_label[idx_]
        bs_size.append(filtered_edge.shape[0])

        # check whether the number of -ve edges are within limit
        if neg_edges.shape[0]>=90:# random sample -ve edge
            idx_ = torch.randperm(neg_edges.shape[0])[:90]
            neg_edges = neg_edges[idx_,:]
        # neg_size.append(neg_edges.shape[0])

        all_edges_ = torch.cat((filtered_edge, neg_edges), 0)
        # all_edges_ = [object_num*object_num-diag, c]
        total_edge += all_edges_.shape[0]
        total_fg += filtered_edge.shape[0]
        edge_labels = torch.cat((rel_label, torch.zeros(neg_edges.shape[0], dtype=torch.long).to(object_feature.device)), 0)
        #now permute all the combination
        idx_ = torch.randperm(all_edges_.shape[0])
        all_edges_ = all_edges_[idx_,:]
        edge_labels = edge_labels[idx_]
        all_edge_lbl.append(edge_labels)

        token_relation_features = []
        for idx, edges in enumerate(all_edges_):
            token_relation_features.append(torch.cat((object_feature[edges[0],:].repeat(relation_token.shape[0],1),object_feature[edges[1],:].repeat(relation_token.shape[0],1),relation_token), 1).view(1, -1))
        relation_features.append(torch.cat(token_relation_features,0))
    relation_features = torch.cat(relation_features, 0)


    all_edge_lbl = torch.cat(all_edge_lbl, 0).to(object_features[0].device)
    return relation_features, all_edge_lbl, bs_size


def concat_relation_features_test(entity_embedding, relation_tokens):
    '''
    entity_embedding: (object_num, feature_size)
    relation_tokens:  (self.token_num, feature_size)
    '''
    object_num = entity_embedding.shape[0]
    token_num = relation_tokens.shape[0]

    relation_features = []
    neg_idx = []
    for i in range(object_num):
        neg_idx.append(int(i*object_num+i))
        for j in range(object_num):
            relation_features.append(torch.cat((entity_embedding[i, :].repeat(token_num, 1), entity_embedding[j, :].repeat(token_num, 1), relation_tokens), 1).view(1, -1))
    
    relation_features = torch.cat(relation_features, 0)

    return relation_features, neg_idx
