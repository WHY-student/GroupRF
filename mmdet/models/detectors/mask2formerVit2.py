# Copyright (c) OpenMMLab. All rights reserved.
import copy

import mmcv
import numpy as np

from mmdet.core import INSTANCE_OFFSET, bbox2result
from mmdet.core.visualization import imshow_det_bboxes
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .mask2formerrelation import Mask2FormerRelation
from .single_stage import SingleStageDetector
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from IPython import embed
from mmdet.utils import AvoidCUDAOOM
import random
import math
from ..utils.group_token import GroupingBlock, GroupingLayer

import matplotlib.pyplot as plt
import seaborn as sns

@DETECTORS.register_module()
class Mask2FormerVit2(Mask2FormerRelation):
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
                 feed_forward=1024,
                 ):
        super().__init__(
            backbone,
            neck=neck,
            panoptic_head=panoptic_head,
            panoptic_fusion_head=panoptic_fusion_head,
            relationship_head=relationship_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        
        norm_layer = nn.LayerNorm
        num_layers = 3
        num_input_token = 100
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(num_layers-1):

            dim = int(embed_dim * embed_factors[i_layer])
            # downsample = None
            # if i_layer < self.num_layers - 1:
            out_dim = int(embed_dim * embed_factors[i_layer + 1])
            downsample = GroupingBlock(
                dim=dim,
                out_dim=out_dim,
                num_heads=num_heads[i_layer],
                num_group_token=num_group_tokens[i_layer],
                num_output_group=num_output_groups[i_layer],
                norm_layer=norm_layer,
                hard=hard_assignment,
                gumbel=hard_assignment)
            num_output_token = num_output_groups[i_layer]

            # if i_layer > 0 and num_group_tokens[i_layer] > 0:
            #     prev_dim = int(embed_dim * embed_factors[i_layer - 1])
            #     group_projector = nn.Sequential(
            #         norm_layer(prev_dim),
            #         MixerMlp(num_group_tokens[i_layer - 1], prev_dim // 2, num_group_tokens[i_layer]))

            #     if dim != prev_dim:
            #         group_projector = nn.Sequential(group_projector, norm_layer(prev_dim),
            #                                         nn.Linear(prev_dim, dim, bias=False))
            # else:
            group_projector = None
            layer = GroupingLayer(
                dim=dim,
                num_input_token=num_input_token,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                num_group_token=num_group_tokens[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=downsample,
                use_checkpoint=use_checkpoint,
                group_projector=group_projector,
                # only zero init group token if we have a projection
                zero_init_group_token=group_projector is not None)
            self.layers.append(layer)
            # if i_layer < self.num_layers - 1:
            num_input_token = num_output_token
        
        self.norm = norm_layer(embed_dim)
        
        self.num_cls = 56
        self.token_num = num_output_groups[-1]
        self.relation_embedding = Mlp(3*embed_dim*self.token_num,feed_forward, 3*embed_dim)
        self.relation_head = Mlp(3*embed_dim,feed_forward, self.num_cls+1)
        
        
    
    @AvoidCUDAOOM.retry_if_cuda_oom
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
        # losses, query_feat, group_token, pos_inds_list, pos_assigned_gt_inds_list  = self.panoptic_head.forward_train(x, img_metas, gt_bboxes,
        #                                           gt_labels, gt_masks,
        #                                           gt_semantic_seg,
        #                                           gt_bboxes_ignore)
        losses, query_feat, pos_inds_list, pos_assigned_gt_inds_list  = self.panoptic_head.forward_train(x, img_metas, gt_bboxes,
                                                  gt_labels, gt_masks,
                                                  gt_semantic_seg,
                                                  gt_bboxes_ignore)

        # 提取出query后加上group rln_tokens
        # query_feat = query_feat + self.get_pos_embed(B, *hw_shape)
        # object_feature_list = []
        # target_relation = []
        # for idx in range(bs):
        #     embedding, target_relationship = self._get_entity_embedding_and_target(
        #         mask_features[idx],
        #         img_metas[idx],
        #         gt_masks[idx],
        #         gt_labels[idx],
        #         gt_semantic_seg[idx],
        #     )
        #     embedding = embedding.squeeze(0)
        #     # relation_target = []
        #     # for t in meta_info['gt_relationship'][0]:
        #     #     relation_target.append([t[0], t[1], t[2]+1])
        #     target_relation.append(target_relationship)
        #     object_feature_list.append(embedding)

        temp_group_token = None
        group_token = query_feat
        for layer in self.layers:
            group_token, temp_group_token, _ = layer(group_token, temp_group_token)

        group_token = self.norm(group_token)
        
        object_feature_list, target_relation = self.get_embedding_relation(query_feat, pos_inds_list, pos_assigned_gt_inds_list, img_metas)


        relation_feature, all_edge_lbl, bs_size = concat_relation_features(object_feature_list, group_token, target_relation)

        relation_pred = self.relation_head(self.relation_embedding(relation_feature))
        # print(relation_pred.shape,all_edge_lbl.shape)

        # exit(0)
        losses_relation = {}
        # loss = F.cross_entropy(relation_pred, all_edge_lbl, reduction='mean')
        target_relation_tensor = torch.zeros_like(relation_pred)
        for id, label in enumerate(all_edge_lbl):
            target_relation_tensor[id, label] = 1
        loss = self.multilabel_categorical_crossentropy(target_relation_tensor, relation_pred)
        loss = loss.mean()
        losses_relation['loss_relationship'] = loss * 30

        
        recall = self.get_recall_N(relation_pred, all_edge_lbl, bs_size, img_metas)
        losses_relation['rela.recall@20'] = recall.mean()

        losses.update(losses_relation)

        torch.cuda.empty_cache()
        return losses
    
    def get_embedding_relation(self, query_feat, pos_inds_list, pos_assigned_gt_inds_list,img_metas):
        bs = query_feat.shape[0]
        device = query_feat.device

        object_features = []
        target_relations = []
        # for query_feat_one, pos_inds_list_one, pos_assigned_gt_inds_list_one,
        for idx in range(bs):
            query_feat_one = query_feat[idx]
            pos_inds_list_one = pos_inds_list[idx]
            pos_assigned_gt_inds_list_one = pos_assigned_gt_inds_list[idx].tolist()
            # 得到映射关系字典，方便处理原relation
            pos_assigned_gt_inds_dict_one = {}
            for new, old in enumerate(pos_assigned_gt_inds_list_one):
                pos_assigned_gt_inds_dict_one[old] = new

            gt_relations_one = img_metas[idx]['gt_relationship'][0]
            object_feature_one = query_feat_one[pos_inds_list_one, :]
            
            target_relations_one = []
            for gt_relation in gt_relations_one:
                # 保证relation已经匹配到
                if gt_relation[0] in pos_assigned_gt_inds_list_one and gt_relation[1] in pos_assigned_gt_inds_list_one:
                    target_relations_one.append([pos_assigned_gt_inds_dict_one[gt_relation[0]],pos_assigned_gt_inds_dict_one[gt_relation[1]], gt_relation[2] + 1])
            object_features.append(object_feature_one)
            target_relations.append(torch.tensor(target_relations_one).to(device))
        
        return object_features, target_relations

    def multilabel_categorical_crossentropy(self, y_true, y_pred):
        """多标签分类的交叉熵
        from 苏剑林/科学空间
        说明：y_true和y_pred的shape一致，y_true的元素非0即1，
            1表示对应的类为目标类，0表示对应的类为非目标类。
        警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
            不用加激活函数，尤其是不能加sigmoid或者softmax！预测
            阶段则输出y_pred大于0的类。
        """
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 9999
        y_pred_pos = y_pred - (1 - y_true) * 9999
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return neg_loss + pos_loss

    def simple_test(self, imgs, img_metas, **kwargs):
        
        feats = self.extract_feat(imgs)
        mask_cls_results, mask_pred_results, mask_features, query_feat = self.panoptic_head.simple_test(feats, img_metas, **kwargs)
        # mask_cls_results, mask_pred_results, mask_features, query_feat, group_token = self.panoptic_head.simple_test(feats, img_metas, **kwargs)
        results = self.panoptic_fusion_head.simple_test(mask_cls_results, mask_pred_results, img_metas, **kwargs)
        
        for i in range(len(results)):
            if 'pan_results' in results[i]:
                results[i]['pan_results'] = results[i]['pan_results'].detach(
                ).cpu().numpy()

            if 'ins_results' in results[i]:
                labels_per_image, bboxes, mask_pred_binary = results[i][
                    'ins_results']
                bbox_results = bbox2result(bboxes, labels_per_image,
                                           self.num_things_classes)
                mask_results = [[] for _ in range(self.num_things_classes)]
                for j, label in enumerate(labels_per_image):
                    mask = mask_pred_binary[j].detach().cpu().numpy()
                    mask_results[label].append(mask)
                results[i]['ins_results'] = bbox_results, mask_results

            assert 'sem_results' not in results[i], 'segmantic segmentation '\
                'results are not supported yet.'

        if self.num_stuff_classes == 0:
            results = [res['ins_results'] for res in results]

        torch.cuda.empty_cache()
        
        return results
    

    def get_recall_N(self, y_pred, y_true, bs_size, img_metas, N=20):
        # y_pred     [n,57]
        # y_true     [n]
        # bs_size   [bs]
        device = y_pred.device

        if len(y_true) == 0:
            return torch.tensor(0).to(device)
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
class Mask2FormerVitForinfer2(Mask2FormerVit2):
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

    def simple_test(self, imgs, img_metas, **kwargs):
        feats = self.extract_feat(imgs)
        mask_cls_results, mask_pred_results, mask_features, query_feat = self.panoptic_head.simple_test(feats, img_metas, **kwargs)
        # mask_cls_results, mask_pred_results, mask_features, query_feat = self.panoptic_head.simple_test(feats, img_metas, **kwargs)
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

        temp_group_token = None
        group_token = query_feat
        for layer in self.layers:
            group_token, temp_group_token, _ = layer(group_token, temp_group_token)
        group_token = self.norm(group_token)
        group_token = group_token[0]

        # # print(group_token.shape)
        # cos_scores = [[0]*4 for _ in range(4)]
        # for i in range(4):
        #     for j in range(4):
        #         cos_scores[i][j] = round(F.cosine_similarity(group_token[i].unsqueeze(0),group_token[j].unsqueeze(0)).item(),2)
        # print(cos_scores)
        # sns.heatmap(cos_scores, annot=True, cmap="YlGnBu")
        # # 可选：添加轴标签
        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.savefig('token_cos_8_4_None.png')
        # exit(0)

        relation_res = []
        if len(entityid_list) != 0:
            relation_embeddings, neg_idx = concat_relation_features_test(entity_embedding, group_token)
            relation_pred = self.relation_head(self.relation_embedding(relation_embeddings))
            relation_pred = torch.softmax(relation_pred, dim=-1)
            # 去除预测为空关系标签的影响
            relation_pred = relation_pred[:,1:]
            relation_pred[neg_idx,:] = -9999
            try:
                _, topk_indices = torch.topk(relation_pred.reshape([-1,]), k=20)
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
        #     relationship_output = self.relationship_head(entity_embedding, attention_mask=None)
        #     relationship_output = relationship_output[0]
        #     for idx_i in range(relationship_output.shape[1]):
        #         relationship_output[:, idx_i, idx_i] = -9999
        #     relationship_output = torch.exp(relationship_output)
        #     # relationship_output = torch.sigmoid(relationship_output)

        #     # relationship_output * subject score * object score
        #     entity_score_tensor = torch.tensor(entity_score_list, device=device, dtype=dtype)
        #     relationship_output = relationship_output * entity_score_tensor[None, :, None]
        #     relationship_output = relationship_output * entity_score_tensor[None, None, :]

        #     # find topk
        #     if relationship_output.shape[1] > 1:
        #         _, topk_indices = torch.topk(relationship_output.reshape([-1,]), k=20)

        #         # subject, object, cls
        #         for index in topk_indices:
        #             pred_cls = index // (relationship_output.shape[1] ** 2)
        #             index_subject_object = index % (relationship_output.shape[1] ** 2)
        #             pred_subject = index_subject_object // relationship_output.shape[1]
        #             pred_object = index_subject_object % relationship_output.shape[1]
        #             pred = [pred_subject.item(), pred_object.item(), pred_cls.item()]
        #             relation_res.append(pred)
            
        rl = dict(
            entityid_list=entityid_list,
            relation=relation_res,
        )

        res['rela_results'] = rl
        res['pan_results'] = res['pan_results'].detach().cpu().numpy()

        return [res]

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



def concat_relation_features(object_features, relation_tokens, target_edges):
    token_num = relation_tokens.shape[1]

    rel_labels = [t[:,2] if len(t)>0 else torch.zeros((0,1), dtype=torch.long).to(relation_tokens.device) for t in target_edges]
    # target_edges = [bs*N*2] object_id,subject_id
    target_edges = [t[:,:2] if len(t)>0 else torch.zeros((0,2), dtype=torch.long).to(relation_tokens.device) for t in target_edges]

    # target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, i in zip(target_edges, pos_assigned_gt_inds_list)]
    # target_edges = [torch.stack(t, 0) if len(t)>0 else torch.zeros((0,2), dtype=torch.long).to(relation_tokens.device) for t in target_edges]


    all_edge_lbl = []
    relation_features = []

    # total_edge = 0
    # total_fg = 0

    bs_size = []
    # neg_size = []
    # loop through each of batch to collect the edge and node
    for b_id, (filtered_edge, rel_label, object_feature, relation_token) in enumerate(zip(target_edges, rel_labels, object_features , relation_tokens)):
        # filtered_edge = [n*2]
        # rel_label = [n*1]
        
        if len(filtered_edge) == 0:
            bs_size.append(0)
            continue
        object_num = object_feature.shape[0]
        # find the -ve edges for training
        full_adj = torch.ones((object_num,object_num)) - torch.diag(torch.ones(object_num))
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
        # total_edge += all_edges_.shape[0]
        # total_fg += filtered_edge.shape[0]
        edge_labels = torch.cat((rel_label, torch.zeros(neg_edges.shape[0], dtype=torch.long).to(object_feature.device)), 0)
        #now permute all the combination
        idx_ = torch.randperm(all_edges_.shape[0])
        all_edges_ = all_edges_[idx_,:]
        edge_labels = edge_labels[idx_]
        all_edge_lbl.append(edge_labels)
        # get the valid predicted matching
        # pred_ids = indices[b_id][0]

        # joint_emb = object_token[b_id, :, :]
        token_relation_features = torch.cat((object_feature[all_edges_[:,0],:].repeat(token_num, 1), object_feature[all_edges_[:,1],:].repeat(token_num, 1), relation_token.unsqueeze(1).expand(token_num, all_edges_.shape[0], -1).reshape(-1, relation_token.shape[-1])),dim=1)
        # token_relation_features = []
        # for idx, edges in enumerate(all_edges_):
        #     # temp_token_relation_features = []
        #     token_relation_features.append(torch.cat((object_feature[edges[0],:].repeat(relation_token.shape[0],1),object_feature[edges[1],:].repeat(relation_token.shape[0],1),relation_token), 1).view(1, -1))
        # # print(token_relation_features[0].shape)
        # # print(len(token_relation_features))
        # relation_features.append(torch.cat(token_relation_features,0))
        relation_features.append(token_relation_features)
    relation_features = torch.cat(relation_features, 0)
    all_edge_lbl = torch.cat(all_edge_lbl, 0)

    return relation_features, all_edge_lbl, bs_size


def concat_relation_features_test(entity_embedding, relation_tokens):
    '''
    entity_embedding: (object_num, feature_size)
    relation_tokens:  (self.token_num, feature_size)
    '''
    object_num = entity_embedding.shape[0]
    token_num = relation_tokens.shape[0]

    # torch.cat((entity_embedding.repeat(object_num, 1).repeat(token_num, 1), entity_embedding.unsqueeze(1).repeat(token_num, 1), relation_tokens.unsqueeze(1).expend(token_num, object_num*object_num, -1)), 1)
    # relation_features = []
    # neg_idx = []
    # for i in range(object_num):
    #     neg_idx.append(int(i*object_num+i))
    #     for j in range(object_num):
    #         relation_features.append(torch.cat((entity_embedding[i, :].repeat(token_num, 1), entity_embedding[j, :].repeat(token_num, 1), relation_tokens), 1).view(1, -1))
    
    # relation_features = torch.cat(relation_features, 0)
    relation_features = torch.cat((entity_embedding.repeat(object_num, 1).repeat(token_num, 1), entity_embedding.unsqueeze(1).expand(object_num, object_num, -1).reshape(object_num*object_num, -1).repeat(token_num, 1), relation_tokens.unsqueeze(1).expand(token_num, object_num*object_num, -1).reshape(token_num*object_num*object_num, -1)), 1)
    neg_idx = []
    for i in range(object_num):
        neg_idx.append(int(i*object_num+i))

    return relation_features, neg_idx


