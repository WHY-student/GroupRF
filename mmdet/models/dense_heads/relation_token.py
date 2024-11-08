from turtle import forward
from mmdet.models.builder import HEADS, build_loss
from mmcv.runner import BaseModule, ModuleList, auto_fp16, force_fp32
from mmcv.runner import load_checkpoint
from mmcv.cnn.bricks.transformer import build_positional_encoding

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from IPython import embed
from ..utils.group_token import GroupingBlock, GroupingLayer, TransformerLayer

import matplotlib.pyplot as plt
import seaborn as sns


# 考虑如何保存token为pt，然后如何去可视化tsne
@HEADS.register_module()
class rlnGroupToken(BaseModule):
    def __init__(
        self, 
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
    ):
        super().__init__()
        norm_layer = nn.LayerNorm
        num_layers = len(num_group_tokens)
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
        self.relation_embedding = Mlp(3*embed_dim,feed_forward, self.num_cls+1)
        self.relation_head = Mlp((self.num_cls+1)*self.token_num, feed_forward, self.num_cls+1)

    def forward_train(self,
        query_feat,
        pos_inds_list, 
        pos_assigned_gt_inds_list,
        img_metas
    ):
        temp_group_token = None
        group_token = query_feat
        for layer in self.layers:
            group_token, temp_group_token, _ = layer(group_token, temp_group_token)

        group_token = self.norm(group_token)


        object_feature_list, target_relation = self.get_embedding_relation(query_feat, pos_inds_list, pos_assigned_gt_inds_list, img_metas)

        relation_feature, all_edge_lbl, bs_size = concat_relation_features(object_feature_list, group_token, target_relation)

        relation_pred = self.relation_head(self.relation_embedding(relation_feature).reshape(relation_feature.shape[0], -1))

        return relation_pred, all_edge_lbl, bs_size
    
    def loss(self, relation_pred, all_edge_lbl):
        target_relation_tensor = all_edge_lbl
        # target_relation_tensor = torch.zeros_like(relation_pred)
        # for id, label in enumerate(all_edge_lbl):
        #     target_relation_tensor[id, label] = 1
        loss = self.multilabel_categorical_crossentropy(target_relation_tensor, relation_pred)
        loss = loss.mean() * 30

        return loss
    
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

    def simple_test(self,
        query_feat,
        target_keep,
        visual=False
    ):
        temp_group_token = None
        group_token = query_feat
        for layer in self.layers:
            group_token, temp_group_token, _ = layer(group_token, temp_group_token)
            # if return_attn and attention['soft'].shape[-1] == 100:
            #     show_attention = {'soft':attention['soft'][0][0].clone(), 'hard':attention['hard'][0][0].clone()}

        group_token = self.norm(group_token)


        group_token = group_token[0]
        # np.savetxt('token.txt', group_token.detach().cpu().numpy(), delimiter=',')
        #         # # print(group_token.shape)
        # cos_scores = [[0]*4 for _ in range(4)]
        # for i in range(4):
        #     for j in range(4):
        #         cos_scores[i][j] = round(F.cosine_similarity(group_token[i].unsqueeze(0),group_token[j].unsqueeze(0)).item(),2)
        # print(cos_scores)
        # sns.heatmap(cos_scores, annot=True, cmap="YlGnBu")
        # # 可选：添加轴标签
        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.savefig('token_cos.png')

        # exit(0)
        entity_embedding = query_feat[0][target_keep,:]

        relation_feature, neg_idx = concat_relation_features_test(entity_embedding, group_token)
        relation_pred = self.relation_head(self.relation_embedding(relation_feature).reshape(relation_feature.shape[0], -1))

        if visual:
            return relation_pred, neg_idx, relation_feature
        return relation_pred, neg_idx

@HEADS.register_module()
class rlnGroupTokenMultiHead(BaseModule):
    def __init__(
        self, 
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
        with_transformer=True,
        with_group_block=True,
        with_token = True,
        with_fuse = True,
    ):
        super().__init__()
        norm_layer = nn.LayerNorm
        num_layers = 3
        num_input_token = 100
        if with_token:
            self.group_token_prompt = nn.Parameter(torch.zeros(1, num_output_groups[-1], embed_dim))
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
                zero_init_group_token=group_projector is not None,
                with_transformer=with_transformer,
                with_group_block=with_group_block,
                )
            self.layers.append(layer)
            # if i_layer < self.num_layers - 1:
            num_input_token = num_output_token
        
        self.norm = norm_layer(embed_dim)
        
        self.num_cls = 56
        self.token_num = num_output_groups[-1]
        
        self.with_token = with_token
        self.with_fuse = with_fuse

        # 这里想模拟多头注意力，将每一个query token看成是一个头的一部分
        if self.with_fuse:
            # 每一个头的结果和其他头的结果做一个self.attention
            self.relation_fuse = TransformerLayer(d_model=3*embed_dim, nhead=1, dropout=0.1)

        self.relation_embedding = Mlp(3*embed_dim,feed_forward, self.num_cls+1)
        self.relation_head = Mlp((self.num_cls+1)*self.token_num, feed_forward, self.num_cls+1)

    def forward_train(self,
        query_feat,
        pos_inds_list, 
        pos_assigned_gt_inds_list,
        img_metas
    ):
        temp_group_token = None
        group_token = query_feat
        for layer in self.layers:
            group_token, temp_group_token, _ = layer(group_token, temp_group_token)
        group_token = self.norm(group_token)

        if self.with_token:
            group_token_prompt = self.group_token_prompt.expand(query_feat.shape[0],-1,-1)
            group_token = group_token + group_token_prompt


        object_feature_list, target_relation = self.get_embedding_relation(query_feat, pos_inds_list, pos_assigned_gt_inds_list, img_metas)

        relation_feature, target_relation_tensor, bs_size = concat_relation_features(object_feature_list, group_token, target_relation)
        # N * 8* 768
        if self.with_fuse:
            relation_feature = relation_feature.permute(1,0,2)

            relation_feature = self.relation_fuse(relation_feature, relation_feature).permute(1,0,2)

        relation_pred = self.relation_head(self.relation_embedding(relation_feature).reshape(relation_feature.shape[0], -1))
        # relation_pred = self.relation_head(relation_feature.reshape(relation_feature.shape[0], -1))

        return relation_pred, target_relation_tensor, bs_size
    
    def loss(self, relation_pred, target_relation_tensor):
        loss = self.multilabel_categorical_crossentropy(target_relation_tensor, relation_pred)
        loss = loss.mean() * 30

        return loss
    
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

    def simple_test(self,
        query_feat,
        target_keep,
        visual=False
    ):
        temp_group_token = None
        group_token = query_feat
        for layer in self.layers:
            group_token, temp_group_token, _ = layer(group_token, temp_group_token)
            # if return_attn and attention['soft'].shape[-1] == 100:
            #     show_attention = {'soft':attention['soft'][0][0].clone(), 'hard':attention['hard'][0][0].clone()}
        group_token = self.norm(group_token)
        
        if self.with_token:
            group_token_prompt = self.group_token_prompt.expand(query_feat.shape[0],-1,-1)
            group_token = group_token + group_token_prompt


        group_token = group_token[0]
        # np.savetxt('token.txt', group_token.detach().cpu().numpy(), delimiter=',')
        #         # # print(group_token.shape)
        # cos_scores = [[0]*4 for _ in range(4)]
        # for i in range(4):
        #     for j in range(4):
        #         cos_scores[i][j] = round(F.cosine_similarity(group_token[i].unsqueeze(0),group_token[j].unsqueeze(0)).item(),2)
        # print(cos_scores)
        # sns.heatmap(cos_scores, annot=True, cmap="YlGnBu")
        # # 可选：添加轴标签
        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.savefig('token_cos.png')

        # exit(0)
        entity_embedding = query_feat[0][target_keep,:]

        relation_feature, neg_idx = concat_relation_features_test(entity_embedding, group_token)
        if self.with_fuse:

            relation_feature = relation_feature.permute(1,0,2)
            relation_feature = self.relation_fuse(relation_feature, relation_feature).permute(1,0,2)

        tokens_scores = self.relation_embedding(relation_feature)

        relation_pred = self.relation_head(tokens_scores.reshape(relation_feature.shape[0], -1))
        
        if visual:
            return relation_pred, neg_idx, tokens_scores
        return relation_pred, neg_idx

@HEADS.register_module()
class rlnToken(BaseModule):
    def __init__(
        self, 
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
    ):
        super().__init__()
        norm_layer = nn.LayerNorm
        num_layers = 3
        num_input_token = 100
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # self.layers = nn.ModuleList()
        # for i_layer in range(num_layers-1):

        #     dim = int(embed_dim * embed_factors[i_layer])
        #     # downsample = None
        #     # if i_layer < self.num_layers - 1:
        #     out_dim = int(embed_dim * embed_factors[i_layer + 1])
        #     downsample = GroupingBlock(
        #         dim=dim,
        #         out_dim=out_dim,
        #         num_heads=num_heads[i_layer],
        #         num_group_token=num_group_tokens[i_layer],
        #         num_output_group=num_output_groups[i_layer],
        #         norm_layer=norm_layer,
        #         hard=hard_assignment,
        #         gumbel=hard_assignment)
        #     num_output_token = num_output_groups[i_layer]

        #     group_projector = None
        #     layer = GroupingLayer(
        #         dim=dim,
        #         num_input_token=num_input_token,
        #         depth=depths[i_layer],
        #         num_heads=num_heads[i_layer],
        #         num_group_token=num_group_tokens[i_layer],
        #         mlp_ratio=mlp_ratio,
        #         qkv_bias=qkv_bias,
        #         qk_scale=qk_scale,
        #         drop=drop_rate,
        #         attn_drop=attn_drop_rate,
        #         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
        #         norm_layer=norm_layer,
        #         downsample=downsample,
        #         use_checkpoint=use_checkpoint,
        #         group_projector=group_projector,
        #         # only zero init group token if we have a projection
        #         zero_init_group_token=group_projector is not None)
        #     self.layers.append(layer)
        #     # if i_layer < self.num_layers - 1:
        #     num_input_token = num_output_token
        
        self.norm = norm_layer(embed_dim)
        
        self.num_cls = 56
        self.token_num = num_output_groups[-1]
        self.relation_embedding = Mlp(3*embed_dim,feed_forward, self.num_cls+1)
        self.relation_head = Mlp((self.num_cls+1)*self.token_num, feed_forward, self.num_cls+1)

    def forward_train(self,
        query_feat,
        pos_inds_list, 
        pos_assigned_gt_inds_list,
        img_metas
    ):
        # temp_group_token = None
        # group_token = query_feat
        # for layer in self.layers:
        #     group_token, temp_group_token, _ = layer(group_token, temp_group_token)

        # group_token = self.norm(group_token)
        object_query = query_feat[:,:100,:]
        group_token = query_feat[:,100:,:]



        object_feature_list, target_relation = self.get_embedding_relation(object_query, pos_inds_list, pos_assigned_gt_inds_list, img_metas)

        relation_feature, all_edge_lbl, bs_size = concat_relation_features(object_feature_list, group_token, target_relation)

        relation_pred = self.relation_head(self.relation_embedding(relation_feature).reshape(relation_feature.shape[0], -1))

        return relation_pred, all_edge_lbl, bs_size
    
    def loss(self, relation_pred, all_edge_lbl):
        target_relation_tensor = all_edge_lbl
        # target_relation_tensor = torch.zeros_like(relation_pred)
        # for id, label in enumerate(all_edge_lbl):
        #     target_relation_tensor[id, label] = 1
        loss = self.multilabel_categorical_crossentropy(target_relation_tensor, relation_pred)
        loss = loss.mean() * 30

        return loss
    
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

    def simple_test(self,
        query_feat,
        target_keep,
        visual=False
    ):
        # temp_group_token = None
        # group_token = query_feat
        # for layer in self.layers:
        #     group_token, temp_group_token, _ = layer(group_token, temp_group_token)
        #     # if return_attn and attention['soft'].shape[-1] == 100:
        #     #     show_attention = {'soft':attention['soft'][0][0].clone(), 'hard':attention['hard'][0][0].clone()}

        # group_token = self.norm(group_token)


        # group_token = group_token[0]
        
        object_query = query_feat[0,:100,:]
        group_token = query_feat[0,100:,:]
        # np.savetxt('token.txt', group_token.detach().cpu().numpy(), delimiter=',')
        #         # # print(group_token.shape)
        # cos_scores = [[0]*4 for _ in range(4)]
        # for i in range(4):
        #     for j in range(4):
        #         cos_scores[i][j] = round(F.cosine_similarity(group_token[i].unsqueeze(0),group_token[j].unsqueeze(0)).item(),2)
        # print(cos_scores)
        # sns.heatmap(cos_scores, annot=True, cmap="YlGnBu")
        # # 可选：添加轴标签
        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.savefig('token_cos.png')

        # exit(0)
        entity_embedding = object_query[target_keep,:]

        relation_feature, neg_idx = concat_relation_features_test(entity_embedding, group_token)
        relation_pred = self.relation_head(self.relation_embedding(relation_feature).reshape(relation_feature.shape[0], -1))

        if visual:
            return relation_pred, neg_idx, relation_feature
        return relation_pred, neg_idx


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
    device = relation_tokens.device
    token_num = relation_tokens.shape[1]
    # 取tagert 对应的relation 的target的数组
    target_rels = []
    relation_features = []
    bs_size = []
    for target_edge, object_feature, relation_token in zip(target_edges, object_features, relation_tokens):
        if len(target_edge) == 0:
            bs_size.append(0)
            continue

        object_num = object_feature.shape[0]
        target_rel = torch.zeros((object_num, object_num, 57)).to(device)
        target_rel[target_edge[:,0], target_edge[:,1], target_edge[:,2]] = 1

        # 找出对应的物体对
        filtered_edge = torch.unique(target_edge[:,:2], dim=0)

        # 找负样本
        full_adj = torch.ones((object_num,object_num)) - torch.diag(torch.ones(object_num))
        full_adj[filtered_edge[:,0], filtered_edge[:,1]] = 0
        neg_edges = torch.nonzero(full_adj).to(filtered_edge.device)
        # neg_edges all relation that shouldn't be predicted

        # restrict unbalance in the +ve/-ve edge
        if filtered_edge.shape[0]>30:
            idx_ = torch.randperm(filtered_edge.shape[0])[:30]
            filtered_edge = filtered_edge[idx_,:]
            # rel_label = rel_label[idx_]
        bs_size.append(filtered_edge.shape[0])

        # check whether the number of -ve edges are within limit
        # limit_neg_num = filtered_edge.shape[0] * 5
        limit_neg_num = filtered_edge.shape[0] * 20
        if neg_edges.shape[0]>= limit_neg_num:# random sample -ve edge
            idx_ = torch.randperm(neg_edges.shape[0])[:limit_neg_num]
            neg_edges = neg_edges[idx_,:]
        
        # 负样本的真值
        target_rel[neg_edges[:,0], neg_edges[:,1], 0] = 1
        # 取出来了所有label
        all_edges_ = torch.cat((filtered_edge, neg_edges), 0)
        # 取对应的rel
        target_rel = target_rel[all_edges_[:,0], all_edges_[:,1], :]

        #now permute all the combination
        idx_ = torch.randperm(all_edges_.shape[0])
        all_edges_ = all_edges_[idx_, : ]
        edge_rels = target_rel[idx_, :]

        target_rels.append(edge_rels)
        # get the valid predicted matching
        # pred_ids = indices[b_id][0]

        # joint_emb = object_token[b_id, :, :]
        n = all_edges_.shape[0]
        token_relation_features = torch.cat((object_feature[all_edges_[:,0],:].unsqueeze(1).expand(n, token_num, -1), object_feature[all_edges_[:,1],:].unsqueeze(1).expand(n, token_num, -1), relation_token.unsqueeze(0).expand(n, token_num, -1)),dim=2)
        relation_features.append(token_relation_features)

    if len(relation_features) != 0:
        relation_features = torch.cat(relation_features, 0)
        all_edge_lbl = torch.cat(target_rels, 0)
    else:
        relation_features = torch.zeros((1,8,768)).to(device)
        all_edge_lbl = torch.zeros((1,57)).to(device).int()

    return relation_features, all_edge_lbl, bs_size


def concat_relation_features_test(entity_embedding, relation_tokens):
    '''
    entity_embedding: (object_num, feature_size)
    relation_tokens:  (self.token_num, feature_size)
    '''
    object_num = entity_embedding.shape[0]
    token_num = relation_tokens.shape[0]
    all_edges_ = []
    neg_idx = []
    for i in range(object_num):
        neg_idx.append(int(i*object_num+i))
        for j in range(object_num):
            all_edges_.append([i, j])
    n = len(all_edges_)
    all_edges_ = torch.tensor(all_edges_, dtype=torch.long)
    # print(all_edges_.shape)

    # torch.cat((entity_embedding.repeat(object_num, 1).repeat(token_num, 1), entity_embedding.unsqueeze(1).repeat(token_num, 1), relation_tokens.unsqueeze(1).expend(token_num, object_num*object_num, -1)), 1)
    # relation_features = []
    # neg_idx = []
    # for i in range(object_num):
    #     neg_idx.append(int(i*object_num+i))
    #     for j in range(object_num):
    #         relation_features.append(torch.cat((entity_embedding[i, :].repeat(token_num, 1), entity_embedding[j, :].repeat(token_num, 1), relation_tokens), 1).view(1, -1))
    
    # relation_features = torch.cat(relation_features, 0)
    relation_features = torch.cat((entity_embedding[all_edges_[:,0],:].unsqueeze(1).expand(n, token_num, -1), entity_embedding[all_edges_[:,1],:].unsqueeze(1).expand(n, token_num, -1), relation_tokens.unsqueeze(0).expand(n, token_num, -1)),dim=2)
    # relation_features = torch.cat((entity_embedding.repeat(object_num, 1).unsqueeze(1).expand(object_num*object_num, token_num, -1), entity_embedding.unsqueeze(1).expand(object_num, object_num, -1).reshape(object_num*object_num, -1).unsqueeze(1).expand(object_num*object_num, token_num, -1), relation_tokens.unsqueeze(0).expand(object_num*object_num, token_num, -1)), 2)
    
    return relation_features, neg_idx

