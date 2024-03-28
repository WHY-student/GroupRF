from typing import Tuple
from mmdet.apis import init_detector, inference_detector
import cv2
import numpy as np
import os
import random
from panopticapi.utils import rgb2id, id2rgb
import mmcv
from tqdm import tqdm
from PIL import Image

from IPython import embed
import json
import os.path as osp

from statistics import mean
import PIL

from detectron2.utils.visualizer import VisImage, Visualizer
from mmdet.evaluation import sgg_evaluation

from util import CLASSES, PREDICATES, get_colormap, write_json, load_json, read_image, Result, get_ann_info

from mmcv import Config, DictAction

from mmdet.models import build_detector

import torch

import time

from mmdet.path import coco_root


def test_matrics(
        cfg,
        ckp,
        mode='v6',
        psg_test_data_file='dataset/psg/psg_test.json',
        img_prefix = coco_root,
        multiple_preds=False,
        iou_thrs=0.5,
        detection_method='pan_seg',
        transformers_model='checkpoints/chinese-roberta-wwm-ext',
    ):
    # psg_test_data_file = os.path.join(DATASETS_ROOT, psg_test_data_file)
    # img_prefix = os.path.join(DATASETS_ROOT, img_prefix)
    INSTANCE_OFFSET = 1000
    print('\nLoading testing groundtruth...\n')
    # prog_bar = mmcv.ProgressBar(len(self))
    gt_results = []
    results = []
    psg_test_data = load_json(psg_test_data_file)
    model = get_model(cfg, ckp, mode, transformers_model=transformers_model)
    for d in tqdm(psg_test_data['data']):
        img_file = os.path.join(img_prefix, d['file_name'])
        img = cv2.imread(img_file)
        img_res = inference_detector(model, img)
        pan_results = img_res['pan_results']
        rela_results = img_res['rela_results']
        relation = rela_results['relation']
        entityid_list = rela_results['entityid_list']
        label_pre = [instance_id % INSTANCE_OFFSET for instance_id in entityid_list]

        if mode=='v0':
            objects_mask = []
            for instance_id in entityid_list:
                # instance_id == 133 background
                mask = pan_results == instance_id
                if instance_id == 133:
                    continue
                mask = mask[..., None]
                mask = np.squeeze(mask, axis=-1)
                objects_mask.append(mask)
        else:
            objects_mask = [ mask.detach().cpu().numpy() for mask in img_res['object_mask'] ]


        results.append(
            Result(
                labels=label_pre,
                rels=relation,
                masks=objects_mask,
            ))
        # NOTE: Change to object class labels 1-index here
        # ann['labels'] += 1

        # load gt pan_seg masks
        ann = get_ann_info(d)
        segment_info = ann['masks']
        gt_img = read_image(img_prefix + '/' + ann['seg_map'],
                            format='RGB')
        gt_img = gt_img.copy()  # (H, W, 3)

        seg_map = rgb2id(gt_img)

        # get separate masks
        gt_masks = []
        label_id = []
        labels_coco = []
        for _, s in enumerate(segment_info):
            label = CLASSES[s['category']]
            labels_coco.append(label)
            label_id.append(s['category'])
            gt_masks.append(seg_map == s['id'])
        # load gt pan seg masks done

        gt_results.append(
            Result(
                labels=label_id,
                rels=ann['rels'],
                masks=gt_masks,
            ))
        # prog_bar.update()

    print('\n')

    return sgg_evaluation(
        ['sgdet'],
        groundtruths=gt_results,
        predictions=results,
        iou_thrs=iou_thrs,
        logger=None,
        ind_to_predicates=['__background__'] + PREDICATES,
        multiple_preds=multiple_preds,
        # predicate_freq=self.predicate_freq,
        nogc_thres_num=None,
        detection_method=detection_method,
    )

def get_model(cfg, ckp, mode, transformers_model, device="cuda:1"):

    cfg = mmcv.Config.fromfile(cfg)
    if mode=='v6':
        cfg['model']['type'] = 'Mask2FormerVitForinfer'
    else:
        cfg['model']['type'] = 'Mask2FormerRelationForinfer'
        cfg['model']['relationship_head']['pretrained_transformers'] = transformers_model
        cfg['model']['relationship_head']['cache_dir'] = './'    
        if 'entity_length' in cfg['model']['relationship_head'] and cfg['model']['relationship_head']['entity_length'] > 1:
            cfg['model']['relationship_head']['entity_part_encoder'] = transformers_model

    # config = cfg

    # if 'pretrained' in config.model:
    #     config.model.pretrained = None
    # elif 'init_cfg' in config.model.backbone:
    #     config.model.backbone.init_cfg = None
    # config.model.train_cfg = None
    # model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    # # if checkpoint is not None:
    # checkpoint = load_checkpoint(model, ckp, map_location='cpu')
    # save_checkpoint(model, filename="output/v6/epoch13.pth", meta=checkpoint["meta"])
    # exit(0)
    model = init_detector(cfg, ckp, device=device)
    return model


def get_tra_val_test_list(psg_tra_data_file, psg_val_data_file):
    psg_tra_data = load_json(psg_tra_data_file)
    psg_val_data = load_json(psg_val_data_file)

    tra_id_list = []
    val_id_list = []
    test_id_list = []

    for d in psg_tra_data['data']:
        if d['image_id'] in psg_tra_data['test_image_ids']:
            val_id_list.append(d['image_id'])
        else:
            tra_id_list.append(d['image_id'])

    for d in psg_val_data['data']:
        test_id_list.append(d['image_id'])
    
    tra_id_list = np.array(tra_id_list)
    val_id_list = np.array(val_id_list)
    test_id_list = np.array(test_id_list)
    print('tra', len(tra_id_list))
    print('val', len(val_id_list))
    print('test', len(test_id_list))
    
    return tra_id_list, val_id_list, test_id_list

def get_val_p(
        cfg, 
        ckp, 
        mode="v0", 
        psg_tra_data_file='dataset/psg/psg_train_val.json', 
        psg_val_data_file='dataset/psg/psg_val_test.json', 
        psg_test_data_file='dataset/psg/psg_test.json',
        img_dir= coco_root,
        val_mode_output_dir='submit/val_v2_latest',
        test_mode_output_dir='submit',
        transformers_model='./checkpoints/chinese-roberta-wwm-ext'
        ):
    # if mode == 'val':
    jpg_output_dir = os.path.join(val_mode_output_dir, 'submission/panseg')
    json_output_dir = os.path.join(val_mode_output_dir, 'submission')
    # else:
    #     jpg_output_dir = os.path.join(test_mode_output_dir, mode,'submission/panseg')
    #     jpg_output_dir = os.path.join(test_mode_output_dir, mode,'submission')

    os.makedirs(jpg_output_dir, exist_ok=True)

    INSTANCE_OFFSET = 1000


    tra_id_list, val_id_list, test_id_list = get_tra_val_test_list(
        psg_tra_data_file=psg_tra_data_file,
        psg_val_data_file=psg_val_data_file,
    )
    psg_val_data = load_json(psg_test_data_file)

    model = get_model(cfg, ckp, mode, transformers_model=transformers_model)
    # cfg = Config.fromfile(cfg)
    # model = build_detector(
    #     cfg.model,
    #     train_cfg=cfg.get('train_cfg'),
    #     test_cfg=cfg.get('test_cfg'))
    
    cur_nb = -1
    nb_vis = None

    all_img_dicts = []
    for d in tqdm(psg_val_data['data']):
        cur_nb += 1
        if nb_vis is not None and cur_nb > nb_vis:
            continue

        # image_id = d['image_id']

        # if image_id not in test_id_list:
        #     continue

        img_file = os.path.join(img_dir, d['file_name'])
        img = cv2.imread(img_file)
        img_res = inference_detector(model, img)

        pan_results = img_res['pan_results']
        # ins_results = img_res['ins_results']
        rela_results = img_res['rela_results']
        entityid_list = rela_results['entityid_list']
        relation = rela_results['relation']


        img_output = np.zeros_like(img)
        segments_info = []
        for instance_id in entityid_list:
            # instance_id == 133 background
            mask = pan_results == instance_id
            if instance_id == 133:
                continue
            r, g, b = random.choices(range(0, 255), k=3)
            
            mask = mask[..., None]
            mask = mask.astype(int)
            coloring_mask = np.concatenate([mask]*3, axis=-1)
            color = np.array([b,g,r]).reshape([1,1,3])
            coloring_mask = coloring_mask * color
            # coloring_mask = np.concatenate([mask[..., None]*1]*3, axis=-1)
            # for j, color in enumerate([b, g, r]):
            #     coloring_mask[:, :, j] = coloring_mask[:, :, j] * color
            img_output = img_output + coloring_mask
            idx_class = instance_id % INSTANCE_OFFSET + 1
            segment = dict(category_id=int(idx_class), id=rgb2id((r, g, b)))
            segments_info.append(segment)

        img_output = img_output.astype(np.uint8)
        # mask = np.sum(img_output, axis=-1) > 0
        # img_output_2 = np.copy(img)
        # img_output_2[mask] = img_output_2[mask] * 0.5 + img_output[mask] * 0.5
        # img_output = np.concatenate([img_output_2, img_output], axis=1)
        cv2.imwrite(f'{jpg_output_dir}/{cur_nb}.png', img_output)

        if len(relation) == 0:
            relation = [[0, 0, 0]]
        if len(segments_info) == 0:
            r, g, b = random.choices(range(0, 255), k=3)
            segments_info = [dict(category_id=1, id=rgb2id((r, g, b)))]

        single_result_dict = dict(
            # image_id=image_id,
            relations=[[s, o, r + 1] for s, o, r in relation],
            segments_info=segments_info,
            pan_seg_file_name='%d.png' % cur_nb,
        )
        all_img_dicts.append(single_result_dict)
    
    # write_json(all_img_dicts, f'{json_output_dir}/relation.json')
    with open(f'{json_output_dir}/relation.json', 'w') as outfile:
        json.dump(all_img_dicts, outfile, default=str)


# def get_val_p(recallK, mode, cfg, ckp, psg_all_data_file, psg_tra_data_file, psg_val_data_file, img_dir, val_mode_output_dir, test_mode_output_dir, transformers_model):


#     INSTANCE_OFFSET = 1000

#     tra_id_list, val_id_list, test_id_list = get_tra_val_test_list(
#         psg_tra_data_file=psg_tra_data_file,
#         psg_val_data_file=psg_val_data_file,
#     )
#     psg_val_data = load_json(psg_all_data_file)
#     # psg_train_data = load_json(psg_tra_data_file)['data'][:2000]

#     model = get_model(cfg, ckp, transformers_model=transformers_model)

#     cur_nb = -1
#     nb_vis = None

#     all_img_dicts = []
#     all_recall_num = 0
#     all_num = 0
#     all_recall_scores = []

#     all_mRecall_scores = []

#     all_predicate_nums = [0]*56
#     all_right_predicate_nums = [0]*56
#     for d in tqdm(psg_val_data['data']):
#         cur_nb += 1
#         if nb_vis is not None and cur_nb > nb_vis:
#             continue

#         image_id = d['image_id']

#         # if image_id not in test_id_list:
#         #     continue

#         img_file = os.path.join(img_dir, d['file_name'])
#         img = cv2.imread(img_file)
#         img_res = inference_detector(model, img)

#         pan_results = img_res['pan_results']
#         objects_mask = [ mask.detach().cpu().numpy() for mask in img_res['object_mask'] ]
#         # objects_mask = img_res['object_mask'].detach().cpu().numpy()
#         # ins_results = img_res['ins_results']
#         rela_results = img_res['rela_results']
#         entityid_list = rela_results['entityid_list']
#         relation = rela_results['relation']
#         label_pre = [instance_id % INSTANCE_OFFSET for instance_id in entityid_list]

#         # 获取真实全景分割标注
#         seg_map = read_image(os.path.join(img_dir, d["pan_seg_file_name"]), format="RGB")
#         seg_map = rgb2id(seg_map)

#         gt_relation = d["relations"]

#         #get category ids
#         gt_category_ids = []
#         # get seperate masks
#         gt_masks = []
#         for i, s in enumerate(d["segments_info"]):
#             gt_category_ids.append(s["category_id"])
#             gt_masks.append(seg_map == s["id"])

#         # 得到匹配关系后
#         # IOU都高于0.5的则进行匹配替换
#         # 不高的，则置为-1
#         gt_dict = separator(objects_mask, gt_masks)
#         new_category_ids = [-1]*len(label_pre)
#         for x in gt_dict.keys():
#             new_category_ids[gt_dict[x]]= gt_category_ids[x]
#         one_predicate_nums = [0]*56
#         one_right_predicate_nums = [0]*56
        
#         # 筛选出分割正确的那一批，找出target relation是否在relation中
#         recall_num = 0
#         for x in gt_relation:
#             all_predicate_nums[x[2]] += 1
#             one_predicate_nums[x[2]] += 1
#             try:
#                 new_relation = [gt_dict[x[0]], gt_dict[x[1]], x[2]]
#                 if new_relation in relation:
#                     all_right_predicate_nums[x[2]] += 1
#                     one_right_predicate_nums[x[2]] += 1
#                     recall_num += 1
#             except:
#                 pass
#         one_result = []
        
#         for i in range(len(one_predicate_nums)):
#             if one_predicate_nums[i] == 0:
#                 continue
#             else:
#                 one_result.append(one_right_predicate_nums[i] / one_predicate_nums[i])
#         all_mRecall_scores.append(mean(one_result))

#         gt_relation_nums = len(gt_relation)
#         if gt_relation_nums == 0:
#             continue
#         all_recall_num += recall_num
#         all_num += gt_relation_nums
#         # print(recall_num / gt_relation_nums)
#         all_recall_scores.append(recall_num / gt_relation_nums)


#     result = []
#     for i in range(len(all_predicate_nums)):
#         if all_predicate_nums[i] == 0:
#             continue
#         else:
#             result.append(all_right_predicate_nums[i] / all_predicate_nums[i])


#     # print(result)
#     num_params = sum(p.numel() for p in model.parameters())
#     print("param=", num_params / 1e6)
#     # print("recall@20={},mRecall={}, classM={}, mclassM={}".format(mean(all_recall_scores), all_recall_num / all_num, mean(result), mean(all_mRecall_scores)))
#     print("recall@20={},mRecall={}, classM={}".format(mean(all_recall_scores), all_recall_num / all_num, mean(result)))


def separator(objects_mask,gt_masks):
    # GT matching
    gt_index_mgt = []
    for mask_id, mask_dict in enumerate(objects_mask):
        max_iou = 0
        for gt_mask_id, gt_mask in enumerate(gt_masks):
            current_iou = iou(gt_mask, mask_dict)
            if current_iou > max_iou and current_iou > 0.5:
                max_iou = current_iou
                gt_index_mgt.append({gt_mask_id: mask_id})

    gt_index_gtm = []
    for gt_mask_id, gt_mask in enumerate(gt_masks):
        max_iou = 0
        for mask_id, mask_dict in enumerate(objects_mask):
            current_iou = iou(gt_mask, mask_dict)
            if current_iou > max_iou and current_iou > 0.5:
                max_iou = current_iou
                gt_index_gtm.append({gt_mask_id: mask_id})

    # common = [x for x in gt_index_gtm if x in gt_index_mgt]
    common = gt_index_gtm
    # convert gt_index to a dictionary
    gt_dict = {}
    for d in common:
        gt_dict.update(d)
    return gt_dict

def iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def show_result(model,
                img,
                is_one_stage=False,
                num_rel=20,
                show=False,
                out_dir="output/r5/",
                out_file=None):

    # Load image
    img = mmcv.imread(img)
    img_res = inference_detector(model, img)
    pan_results = img_res['pan_results']
    # ins_results = img_res['ins_results']
    rela_results = img_res['rela_results']
    entityid_list = rela_results['entityid_list']
    relation = rela_results['relation']

    INSTANCE_OFFSET = 1000

    # 将得到的结果处理为id和mask0-1矩阵对应的关系
    objects_mask = []
    label_pre = []
    for instance_id in entityid_list:
        # instance_id == 133 background
        mask = pan_results == instance_id
        if instance_id == 133:
            continue
        objects_mask.append(mask)
        label_pre.append(instance_id % INSTANCE_OFFSET)

    img = img.copy()  # (H, W, 3)
    img_h, img_w = img.shape[:-1]

    # Decrease contrast
    img = PIL.Image.fromarray(img)
    converter = PIL.ImageEnhance.Color(img)
    img = converter.enhance(0.01)
    if out_file is not None:
        mmcv.imwrite(np.asarray(img), 'bw'+out_file)

    # Draw masks
    # pan_results = result.pan_results

    ids = np.unique(pan_results)[::-1]
    num_classes = 133
    legal_indices = (ids != num_classes)  # for VOID label
    ids = ids[legal_indices]

    # Get predicted labels
    labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
    labels = [CLASSES[l] for l in labels]

    # For psgtr
    rel_obj_labels = [ entityId % INSTANCE_OFFSET for entityId in entityid_list]
    rel_obj_labels = [CLASSES[l] for l in rel_obj_labels]

    # (N_m, H, W)
    # segms = pan_results[None] == ids[:, None, None]
    # # Resize predicted masks
    # segms = [
    #     mmcv.image.imresize(m.astype(float), (img_w, img_h)) for m in segms
    # ]
    segms = objects_mask
    # One stage segmentation
    masks = objects_mask
 
    # Choose colors for each instance in coco
    colormap_coco = get_colormap(
        len(masks)) if is_one_stage else get_colormap(len(segms))
    colormap_coco = (np.array(colormap_coco) / 255).tolist()

    # Viualize masks
    viz = Visualizer(img)
    viz.overlay_instances(
        labels=rel_obj_labels if is_one_stage else labels,
        masks=masks if is_one_stage else segms,
        assigned_colors=colormap_coco,
    )
    viz_img = viz.get_output().get_image()
    if out_file is not None:
        mmcv.imwrite(viz_img, out_file)

    # Draw relations

    # # Filter out relations
    # n_rel_topk = num_rel
    # # Exclude background class
    # rel_dists = result.rel_dists[:, 1:]
    # # rel_dists = result.rel_dists
    # rel_scores = rel_dists.max(1)
    # # rel_scores = result.triplet_scores
    # # Extract relations with top scores
    # rel_topk_idx = np.argpartition(rel_scores, -n_rel_topk)[-n_rel_topk:]
    # rel_labels_topk = rel_dists[rel_topk_idx].argmax(1)
    # rel_pair_idxes_topk = result.rel_pair_idxes[rel_topk_idx]
    # relations = np.concatenate(
    #     [rel_pair_idxes_topk, rel_labels_topk[..., None]], axis=1)
    n_rels = len(relation)

    top_padding = 20
    bottom_padding = 20
    left_padding = 20
    text_size = 10
    text_padding = 5
    text_height = text_size + 2 * text_padding
    row_padding = 10
    height = (top_padding + bottom_padding + n_rels *
              (text_height + row_padding) - row_padding)
    width = img_w
    curr_x = left_padding
    curr_y = top_padding

    # # Adjust colormaps
    # colormap_coco = [adjust_text_color(c, viz) for c in colormap_coco]
    viz_graph = VisImage(np.full((height, width, 3), 255))

    for i, r in enumerate(relation):
        s_idx, o_idx, rel_id = r
        s_label = rel_obj_labels[s_idx]
        o_label = rel_obj_labels[o_idx]
        rel_label = PREDICATES[rel_id]
        viz = Visualizer(img)
        viz.overlay_instances(
            labels=[s_label, o_label],
            masks=[masks[s_idx], masks[o_idx]],
            assigned_colors=[colormap_coco[s_idx], colormap_coco[o_idx]],
        )
        viz_masked_img = viz.get_output().get_image()

        viz_graph = VisImage(np.full((40, width, 3), 255))
        curr_x = 2
        curr_y = 2
        text_size = 25
        text_padding = 20
        font = 36
        text_width = draw_text(
            viz_img=viz_graph,
            text=s_label,
            x=curr_x,
            y=curr_y,
            color=colormap_coco[s_idx],
            size=text_size,
            padding=text_padding,
            font=font,
        )
        curr_x += text_width
        # Draw relation text
        text_width = draw_text(
            viz_img=viz_graph,
            text=rel_label,
            x=curr_x,
            y=curr_y,
            size=text_size,
            padding=text_padding,
            box_color='gainsboro',
            font=font,
        )
        curr_x += text_width

        # Draw object text
        text_width = draw_text(
            viz_img=viz_graph,
            text=o_label,
            x=curr_x,
            y=curr_y,
            color=colormap_coco[o_idx],
            size=text_size,
            padding=text_padding,
            font=font,
        )
        output_viz_graph = np.vstack([viz_masked_img, viz_graph.get_image()])
        # if out_file is not None:
        mmcv.imwrite(output_viz_graph, osp.join(
            out_dir, '{}.jpg'.format(i)))


def draw_text(
    viz_img: VisImage = None,
    text: str = None,
    x: float = None,
    y: float = None,
    color: Tuple[float, float, float] = [0, 0, 0],
    size: float = 10,
    padding: float = 5,
    box_color: str = 'black',
    font: str = None,
) -> float:
    text_obj = viz_img.ax.text(
        x,
        y,
        text,
        size=size,
        # family="sans-serif",
        bbox={
            'facecolor': box_color,
            'alpha': 0.8,
            'pad': padding,
            'edgecolor': 'none',
        },
        verticalalignment='top',
        horizontalalignment='left',
        color=color,
        zorder=10,
        rotation=0,
    )
    viz_img.get_image()
    text_dims = text_obj.get_bbox_patch().get_extents()

    return text_dims.width


def show_gt(img,
                out_dir="output/v0/",
                out_file=None):

    # Load image
    img = mmcv.imread(img)
    img_res = {}
    pan_results = img_res['pan_results']
    # ins_results = img_res['ins_results']
    rela_results = img_res['rela_results']
    entityid_list = rela_results['entityid_list']
    relation = rela_results['relation']

    INSTANCE_OFFSET = 1000

    # 将得到的结果处理为id和mask0-1矩阵对应的关系
    objects_mask = []
    label_pre = []
    for instance_id in entityid_list:
        # instance_id == 133 background
        mask = pan_results == instance_id
        if instance_id == 133:
            continue
        objects_mask.append(mask)
        label_pre.append(instance_id % INSTANCE_OFFSET)

    img = img.copy()  # (H, W, 3)
    img_h, img_w = img.shape[:-1]

    # Decrease contrast
    img = PIL.Image.fromarray(img)
    converter = PIL.ImageEnhance.Color(img)
    img = converter.enhance(0.01)
    if out_file is not None:
        mmcv.imwrite(np.asarray(img), 'bw'+out_file)

    # Draw masks
    # pan_results = result.pan_results

    ids = np.unique(pan_results)[::-1]
    num_classes = 133
    legal_indices = (ids != num_classes)  # for VOID label
    ids = ids[legal_indices]

    # Get predicted labels
    labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
    labels = [CLASSES[l] for l in labels]

    # For psgtr
    rel_obj_labels = [ entityId % INSTANCE_OFFSET for entityId in entityid_list]
    rel_obj_labels = [CLASSES[l] for l in rel_obj_labels]

    # (N_m, H, W)
    # segms = pan_results[None] == ids[:, None, None]
    # # Resize predicted masks
    # segms = [
    #     mmcv.image.imresize(m.astype(float), (img_w, img_h)) for m in segms
    # ]
    segms = objects_mask
    # One stage segmentation
    masks = objects_mask
 
    # Choose colors for each instance in coco
    colormap_coco =  get_colormap(len(segms))
    colormap_coco = (np.array(colormap_coco) / 255).tolist()

    # Viualize masks
    viz = Visualizer(img)
    viz.overlay_instances(
        labels,
        segms,
        assigned_colors=colormap_coco,
    )
    viz_img = viz.get_output().get_image()
    if out_file is not None:
        mmcv.imwrite(viz_img, out_file)

    n_rels = len(relation)

    top_padding = 20
    bottom_padding = 20
    left_padding = 20
    text_size = 10
    text_padding = 5
    text_height = text_size + 2 * text_padding
    row_padding = 10
    height = (top_padding + bottom_padding + n_rels *
              (text_height + row_padding) - row_padding)
    width = img_w
    curr_x = left_padding
    curr_y = top_padding

    # # Adjust colormaps
    # colormap_coco = [adjust_text_color(c, viz) for c in colormap_coco]
    viz_graph = VisImage(np.full((height, width, 3), 255))

    for i, r in enumerate(relation):
        s_idx, o_idx, rel_id = r
        s_label = rel_obj_labels[s_idx]
        o_label = rel_obj_labels[o_idx]
        rel_label = PREDICATES[rel_id]
        viz = Visualizer(img)
        viz.overlay_instances(
            labels=[s_label, o_label],
            masks=[masks[s_idx], masks[o_idx]],
            assigned_colors=[colormap_coco[s_idx], colormap_coco[o_idx]],
        )
        viz_masked_img = viz.get_output().get_image()

        viz_graph = VisImage(np.full((40, width, 3), 255))
        curr_x = 2
        curr_y = 2
        text_size = 25
        text_padding = 20
        font = 36
        text_width = draw_text(
            viz_img=viz_graph,
            text=s_label,
            x=curr_x,
            y=curr_y,
            color=colormap_coco[s_idx],
            size=text_size,
            padding=text_padding,
            font=font,
        )
        curr_x += text_width
        # Draw relation text
        text_width = draw_text(
            viz_img=viz_graph,
            text=rel_label,
            x=curr_x,
            y=curr_y,
            size=text_size,
            padding=text_padding,
            box_color='gainsboro',
            font=font,
        )
        curr_x += text_width

        # Draw object text
        text_width = draw_text(
            viz_img=viz_graph,
            text=o_label,
            x=curr_x,
            y=curr_y,
            color=colormap_coco[o_idx],
            size=text_size,
            padding=text_padding,
            font=font,
        )
        output_viz_graph = np.vstack([viz_masked_img, viz_graph.get_image()])
        # if out_file is not None:
        mmcv.imwrite(output_viz_graph, osp.join(
            out_dir, '{}.jpg'.format(i)))

def testFPS(config):
    # ori_shape = (3, h, w)
    # divisor = args.size_divisor
    # if divisor > 0:
    #     h = int(np.ceil(h / divisor)) * divisor
    #     w = int(np.ceil(w / divisor)) * divisor

    cfg = Config.fromfile(config)
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.cuda()
    model.eval()

    input_shape = (3, 640, 640)
    batch = torch.ones(()).new_empty((1, *input_shape), dtype=next(model.parameters()).dtype, device=next(model.parameters()).device)

    start_time = time.time()
    for i in range(100):
        _ = model.forward_dummy(batch)
    end_time = time.time()
    print(1/((end_time - start_time)/100))


if __name__ == '__main__':
    
    cfg='configs/psg/v6.py'
    ckp=''
    mode='v6'
    # testFPS(cfg)
    # exit(0)

    # 测试
    test_matrics(
        cfg = cfg,
        ckp = ckp,
        mode = mode
    )
    
    # 获取提交的submit
    # get_val_p(
    #     cfg = cfg,
    #     ckp = ckp,
    #     mode = mode,
    #     val_mode_output_dir='submit/val_none_latest',
    # )

































