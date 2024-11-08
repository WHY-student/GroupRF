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

from util import CLASSES, PREDICATES, get_colormap, write_json, load_json, read_image


def get_model(cfg, ckp, transformers_model):
    cfg = mmcv.Config.fromfile(cfg)

    cfg['model']['type'] = 'Mask2FormerRelationForinfer'

    cfg['model']['relationship_head']['pretrained_transformers'] = transformers_model
    cfg['model']['relationship_head']['cache_dir'] = './'    
    if 'entity_length' in cfg['model']['relationship_head'] and cfg['model']['relationship_head']['entity_length'] > 1:
        cfg['model']['relationship_head']['entity_part_encoder'] = transformers_model

    model = init_detector(cfg, ckp)
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


def get_val_p(mode, cfg, ckp, psg_all_data_file, psg_tra_data_file, psg_val_data_file, img_dir, val_mode_output_dir, test_mode_output_dir, transformers_model):

    INSTANCE_OFFSET = 1000


    tra_id_list, val_id_list, test_id_list = get_tra_val_test_list(
        psg_tra_data_file=psg_tra_data_file,
        psg_val_data_file=psg_val_data_file,
    )
    psg_val_data = load_json(psg_all_data_file)
    # psg_train_data = load_json(psg_tra_data_file)['data'][:2000]

    model = get_model(cfg, ckp, transformers_model=transformers_model)

    cur_nb = -1
    nb_vis = None

    all_img_dicts = []
    all_recall_num = 0
    all_num = 0
    all_recall_scores = []
    all_mRecall_scores = []

    all_predicate_nums = [0]*56
    all_right_predicate_nums = [0]*56
    for d in tqdm(psg_val_data['data']):
        cur_nb += 1
        if nb_vis is not None and cur_nb > nb_vis:
            continue

        image_id = d['image_id']

        if image_id not in test_id_list:
            continue

        img_file = os.path.join(img_dir, d['file_name'])
        img = cv2.imread(img_file)
        img_res = inference_detector(model, img)
        objects_mask = []
        pan_results = img_res['pan_results']
        rela_results = img_res['rela_results']
        entityid_list = rela_results['entityid_list']
        relation = rela_results['relation']

        for instance_id in entityid_list:
            # instance_id == 133 background
            mask = pan_results == instance_id
            if instance_id == 133:
                continue
            r, g, b = random.choices(range(0, 255), k=3)
            
            mask = mask[..., None]
            mask = mask.astype(int)
            objects_mask.append(mask)
        
        # objects_mask = [ mask.detach().cpu().numpy() for mask in img_res['object_mask'] ]
        # objects_mask = img_res['object_mask'].detach().cpu().numpy()
        # ins_results = img_res['ins_results']
        label_pre = [instance_id % INSTANCE_OFFSET for instance_id in entityid_list]

        # 获取真实全景分割标注
        seg_map = read_image(os.path.join(img_dir, d["pan_seg_file_name"]), format="RGB")
        seg_map = rgb2id(seg_map)

        gt_relation = d["relations"]

        #get category ids
        gt_category_ids = []
        # get seperate masks
        gt_masks = []
        for i, s in enumerate(d["segments_info"]):
            gt_category_ids.append(s["category_id"])
            gt_masks.append(seg_map == s["id"])

        # 得到匹配关系后
        # IOU都高于0.5的则进行匹配替换
        # 不高的，则置为-1
        gt_dict = separator(objects_mask, gt_masks)
        new_category_ids = [-1]*len(label_pre)
        for x in gt_dict.keys():
            new_category_ids[gt_dict[x]]= gt_category_ids[x]
        
        # # 筛选出分割正确的那一批，找出target relation是否在relation中
        # recall_num = 0
        # for x in gt_relation:
        #     all_predicate_nums[x[2]] += 1
        #     try:
        #         new_relation = [gt_dict[x[0]], gt_dict[x[1]], x[2]]
        #         if new_relation in relation:
        #             all_right_predicate_nums[x[2]] += 1
        #             recall_num += 1
        #     except:
        #         pass
        one_predicate_nums = [0]*56
        one_right_predicate_nums = [0]*56
        
        # 筛选出分割正确的那一批，找出target relation是否在relation中
        recall_num = 0
        for x in gt_relation:
            all_predicate_nums[x[2]] += 1
            one_predicate_nums[x[2]] += 1
            try:
                new_relation = [gt_dict[x[0]], gt_dict[x[1]], x[2]]
                if new_relation in relation:
                    all_right_predicate_nums[x[2]] += 1
                    one_right_predicate_nums[x[2]] += 1
                    recall_num += 1
            except:
                pass
        one_result = []
        
        for i in range(len(one_predicate_nums)):
            if one_predicate_nums[i] == 0:
                continue
            else:
                one_result.append(one_right_predicate_nums[i] / one_predicate_nums[i])
        all_mRecall_scores.append(mean(one_result))

        gt_relation_nums = len(gt_relation)
        if gt_relation_nums == 0:
            continue
        all_recall_num += recall_num
        all_num += gt_relation_nums
        # print(recall_num / gt_relation_nums)
        all_recall_scores.append(recall_num / gt_relation_nums)


    result = []
    for i in range(len(all_predicate_nums)):
        if all_predicate_nums[i] == 0:
            continue
        else:
            result.append(all_right_predicate_nums[i] / all_predicate_nums[i])

    # print(result)
    print("recall@20={},mRecall={}, classM={}, mclassM={}".format(mean(all_recall_scores), all_recall_num / all_num, mean(result), mean(all_mRecall_scores)))
    # print("recall@20={},mRecall={}, classM={}".format(mean(all_recall_scores), all_recall_num / all_num, mean(result)))
    # print("recall@20={},mRecall={}".format(mean(all_recall_scores), mean(result)))


def separator(objects_mask,gt_masks):
    # GT matching
    gt_index_mgt = []
    for mask_id, mask_dict in enumerate(objects_mask):
        mask_dict = np.squeeze(mask_dict, axis=-1)
        # print(mask_dict.shape)
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
            mask_dict = np.squeeze(mask_dict, axis=-1)
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



if __name__ == '__main__':
    # get_tra_val_test_list()
    # get_test_p()
    get_val_p(
        mode='v0',
        cfg='configs/psg/v2.py',
        ckp='output/v2/latest.pth',
        val_mode_output_dir='submit/val_v5_latest',
        test_mode_output_dir='submit',

        psg_all_data_file='/root/autodl-tmp/dataset/psg/psg_test.json',
        psg_tra_data_file='/root/autodl-tmp/dataset/psg/psg_train_val.json',
        psg_val_data_file='/root/autodl-tmp/dataset/psg/psg_val_test.json',
        img_dir='/root/autodl-tmp/dataset/coco',
        transformers_model='/root/autodl-tmp/psg/mfpsg/checkpoints/chinese-roberta-wwm-ext',
    )
    # psg_dataset_file = load_json(psg_all_data_file)
    # # print('keys: ', list(psg_dataset_file.keys()))

    # psg_dataset_file = load_json('/root/autodl-tmp/dataset/psg/psg.json')
    # psg_thing_cats = psg_dataset_file['thing_classes']
    # psg_stuff_cats = psg_dataset_file['stuff_classes']
    # psg_obj_cats = psg_thing_cats + psg_stuff_cats

    # # psg_dataset = {d["image_id"]: d for d in psg_dataset_file['data']}
    # model = get_model('configs/psg/v5.py', 'output/v5/epoch_12.pth', transformers_model=None)
    # test_images = ["166-img.jpg",
    #         "285-img.jpg",
    #         "668-img.jpg",
    #         "1340-img.jpg",
    #         "1776-img.jpg",
    #         "2393-img.jpg",
    #         "2578-img.jpg"
    # ]
    # for file_name in tqdm(test_images):
    # # # img_id = random.choice()
    #     # data = psg_dataset[img_id]
    #     # file_name = data['file_name']

    #     show_result(
    #         # mode='v0',
    #         model = model,
    #         # cfg='configs/psg/v5.py',
    #         # ckp='output/v5/epoch_12.pth',
    #         # val_mode_output_dir='submit/val_v0_latest',
    #         # test_mode_output_dir='submit',

    #         # psg_all_data_file='/root/autodl-tmp/dataset/psg/psg.json',
    #         # psg_tra_data_file='/root/autodl-tmp/dataset/psg/psg_train_val.json',
    #         # psg_val_data_file='/root/autodl-tmp/dataset/psg/psg_val_test.json',
    #         img='/root/autodl-tmp/psg/mfpsg/test/image/'+file_name,
    #         # transformers_model=None,
    #         out_dir="/root/autodl-tmp/psg/mfpsg/test/result/{}/".format(file_name),
    #     )

    # landmark
    # best v1 ep30 31.3
    # v4 ep30 32.36
    # v5 ep30 31.94



































