import mmcv
from mmdet.apis import init_detector, inference_detector
import torch
from tqdm import tqdm
import time
import cv2
from torchvision import transforms
from PIL import Image
import PIL
from util import CLASSES, PREDICATES, get_colormap, load_json, read_image, show_relations
import os
from panopticapi.utils import rgb2id, id2rgb

import numpy as np
from detectron2.utils.visualizer import VisImage, Visualizer

import matplotlib.pyplot as plt
import seaborn as sns

def get_model(cfg, ckp, transformers_model):
    # print(cfg)
    # print(type(cfg))
    cfg = mmcv.Config.fromfile(cfg)

    cfg['model']['type'] = 'Mask2FormerVitForinfer2'

    # cfg['model']['relationship_head']['pretrained_transformers'] = transformers_model
    # cfg['model']['relationship_head']['cache_dir'] = './'    
    # if 'entity_length' in cfg['model']['relationship_head'] and cfg['model']['relationship_head']['entity_length'] > 1:
    #     cfg['model']['relationship_head']['entity_part_encoder'] = transformers_model

    model = init_detector(cfg, ckp)
    return model

def show_gt(image_id, flag_show_relation=False):
    psg_val_data_file = '/root/autodl-tmp/dataset/psg/psg_test.json'
    img_dir = "/root/autodl-tmp/dataset/coco/"
    psg_val_data = load_json(psg_val_data_file)
    for d in tqdm(psg_val_data['data']):
        if d['image_id'] == str(image_id):
            seg_map = read_image(os.path.join(img_dir, d["pan_seg_file_name"]), format="RGB")
            seg_map = rgb2id(seg_map)

            gt_relation = d["relations"]
            #get category ids
            gt_category_ids = []
            gt_label = []
            # get seperate masks
            gt_masks = []
            for i, s in enumerate(d["segments_info"]):
                gt_category_ids.append(s["category_id"])
                gt_label.append(CLASSES[s["category_id"]])
                gt_masks.append(seg_map == s["id"])
            print(gt_category_ids)
            for r in gt_relation:
                print(gt_label[r[0]], PREDICATES[r[2]], gt_label[r[1]])
            # Viualize masks
            colormap_coco =  get_colormap(len(gt_masks))
            colormap_coco = (np.array(colormap_coco) / 255).tolist()

            # Viualize masks
            image_name = img_dir + d['file_name']
            img = mmcv.imread(image_name)
            img = PIL.Image.fromarray(img)
            converter = PIL.ImageEnhance.Color(img)
            img = converter.enhance(0.01)
            # img_h,  = img.size

            viz = Visualizer(img)
            # print()
            # print(gt_label)
            # print(gt_masks)
            viz.overlay_instances(
                labels=gt_label,
                masks=gt_masks,
                assigned_colors=colormap_coco,
            )
            viz_img = viz.get_output().get_image()
            # if out_file is not None:
            out_file_name = d['file_name'].split('/')[-1].split('.')[0]
            img_metas = [{'batch_input_shape': [d['height'], d['width']],'img_shape':[d['height'], d['width']], 'ori_shape': [d['height'], d['width']]}]
            # img_metas = [{'batch_input_shape': [640, 481],'img_shape':[640, 481], 'ori_shape': [640, 481]}]
            mmcv.imwrite(viz_img,  out_file_name + '_gt.jpg')

            if flag_show_relation:
                show_relations(d["relations"], img, gt_label, gt_masks, colormap_coco, str(image_id)+'/')

            return image_name, img_metas
    # exit(0)


def show_result(pan_results, image_name, out_file_name, image_id):
    
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

    colormap_coco = get_colormap(len(objects_mask))
    colormap_coco = (np.array(colormap_coco) / 255).tolist()

    # ids = np.unique(pan_results)[::-1]
    # labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
    labels = [CLASSES[l] for l in label_pre]
    print(labels)

    # Viualize masks
    img = mmcv.imread(image_name)
    img = PIL.Image.fromarray(img)
    converter = PIL.ImageEnhance.Color(img)
    img = converter.enhance(0.01)
    mmcv.imwrite(np.asarray(img), 'bw'+ out_file_name + '_' + str(image_id) + '.jpg')

    viz = Visualizer(img)
    viz.overlay_instances(
        labels=labels,
        masks=objects_mask,
        assigned_colors=colormap_coco,
    )
    viz_img = viz.get_output().get_image()
    # if out_file is not None:
    mmcv.imwrite(viz_img, out_file_name + '_' + str(image_id) + '.jpg')

def show_hot(show_attention):
    print('show_attention', show_attention.shape)
    # print(group_token.shape)
    # attention = torch.softmax(entity_embedding @ group_token[0].permute(1,0), dim=0)
    sns.heatmap(show_attention.detach().cpu().tolist(), annot=True, cmap="YlGnBu")
    # 可选：添加轴标签
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.savefig('savefig_example.png')

def show_tokens_scores(tokens_scores, relation_res, entity_embedding):

    object_num = entity_embedding.shape[0]
    relation = relation_res[19]
    show_scores = tokens_scores[relation[0]*object_num+relation[1]]
    # show_scores = torch.softmax(show_scores, dim=-1)
    print(show_scores.shape)
    sns.heatmap(show_scores.detach().cpu().tolist(), annot=True, cmap="YlGnBu")
    # 可选：添加轴标签
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(str(relation[2]))

    plt.savefig('savefig_example.png')

if __name__=="__main__":

    INSTANCE_OFFSET = 1000
    cfg='configs/psg/v5_token_ablation_8_4.py'
    # cfg='configs/psg/v6.py'
    ckp=None
    # ckp='output/v5_token_ablation_8_4/latest.pth'
    # ckp='output/v6/epoch_1.pth'
    image_id = 2364856

    model = get_model(cfg, ckp, transformers_model=None)
    model.eval()
    # 模拟输入数据（替换为你的实际数据）
    # image_name = '/root/autodl-tmp/dataset/coco/val2017/000000507015.jpg'
    image_name, img_metas = show_gt(image_id)
    out_file_name = image_name.split('/')[-1].split('.')[0]

    image = Image.open(image_name)
    preprocess = transforms.Compose([
        # transforms.Resize((224, 224)),  # 根据需要调整大小
        transforms.ToTensor(),  # 将图像转换为 torch.Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化操作
    ])
    input_data = torch.tensor(preprocess(image)).to("cuda:0")
    input_data = input_data.unsqueeze(0)

    imgs = input_data
    result = model.simple_test(imgs, img_metas)
    # feats = model.extract_feat(imgs)
    # mask_cls_results, mask_pred_results, mask_features, query_feat = model.panoptic_head.simple_test(feats, img_metas)
    # # mask_cls_results, mask_pred_results, mask_features, query_feat = self.panoptic_head.simple_test(feats, img_metas, **kwargs)
    # results = model.panoptic_fusion_head.simple_test(mask_cls_results, mask_pred_results, img_metas)
    
    # device = mask_features.device
    # dtype = mask_features.dtype

    # res = results[0]
    # entityid_list = res['entityid_list']
    # target_keep = res['target_keep']
    # entity_embedding = query_feat[0][target_keep,:]

    # res['pan_results'] = res['pan_results'].detach().cpu().numpy()
    # pan_results = res['pan_results']

    # relation_pred, neg_idx, relation_feature = model.relationship_head.simple_test(query_feat, entity_embedding, visual=True)

    # tokens_scores = model.relationship_head.relation_embedding(relation_feature)

    # relation_res = []
    # if len(entityid_list) != 0:
    #     relation_pred, neg_idx = model.relationship_head.simple_test(query_feat, entity_embedding)
    #     # print(relation_pred.shape)
    #     relation_pred = torch.softmax(relation_pred, dim=-1)
    #     # 去除预测为空关系标签的影响
    #     relation_pred = relation_pred[:,1:]
    #     relation_pred[neg_idx,:] = -9999
    #     try:
    #         _, topk_indices = torch.topk(relation_pred.reshape([-1,]), k=20)
    #     except:
    #         topk_indices = torch.tensor(range(0,len(relation_pred.reshape([-1,]))))
    #     # subject, object, cls
    #     for index in topk_indices:
    #         pred_cls = index % relation_pred.shape[1]
    #         index_subject_object = index // relation_pred.shape[1]
    #         pred_subject = index_subject_object // entity_embedding.shape[0]
    #         pred_object = index_subject_object % entity_embedding.shape[0]
    #         pred = [pred_subject.item(), pred_object.item(), pred_cls.item()]
    #         relation_res.append(pred)
    

    # show_tokens_scores(tokens_scores, relation_res, entity_embedding)

    # show_result(pan_results, image_name, out_file_name, image_id)


    # # 运行多次以计算平均推理时间
    # num_iterations = 100  # 可根据需要进行调整
    # total_time = 0

    # for _ in tqdm(range(num_iterations)):
    #     start_time = time.time()
    #     with torch.no_grad():
    #         # 进行推理
    #         output = model.simple_test(input_data, img_metas)
    #     end_time = time.time()
    #     total_time += end_time - start_time

    # # 计算平均推理时间
    # average_inference_time = total_time / num_iterations

    # # 计算FPS
    # fps = 1 / average_inference_time
    # print(fps)
    # num_params = sum(p.numel() for p in model.parameters())
    # print(num_params / 1e6)
    # psg_all_data_file = '/root/autodl-tmp/dataset/psg/psg.json'
    # # 获取class列表
    # psg_dataset_file = load_json(psg_all_data_file)
    # # print('keys: ', list(psg_dataset_file.keys()))

    # psg_thing_cats = psg_dataset_file['thing_classes']
    # psg_stuff_cats = psg_dataset_file['stuff_classes']
    # psg_obj_cats = psg_thing_cats + psg_stuff_cats
    # datas = psg_dataset_file['data']
    # relation_dict = [0]*44
    # for data in datas:
    #     relation_len = len(data['relations'])
    #     relation_dict[relation_len] += 1

    # print(relation_dict)
    # # use 30 is suitable
    # object_features = [torch.rand(10,256),torch.rand(9,256),torch.rand(8,256),torch.rand(7,256)]
    # relation_tokens = [torch.rand(8,256),torch.rand(8,256),torch.rand(8,256),torch.rand(8,256)]
    # target_edges = [
    #     torch.tensor([[0, 1, 21], [0, 1, 21], [0, 3, 3], [0, 3, 14], [0, 4, 2], [3, 4, 2], [5, 4, 0], [6, 2, 16], [7, 2, 16]]),
    #     torch.tensor([[0, 1, 2], [0, 6, 14], [1, 2, 21], [1, 6, 14], [4, 3, 1], [5, 3, 0]]),
    #     torch.tensor([[0, 7, 48], [0, 7, 48], [1, 7, 11], [4, 6, 5], [7, 1, 5]]),
    #     torch.tensor([[0, 4, 3], [0, 4, 5], [1, 4, 3], [1, 5, 2], [2, 4, 3], [3, 5, 3]]),
    # ]
    # relation_features, all_edge_lbl = concat_relation_features(object_features, relation_tokens, target_edges)
    # print(relation_features.shape)
    # print(all_edge_lbl.shape)

    