import mmcv
from mmdet.apis import init_detector, inference_detector
import torch
from tqdm import tqdm
import time
import cv2
from torchvision import transforms
from PIL import Image
import PIL
from util import CLASSES, PREDICATES, get_colormap, load_json, read_image, show_relations, draw_text
import os
from panopticapi.utils import rgb2id, id2rgb

import numpy as np
from detectron2.utils.visualizer import VisImage, Visualizer

import matplotlib.pyplot as plt
import seaborn as sns
from infer_p import get_model

import networkx as nx
import matplotlib.pyplot as plt

# def get_model(cfg, ckp, transformers_model):
#     # print(cfg)
#     # print(type(cfg))
#     cfg = mmcv.Config.fromfile(cfg)

#     cfg['model']['type'] = 'Mask2FormerVitForinfer2'

#     # cfg['model']['relationship_head']['pretrained_transformers'] = transformers_model
#     # cfg['model']['relationship_head']['cache_dir'] = './'    
#     # if 'entity_length' in cfg['model']['relationship_head'] and cfg['model']['relationship_head']['entity_length'] > 1:
#     #     cfg['model']['relationship_head']['entity_part_encoder'] = transformers_model

#     model = init_detector(cfg, ckp)
#     return model

def show_gt(image_id, flag_show_relation=False):
    psg_val_data_file = '/root/autodl-tmp/dataset/psg/psg_test.json'
    img_dir = "/root/autodl-tmp/dataset/coco/"
    psg_val_data = load_json(psg_val_data_file)
    for d in tqdm(psg_val_data['data']):
        if d['image_id'] == str(image_id):
            seg_map = read_image(os.path.join(img_dir, d["pan_seg_file_name"]), format="RGB")
            seg_map = rgb2id(seg_map)

            raw_realtion = d["relations"]
            gt_relation = []
            for r in raw_realtion:
                if r in gt_relation:
                    continue
                gt_relation.append(r)
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
            print(gt_label)
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
            show_relation(gt_relation, gt_label, colormap_coco, gt=True)
            # if out_file is not None:
            out_file_name = d['file_name'].split('/')[-1].split('.')[0]
            img_metas = [{'batch_input_shape': [d['height'], d['width']],'img_shape':[d['height'], d['width']], 'ori_shape': [d['height'], d['width']]}]
            # img_metas = [{'batch_input_shape': [640, 481],'img_shape':[640, 481], 'ori_shape': [640, 481]}]
            mmcv.imwrite(viz_img,  out_file_name + '_gt.jpg')

            visualize_objects_and_relations(viz_img, gt_masks, gt_relation, gt=True)

            if flag_show_relation:
                show_relations(d["relations"], img, gt_label, gt_masks, str(image_id)+'/')

            return image_name, img_metas, colormap_coco
    # exit(0)

def show_relation(gt_relation, gt_label, colormap_coco, gt=False):
    if gt:
        viz_graph = VisImage(np.full((570, 280, 3), 255))
    else:
        viz_graph = VisImage(np.full((570, 310, 3), 255))
    curr_y = 2
    text_size = 12
    text_padding = 5
    for index, r in enumerate(gt_relation):
        # box_color = 'white'
        # font = 18
        fontweight = 'normal'
        # if not gt and index in [0,1,10,15,19]:
        #     # box_color = 'pink'
        #     # font = 36
        #     fontweight = 'semibold'
        #     # pass
        # if gt and index == 2:
        #     continue
        s_idx, o_idx, rel_label_id = r
        s_label = gt_label[s_idx]
        o_label = gt_label[o_idx]
        rel_label = PREDICATES[rel_label_id]
        curr_x = 2
        text_width = draw_text(
            viz_img=viz_graph,
            text=s_label,
            x=curr_x,
            y=curr_y,
            color=colormap_coco[s_idx],
            size=text_size,
            padding=text_padding,
            # font=font,
            fontweight=fontweight,
            # box_color = box_color,
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
            fontweight=fontweight,
            # box_color='gainsboro',
            # box_color = box_color,
            # font=font,
        )
        curr_x += text_width

        # Draw object text
        text_width, text_height = draw_text(
            viz_img=viz_graph,
            text=o_label,
            x=curr_x,
            y=curr_y,
            color=colormap_coco[o_idx],
            size=text_size,
            padding=text_padding,
            # font=font,
            fontweight=fontweight,
            # box_color = box_color,
            return_height=True,
        )
        # print(text_height)
        curr_y += text_height - 3
    if gt:
        mmcv.imwrite(viz_graph.get_image(),  'relation_gt.png')
    else:
        mmcv.imwrite(viz_graph.get_image(),  'relation_predict.png')
        

def show_result(pan_results, image_name, out_file_name, image_id, relations):
    
    # 将得到的结果处理为id和mask0-1矩阵对应的关系
    object_masks = []
    label_pre = []
    labels = []

    for instance_id in entityid_list:
        # instance_id == 133 background
        mask = pan_results == instance_id
        if instance_id == 133:
            continue
        object_masks.append(mask)
        label_pre.append(instance_id % INSTANCE_OFFSET)
        # labels.append(CLASSES[instance_id % INSTANCE_OFFSET])
    # list_gt = [7,9,3,4,1,0,6,8,2,5]
    # list_gt = [5, 2, 3, 1, 0, 4]
    
    colormap_coco = get_colormap(len(object_masks))
    colormap_coco = (np.array(colormap_coco) / 255).tolist()
    
    # ids = np.unique(pan_results)[::-1]
    # labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
    labels = [CLASSES[l] for l in label_pre]
    # labels = [ labels[gt] for gt in list_gt]
    # object_masks = [ object_masks[gt] for gt in list_gt]
    # new_relations = []
    # for r in relations:
    #     new_relations.append([list_gt.index(r[0]), list_gt.index(r[1]), r[2]])
    new_relations = relations

    # Viualize masks
    img = mmcv.imread(image_name)
    img = PIL.Image.fromarray(img)
    converter = PIL.ImageEnhance.Color(img)
    img = converter.enhance(0.01)
    mmcv.imwrite(np.asarray(img), 'bw'+ out_file_name + '_' + str(image_id) + '.jpg')

    viz = Visualizer(img)
    viz.overlay_instances(
        labels=labels,
        masks=object_masks,
        assigned_colors=colormap_coco,
    )
    viz_img = viz.get_output().get_image()
    # if out_file is not None:
    mmcv.imwrite(viz_img, out_file_name + '_' + str(image_id) + '.jpg')
    show_relation(new_relations, labels, colormap_coco)
    return viz_img

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

def visualize_objects_and_relations(vis_image, object_masks, relations, gt=False):

    # Create a blank image to draw the visualization
    # vis_image = np.zeros_like(image)
    total_relation = []

    # Draw each object's contour with a unique color
    # for object_index, object_mask in enumerate(object_masks):
    #     color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    #     object_contour, _ = cv2.findContours(object_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     cv2.drawContours(vis_image, object_contour, -1, color, thickness=cv2.FILLED)

    # Draw relations between objects
    for relation in relations:
        if relation[:2] in total_relation:
            continue
        total_relation.append(relation[:2])
        object1_index, object2_index, relation_type = relation
        # color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        color = (0, 0, 0)
        object1_mask = object_masks[object1_index]
        object2_mask = object_masks[object2_index]
        object1_mask = object1_mask.astype(np.uint8)
        object2_mask = object2_mask.astype(np.uint8)

        # Calculate moments only if the object has non-zero area
        if cv2.countNonZero(object1_mask) > 0:
            object1_center = np.array([cv2.moments(object1_mask)['m10'],cv2.moments(object1_mask)['m01']]) / cv2.moments(object1_mask)['m00']
        else:
            object1_center = np.zeros(2)

        if cv2.countNonZero(object2_mask) > 0:
            object2_center = np.array([cv2.moments(object2_mask)['m10'],cv2.moments(object2_mask)['m01']]) / cv2.moments(object2_mask)['m00']
        else:
            object2_center = np.zeros(2)
        
       
        # Draw a line between object centers
        cv2.line(vis_image, tuple(object1_center.astype(int)), tuple(object2_center.astype(int)), color, thickness=2)

        # Calculate the midpoint for placing the relation label
        mid_point = ((object1_center + object2_center) / 2).astype(int)

        # Display the relation type as text near the midpoint
        cv2.putText(vis_image, PREDICATES[relation_type], tuple(mid_point), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Display the visualization
    plt.imshow(vis_image)
    plt.axis('off')
    if gt:
        plt.savefig('graph_image_gt.png')
    else:
        plt.savefig('graph_image.png')


def visualize_scene_graph(objects, relationships):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes for each object
    for obj_id, obj_label in enumerate(objects):
        G.add_node(obj_id, label=obj_label)

    total_relation = []
    # Add edges for relationships
    for rel in relationships:
        if rel[:2] in total_relation:
            continue
        total_relation.append(rel[:2])

        subject, obj, predicate = rel
        G.add_edge(subject, obj, label=PREDICATES[predicate])

    # Get node labels
    node_labels = {node: G.nodes[node]['label'] for node in G.nodes}

    # Get edge labels
    edge_labels = {(edge[0], edge[1]): G.edges[edge]['label'] for edge in G.edges}

    # Draw the graph
    pos = nx.circular_layout(G)  # You can try different layout algorithms
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=1000, font_size=8, node_color='skyblue', font_color='black', font_weight='bold', arrowsize=15)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)

    plt.savefig('savefig_example2.png')

if __name__=="__main__":

    INSTANCE_OFFSET = 1000
    # cfg='configs/psg/v6_token_ablation_64_4.py'
    cfg='configs/psg/v6.py'
    # ckp=None
    # ckp='output/v6_token_ablation_64_4/latest.pth'
    ckp='output/v6/epoch_12.pth'
    image_id = 285707

    model = get_model(cfg, ckp, mode='v6', transformers_model=None)
    model.eval()
    # 模拟输入数据（替换为你的实际数据）
    # image_name = '/root/autodl-tmp/dataset/coco/val2017/000000507015.jpg'
    image_name, img_metas, colormap_coco = show_gt(image_id)
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
    results = model.simple_test(imgs, img_metas)

    res = results[0]
    # show_tokens_scores(tokens_scores, relation_res, entity_embedding)

    pan_results = res['pan_results']
    rela_results = res['rela_results']
    relations = rela_results['relation']
    entityid_list = rela_results['entityid_list']
    label_pre = [instance_id % INSTANCE_OFFSET for instance_id in entityid_list]
    # 将得到的结果处理为id和mask0-1矩阵对应的关系
    object_masks = []
    label_pre = []
    for instance_id in entityid_list:
        # instance_id == 133 background
        mask = pan_results == instance_id
        if instance_id == 133:
            continue
        object_masks.append(mask)
        label_pre.append(instance_id % INSTANCE_OFFSET)
    labels = [CLASSES[l] for l in label_pre]

    img = show_result(pan_results, image_name, out_file_name, image_id, relations[:20])

    # img = mmcv.imread(image_name)
    # img = PIL.Image.fromarray(img)
    # converter = PIL.ImageEnhance.Color(img)
    # img = converter.enhance(0.01)

    
    visualize_objects_and_relations(img, object_masks, relations[:20])
    # visualize_scene_graph(labels, relation[:20])
    # show_relations(relation[:20], image_name, labels, object_masks, out_dir='000000043816_2364856/')



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

    