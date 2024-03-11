import os
import numpy as np
import json
from detectron2.utils.colormap import colormap
from PIL import Image
import PIL
from detectron2.utils.visualizer import VisImage, Visualizer
import mmcv
from typing import Tuple
import os.path as osp
import random


PROJECT_ROOT = os.getcwd()
DATASETS_ROOT = os.path.dirname(PROJECT_ROOT)
CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard',
    'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit',
    'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform',
    'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea',
    'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone',
    'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other',
    'tree', 'fence', 'ceiling-merged', 'sky-other-merged',
    'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged',
    'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged',
    'food-other-merged', 'building-other-merged', 'rock-merged',
    'wall', 'rug-merged', 'background'
]

PREDICATES = [
    'over',
    'in front of',
    'beside',
    'on',
    'in',
    'attached to',
    'hanging from',
    'on back of',
    'falling off',
    'going down',
    'painted on',
    'walking on',
    'running on',
    'crossing',
    'standing on',
    'lying on',
    'sitting on',
    'flying over',
    'jumping over',
    'jumping from',
    'wearing',
    'holding',
    'carrying',
    'looking at',
    'guiding',
    'kissing',
    'eating',
    'drinking',
    'feeding',
    'biting',
    'catching',
    'picking',
    'playing with',
    'chasing',
    'climbing',
    'cleaning',
    'playing',
    'touching',
    'pushing',
    'pulling',
    'opening',
    'cooking',
    'talking to',
    'throwing',
    'slicing',
    'driving',
    'riding',
    'parked on',
    'driving on',
    'about to hit',
    'kicking',
    'swinging',
    'entering',
    'exiting',
    'enclosing',
    'leaning on',
]

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

def get_ann_info(d):
    # d['pan_seg_file_name']
    # Process bbox annotations
    gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

    gt_bboxes = []
    gt_labels = []

    # FIXME: Do we have to filter out `is_crowd`?
    # Do not train on `is_crowd`,
    # i.e just follow the mmdet dataset classes
    # Or treat them as stuff classes?
    # Can try and train on datasets with iscrowd
    # and without and see the difference

    for a, s in zip(d['annotations'], d['segments_info']):
        # NOTE: Only thing bboxes are loaded
        if s['isthing']:
            gt_bboxes.append(a['bbox'])
            gt_labels.append(a['category_id'])

    if gt_bboxes:
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)
    else:
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        gt_labels = np.array([], dtype=np.int64)

    # Process segment annotations
    gt_mask_infos = []
    for s in d['segments_info']:
        gt_mask_infos.append({
            'id': s['id'],
            'category': s['category_id'],
            'is_thing': s['isthing']
        })

    # Process relationship annotations
    gt_rels = d['relations'].copy()

    # for test or val set, filter the duplicate triplets,
    # but allow multiple labels for each pair
    all_rel_sets = []
    for (o0, o1, r) in gt_rels:
        if (o0, o1, r) not in all_rel_sets:
            all_rel_sets.append((o0, o1, r))
    gt_rels = np.array(all_rel_sets, dtype=np.int32)

    # add relation to target
    num_box = len(gt_mask_infos)
    relation_map = np.zeros((num_box, num_box), dtype=np.int64)
    for i in range(gt_rels.shape[0]):
        # If already exists a relation?
        if relation_map[int(gt_rels[i, 0]), int(gt_rels[i, 1])] > 0:
            if random.random() > 0.5:
                relation_map[int(gt_rels[i, 0]),
                                int(gt_rels[i, 1])] = int(gt_rels[i, 2])
        else:
            relation_map[int(gt_rels[i, 0]),
                            int(gt_rels[i, 1])] = int(gt_rels[i, 2])

    ann = dict(
        bboxes=gt_bboxes,
        labels=gt_labels,
        rels=gt_rels,
        rel_maps=relation_map,
        bboxes_ignore=gt_bboxes_ignore,
        masks=gt_mask_infos,
        seg_map=d['pan_seg_file_name'],
    )

    return ann


def get_colormap(num_colors: int):
    return (np.resize(colormap(), (num_colors, 3))).tolist()

def write_json(x_struct: dict, json_file: str):
    # json_str = json.dumps(x_struct,indent=2,ensure_ascii=False)
    with open(json_file, 'w+') as fd:
        json.dump(x_struct, fd, indent=4, ensure_ascii=False)

def load_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data



def read_image(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR" or "YUV-BT.601".

    Returns:
        image (np.ndarray):
            an HWC image in the given format, which is 0-255, uint8 for
            supported image modes in PIL or "BGR"; float (0-1 for Y) for YUV-BT.601.
    """
    with open(file_name, "rb") as f:
        image = Image.open(f)

        # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
        # image = _apply_exif_orientation(image)
        return convert_PIL_to_numpy(image, format)

def convert_PIL_to_numpy(image, format):
    """
    Convert PIL image to numpy array of target format.

    Args:
        image (PIL.Image): a PIL image
        format (str): the format of output image

    Returns:
        (np.ndarray): also see `read_image`
    """
    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format in ["BGR", "YUV-BT.601"]:
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)

    # handle formats not supported by PIL
    elif format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    # elif format == "YUV-BT.601":
    #     image = image / 255.0
    #     image = np.dot(image, np.array(_M_RGB2YUV).T)

    return image


def show_relations(relations, image_name, rel_obj_labels, masks, out_dir):
    img = mmcv.imread(image_name)
    img = PIL.Image.fromarray(img)
    converter = PIL.ImageEnhance.Color(img)
    img = converter.enhance(0.01)
    img_w, img_h = img.size

    colormap_coco = get_colormap(len(masks))
    colormap_coco = (np.array(colormap_coco) / 255).tolist()

    n_rels = len(relations)

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

    for i, r in enumerate(relations):
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
    box_color: str = 'white',
    fontweight: str = None,
    return_height = False
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
        fontweight=fontweight,
    )
    viz_img.get_image()
    text_dims = text_obj.get_bbox_patch().get_extents()
    if return_height:
        return text_dims.width, text_dims.height
    return text_dims.width


