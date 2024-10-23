# GroupRF


整体框架还是基于[mmdet](https://github.com/open-mmlab/mmdetection)

<br>

## Install
环境参考 [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) 
```
pip install -e .
```

panopticapi
```
pip install git+https://github.com/cocodataset/panopticapi.git
```

<br>

## 数据准备

下载 [coco instance val 2017](https://cocodataset.org/#download)，用于验证 psg val 的instance map



<br>

## (可能)需要的一些预训练权重
### 分割部分:
[mask2former](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask2former)，
[预训练权重](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask2former)


<br>

## 训练
+ 调整 `configs/psg/v6.py` 中预训练路径、输出路径、tra val 数据路径
```python
# 模型中预训练部分
model['relationship_head']['pretrained_transformers'] = "/path/chinese-roberta-wwm-ext"
load_from = "/path/mask2former_r50_lsj_8x2_50e_coco-panoptic_20220326_224516-11a44721.pth"
# tra 数据部分
data['train']['ann_file'] = 'data/psg_tra.json'
data['train']['img_prefix'] = '/path/psg/dataset/'
data['train']['seg_prefix'] = '/path/psg/dataset/'
# val 数据部分
data['val']['ann_file'] = 'data/psg_val.json'
data['val']['img_prefix'] = '/path/psg/dataset/'
data['val']['seg_prefix'] = '/path/psg/dataset/'
data['val']['ins_ann_file'] = 'data/psg_instance_val.json'
# test 数据部分
data['test']['ann_file'] = 'data/psg_val.json'
data['test']['img_prefix'] = '/path/psg/dataset/'
data['test']['seg_prefix'] = '/path/psg/dataset/'
data['test']['ins_ann_file'] = 'data/psg_instance_val.json'
# 输出路径
work_dir = 'output/v6'
```

```
# 8卡训练
bash tools/dist_train.sh configs/psg/v6.py 8 
```

<br>










