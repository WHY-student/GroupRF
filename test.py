from util import load_json, write_json
import numpy as np
from tqdm import tqdm


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
    # print('tra', len(tra_id_list))
    # print('val', len(val_id_list))
    # print('test', len(test_id_list))
    
    return tra_id_list, val_id_list, test_id_list

import torch
import time


if __name__=="__main__":
    psg_all_data_file='/root/autodl-tmp/dataset/psg/psg.json'
    psg_tra_data_file = '/root/autodl-tmp/dataset/psg/psg_train_val.json'
    psg_val_data_file='/root/autodl-tmp/dataset/psg/psg_val_test.json'
    tra_id_list, val_id_list, test_id_list = get_tra_val_test_list(
        psg_tra_data_file=psg_tra_data_file,
        psg_val_data_file=psg_val_data_file,
    )
    psg_all_data = load_json(psg_all_data_file)
    psg_val_data = []
    for d in tqdm(psg_all_data['data']):
        if d['image_id'] in val_id_list:
            psg_val_data.append(d)
    write_json({"data":psg_val_data}, '/root/autodl-tmp/dataset/psg/psg_val.json')

    # pass
    
    # for i in range(1000):
    #     result = torch.cat((entity_embedding.repeat(object_num, 1).repeat(token_num, 1), entity_embedding.unsqueeze(1).expand(object_num, object_num, -1).reshape(object_num*object_num, -1).repeat(token_num, 1), relation_tokens.unsqueeze(1).expand(token_num, object_num*object_num, -1).reshape(token_num*object_num*object_num, -1)), 1)
    #     neg_idx = []
    #     for i in range(object_num):
    #         neg_idx.append(int(i*object_num+i))
    # end_time = time.time()
    # print(end_time- start_time)
    # start_time = time.time()
    # for i in range(1000):
    #     relation_features = []
    #     neg_idx = []
    #     for i in range(object_num):
    #         neg_idx.append(int(i*object_num+i))
    #         for j in range(object_num):
    #             relation_features.append(torch.cat((entity_embedding[i, :].repeat(token_num, 1), entity_embedding[j, :].repeat(token_num, 1), relation_tokens), 1).view(1, -1))
        
    #     relation_features = torch.cat(relation_features, 0)
    # end_time = time.time()
    # print(end_time- start_time)

