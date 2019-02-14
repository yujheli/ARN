import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as trans
from reid_dataset import AdaptReID_Dataset
from reid_loss import *

def extract_feature(model, dataloader):
    features, infos = [], []
    for batch in dataloader:
        image, label, camera_id = split_datapack(batch)
        feature = model.encoder_base(image)
        feature = model.encoder_c(feature, use_avg=True)
        feature = feature.view(feature.size()[0], -1).data.cpu().numpy()
        for i in range(feature.shape[0]):
            features.append(feature[i])
            infos.append((label[i].cpu().numpy(), camera_id[i].cpu().numpy()))
        
    return features, infos

def rank_func(query_len, test_len, match, junk, matrix_argsort, use_rank):
    CMC = np.zeros([query_len, test_len], dtype=np.float32)    
    for q_idx in range(query_len):
        counter = 0
        for t_idx in range(test_len):
            if matrix_argsort[q_idx][t_idx] in junk[q_idx]:
                continue
            else:
                counter += 1
                if matrix_argsort[q_idx][t_idx] in match[q_idx]:
                    CMC[q_idx, counter-1:] = 1.0
                if counter == use_rank:
                    break
    rank_score = np.mean(CMC[:,0])
    return rank_score

def dist_func(metric, query_features, test_features):
    if metric == 'L1':
        metric = 'hamming'
    elif metric == 'L2':
        metric = 'euclidean'
    matrix = cdist(query_features, test_features, metric=metric)
    return matrix
    
def get_match(query_infos, test_infos):
    match, junk = [], []
    for (query_label, query_camera_id) in query_infos:
        tmp_match, tmp_junk = [], []
        for idx, (test_label, test_camera_id) in enumerate(test_infos):
            if test_label == query_label and query_camera_id != test_camera_id:
                tmp_match.append(idx)
            elif test_label == query_label or test_label < 0:
                tmp_junk.append(idx)
        match.append(tmp_match)
        junk.append(tmp_junk)
    return match, junk

def evaluate(args, model, transform_list, match, junk):
    print("Evaluating...")
    model.eval()
    testData = AdaptReID_Dataset(dataset_name=args.target_dataset, mode='test', transform=trans.Compose(transform_list))
    queryData = AdaptReID_Dataset(dataset_name=args.target_dataset, mode='query', transform=trans.Compose(transform_list))
    testDataloader = DataLoader(testData, batch_size=args.batch_size, shuffle=False)
    queryDataloader = DataLoader(queryData, batch_size=args.batch_size, shuffle=False)
    test_features, test_infos = extract_feature(model, testDataloader)
    query_features, query_infos = extract_feature(model, queryDataloader)

    if match == None:
        match, junk = get_match(query_infos, test_infos)
    dist_matrix = dist_func(args.dist_metric, query_features, test_features)
    matrix_argsort = np.argsort(dist_matrix, axis=1)
    rank_score = rank_func(len(query_features), len(test_features), match, junk, matrix_argsort, args.rank)
    print('Evaluation: Rank score: {}'.format(rank_score))
    return rank_score, match, junk