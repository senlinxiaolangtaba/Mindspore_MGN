import mindspore.ops as ops
import os
import argparse
import ast
from mindspore import context
from mindspore import Tensor
from test2 import test_v1
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
import mindspore
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.common import set_seed
from mindspore.parallel import set_algo_parameters
import mindspore.nn as nn
import mindspore.common.initializer as weight_init
from mindspore.train.callback._lr_scheduler_callback import LearningRateScheduler
import sys
from MGN_with_PCB import MGN_test
from create_dataset_forevaluate import create_dataset_384_evaluate, create_dataset_398_evaluate
import numpy as np
import time
from reranking import re_ranking
import moxing as mox
import scipy.io


def dowload_checkpoint_set(obspath='obs://wangl97/LIHUADONG/V0056/', datasetname='check'):
    localpath = os.path.join('/home/work/user-job-dir/code_pcb', 'checkpoint')
    print(localpath)
    if not os.path.exists(localpath):
        os.mkdir(localpath)
    dpath = os.path.join(localpath, datasetname)
    print(dpath)
    mobspath = obspath
    if not os.path.exists(dpath):
        os.mkdir(dpath)

    mox.file.copy_parallel(mobspath, dpath)
    return dpath


def get_id(img_path):
    camera_id = []
    labels = []
    for path in img_path:
        # filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


def fliplr(img):
    op = ops.ReverseV2(axis=[3])
    img_flip = op(img)
    return img_flip


def extract_feature(model, dataset):
    model = model.set_train(False)
    features = None
    count = 0
    iter = dataset.create_dict_iterator()
    for data in iter:

        images = data["image"]
        target = data["label"]
        n = images.shape[0]
        c = images.shape[1]
        h = images.shape[2]
        w = images.shape[3]
        count += n
        print(count)
        zero = ops.Zeros()
        ff = zero((n, 11008), mindspore.float32)  # 这个类型可能有问题
        for i in range(2):
            if (i == 1):
                images = fliplr(images)
            outputs = model(images)
            add = ops.Add()
            ff = add(ff, outputs)
        norm = nn.Norm(axis=1, keep_dims=True)
        fnorm = norm(ff) * np.sqrt(8)
        fnorm = fnorm.expand_as(ff)
        divide = ops.Div()
        ff = divide(ff, fnorm)
        ff = ff.view((n, 11008))

        if features == None:
            features = ff
        else:
            concat = ops.Concat(axis=0)
            features = concat((features, ff))

    return features


def dowload_data_set(obspath='obs://lihd-bucket/ew_Mar/', datasetname='market'):
    localpath = os.path.join('/home/work/user-job-dir/code_pcb', 'dataset')
    print(localpath)
    if not os.path.exists(localpath):
        os.mkdir(localpath)
    dpath = os.path.join(localpath, datasetname)
    print(dpath)
    mobspath = obspath
    if not os.path.exists(dpath):
        os.mkdir(dpath)

        mox.file.copy_parallel(mobspath, dpath)


def evaluate_v1(check_path):
    model = MGN_test(751)

    print(check_path)
    dic = mindspore.train.serialization.load_checkpoint(check_path)
    mindspore.train.serialization.load_param_into_net(model, parameter_dict=dic)

    dowload_data_set()
    dataset_path = os.path.join('/home/work/user-job-dir/code_pcb', 'dataset')
    dataset_path = os.path.join(dataset_path, 'market')
    query_path = os.path.join(dataset_path, 'query')
    gallery_path = os.path.join(dataset_path, 'bounding_box_test')

    # 获取数据集
    query, query_dataset = create_dataset_384_evaluate(query_path)
    gallery, gallery_dataset = create_dataset_384_evaluate(gallery_path)

    gallery_path = gallery.imgpath
    query_path = query.imgpath

    gallery_cam, gallery_label = get_id(gallery_path)
    query_cam, query_label = get_id(query_path)

    gallery_feature = extract_feature(model, gallery_dataset)
    query_feature = extract_feature(model, query_dataset)
    result = {'gallery_f': gallery_feature.asnumpy(), 'gallery_label': gallery_label, 'gallery_cam': gallery_cam,
              'query_f': query_feature.asnumpy(), 'query_label': query_label, 'query_cam': query_cam}

    return result


def evaluate(score, ql, qc, gl, gc):
    index = np.argsort(score)  # from small to large
    # index = index[::-1]
    # good index
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    # cmc = torch.IntTensor(len(index)).zero_()
    cmc = np.zeros((len(index)), dtype=int)
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


def save_features(check, check_name):
    result = evaluate_v1(check)
    save_path = os.path.join('/home/work/user-job-dir/code_pcb', 'evaluate')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print(save_path)
    save_file = os.path.join(save_path, 'features_' + 'V0056' + check_name[:-5] + '.mat')
    scipy.io.savemat(save_file, result)
    mox.file.copy_parallel(save_path, 'obs://lihd-bucket/evaluate/')


def test_byre(check_name):
    result = evaluate_v1(check_name)
    query_feature = result['query_f']
    query_cam = result['query_cam']
    # print(query_cam)
    query_label = result['query_label']
    query_label = np.array(query_label)
    query_cam = np.array(query_cam)
    gallery_feature = result['gallery_f']
    gallery_cam = result['gallery_cam']
    gallery_label = result['gallery_label']
    gallery_label = np.array(gallery_label)
    gallery_cam = np.array(gallery_cam)

    # multi = os.path.isfile('multi_query.mat')
    #
    # if multi:
    #     m_result = scipy.io.loadmat('multi_query.mat')
    #     mquery_feature = m_result['mquery_f']
    #     mquery_cam = m_result['mquery_cam'][0]
    #     mquery_label = m_result['mquery_label'][0]

    # CMC = torch.IntTensor(len(gallery_label)).zero_()
    # CMC = torch.IntTensor(len(gallery_label)).zero_()
    CMC = np.zeros((len(gallery_label)), dtype=int)
    ap = 0.0
    # re-ranking
    print('calculate initial distance')
    q_g_dist = np.dot(query_feature, np.transpose(gallery_feature))
    print('1')
    q_q_dist = np.dot(query_feature, np.transpose(query_feature))
    print('2')
    g_g_dist = np.dot(gallery_feature, np.transpose(gallery_feature))
    print('3')
    since = time.time()
    re_rank = re_ranking(q_g_dist, q_q_dist, g_g_dist)
    time_elapsed = time.time() - since
    print('Reranking complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(re_rank[i, :], query_label[i], query_cam[i], gallery_label, gallery_cam)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        # print(i, CMC_tmp[0])

    # CMC = CMC.float()
    CMC = CMC / len(query_label)  # average CMC
    print(
        ' wpoch:%d Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (
            check_name[-18:-5], CMC[0], CMC[4], CMC[9], ap / len(query_label)))
    return CMC, ap / len(query_label)


def Evaluate_v1():
    check_path = dowload_checkpoint_set()
    best_t1 = 0
    best_epoch = 0
    for check_name in os.listdir(check_path):
        if check_name[-1] == 't':
            print(check_name)
            check = os.path.join(check_path, check_name)
            save_features(check, check_name)
        # cmc, map = test_byre(check)
        # if cmc[0] > best_t1:
        #     best_t1 = cmc[0]
        #     best_epoch = check_name[-17:-5]
        #     print('now best epoch is %d ,top1 is %f' % (best_epoch, best_t1))


if __name__ == '__main__':
    Evaluate_v1()
