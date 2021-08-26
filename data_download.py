import sys
import os
import moxing as mox


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


def dowload_data_set2(obspath='obs://lihd-bucket/dataset/Market-1501-v15.09.15/', dataset_name='market'):  # 'ew_Mar' or 'data_demo'
    print(obspath)
    localpath = os.path.join('/home/work/user-job-dir/code_pcb', 'dataset')
    # localpath = os.path.join('/cache/user-job-dir/code-obs', 'dataset')
    print(localpath)
    if not os.path.exists(localpath):
        os.mkdir(localpath)
    dpath = os.path.join(localpath, dataset_name)
    print(dpath)
    mobspath = obspath
    if not os.path.exists(dpath):
        os.mkdir(dpath)
    mox.file.copy_parallel(mobspath, dpath)
