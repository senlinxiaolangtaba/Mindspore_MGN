import os
import numpy as np

from mindspore import Tensor
from mindspore.train.model import Model
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2


def create_dataset(dataset_path, do_train=True, repeat_num=1, batxhsize=1, image_height=398, image_width=192,
                   platform="CPU"):
    """
    create a train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        config(struct): the config of train and eval in diffirent platform.
        repeat_num(int): the repeat times of dataset. Default: 1.
        batxhsize(int):batchsize
        image_height:height
        platform:which platform wangt use

    Returns:
        dataset
    """
    if platform == "Ascend":
        # 可能有问题
        rank_size = int(os.getenv("RANK_SIZE", '1'))
        rank_id = int(os.getenv("RANK_ID", '0'))
        if rank_size == 1:
            data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True)
        else:
            data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                             num_shards=rank_size, shard_id=rank_id)
    elif platform == "GPU":
        if do_train:
            # if config.run_distribute:
            #     from mindspore.communication.management import get_rank, get_group_size
            #     data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True,
            #                                      num_shards=get_group_size(), shard_id=get_rank())
            # else:
            data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True)
        else:
            data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True)
    elif platform == "CPU":
        data_set = ds.ImageFolderDataset(dataset_dir=dataset_path, num_parallel_workers=1, shuffle=True)

    # resize_height = image_height
    # resize_width = image_width
    buffer_size = 1000

    # define map operations
    decode_op = C.Decode()
    # resize_crop_op = C.RandomCropDecodeResize(resize_height, scale=(0.08, 1.0), ratio=(0.75, 1.333))
    horizontal_flip_op = C.RandomHorizontalFlip(prob=0.5)

    resize_op = C.Resize((image_height, image_width))
    # center_crop = C.CenterCrop(resize_width)
    # rescale_op = C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4)
    normalize_op = C.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                               std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    change_swap_op = C.HWC2CHW()

    cut = C.CutOut(40, num_patches=2)

    if do_train:
        trans = [decode_op, resize_op, horizontal_flip_op, normalize_op, change_swap_op]
    else:
        trans = [decode_op, resize_op, normalize_op, change_swap_op]

    type_cast_op = C2.TypeCast(mstype.int32)

    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=1)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=1)

    # apply shuffle operations
    data_set = data_set.shuffle(buffer_size=buffer_size)

    # apply batch operations
    data_set = data_set.batch(batxhsize, drop_remainder=True)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set


def create_dataset_384(dataset_path, do_train=True, repeat_num=1, batxhsize=1, image_height=384, image_width=128,
                       platform="CPU"):
    """
    create a train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        config(struct): the config of train and eval in diffirent platform.
        repeat_num(int): the repeat times of dataset. Default: 1.
        batxhsize(int):batchsize
        image_height:height
        platform:which platform wangt use

    Returns:
        dataset
    """
    if platform == "Ascend":
        # 可能有问题
        rank_size = int(os.getenv("RANK_SIZE", '1'))
        rank_id = int(os.getenv("RANK_ID", '0'))
        sampler = ds.RandomSampler(num_samples=9)
        if rank_size == 1:
            data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True)
            # data_set = ds.ImageFolderDataset(dataset_path, sampler=sampler,
            #                                  num_parallel_workers=8)
        else:
            data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                             num_shards=rank_size, shard_id=rank_id)
            # data_set = ds.ImageFolderDataset(dataset_path, sampler=sampler,
            #                                  num_parallel_workers=8)
    elif platform == "GPU":
        if do_train:
            # if config.run_distribute:
            #     from mindspore.communication.management import get_rank, get_group_size
            #     data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True,
            #                                      num_shards=get_group_size(), shard_id=get_rank())
            # else:
            data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True)
        else:
            data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True)
    elif platform == "CPU":
        data_set = ds.ImageFolderDataset(dataset_dir=dataset_path, num_parallel_workers=1, shuffle=True)

    # resize_height = image_height
    # resize_width = image_width
    buffer_size = 1000

    # define map operations
    decode_op = C.Decode()
    # resize_crop_op = C.RandomCropDecodeResize(resize_height, scale=(0.08, 1.0), ratio=(0.75, 1.333))
    horizontal_flip_op = C.RandomHorizontalFlip(prob=0.5)
    rotatin_op = C.RandomRotation(5.0)
    resize_op = C.Resize((image_height, image_width))
    # center_crop = C.CenterCrop(resize_width)
    # rescale_op = C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4)
    normalize_op = C.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                               std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    change_swap_op = C.HWC2CHW()

    cut = C.CutOut(40, num_patches=2)

    if do_train:
        trans = [decode_op, rotatin_op, resize_op, horizontal_flip_op, normalize_op, change_swap_op]
    else:
        trans = [decode_op, resize_op, normalize_op, change_swap_op]

    type_cast_op = C2.TypeCast(mstype.int32)

    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=1)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=1)

    # apply shuffle operations
    data_set = data_set.shuffle(buffer_size=buffer_size)

    # apply batch operations
    data_set = data_set.batch(batxhsize, drop_remainder=True)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set


if __name__ == '__main__':
    data_set = ds.ImageFolderDataset(dataset_dir='C:/dataset/ew_Mar/train', num_parallel_workers=1, shuffle=True,
                                     extensions=['.jpg'])
    iter = data_set.create_dict_iterator()
    # marset=create_dataset('C:/dataset/ew_Mar/train')
    for i in iter:
        print(type(i))

    print(data_set.dataset_size)
