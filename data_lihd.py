import os
import numpy as np
from PIL import Image
from mindspore import Tensor
from mindspore.train.model import Model
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
from mindspore.dataset import samplers
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.transforms.py_transforms import Compose


def get_trans_train():

    decode_op = py_vision.Decode()

    horizontal_flip_op = py_vision.RandomHorizontalFlip(prob=0.5)
    resize_op = py_vision.Resize([384, 128])
    randomerasing = py_vision.RandomErasing(prob=0.5, value=0)
    normalize_op = py_vision.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
    change_swap_op = py_vision.HWC2CHW()
    totensor = py_vision.ToTensor()

    # return Compose([resize_op, horizontal_flip_op, totensor, normalize_op, randomerasing])
    return Compose([decode_op, resize_op, totensor, normalize_op])

def get_trans_test():

    decode_op = py_vision.Decode()

    horizontal_flip_op = py_vision.RandomHorizontalFlip(prob=0.5)
    # randomrotation_op = py_vision.RandomRotation()
    resize_op = py_vision.Resize([384, 128])
    randomerasing = py_vision.RandomErasing(prob=0.5, value=0)
    normalize_op = py_vision.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
    change_swap_op = py_vision.HWC2CHW()
    totensor = py_vision.ToTensor()

    return Compose([decode_op, resize_op, totensor])


class Market:
    """
    person Reid dataset interface
    """

    def __init__(
            self,
            root,

            **kwargs):
        '''

        :param root:
        直接读取文件夹下所有的文件名就行
        '''
        self.rootpath = root
        self.imgpath = []
        # self.trans = trans
        for name in os.listdir(self.rootpath):
            if os.path.splitext(name)[1] == '.jpg':
                self.imgpath.append(name)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the index of the target class
        """
        imgname = self.imgpath[index]
        imgname = os.path.join(self.rootpath, imgname)
        imgs = Image.open(imgname).convert('RGB')

        filename = os.path.basename(self.imgpath[index])
        label = filename[0:4]

        if label[0:2] == '-1':
            target = -1
        else:
            target = int(label)

        return (imgs, target)

    # useless for personal batch sampler
    def __len__(self):
        return len(self.imgpath)


def create_dataset_py(dataset_path, do_train=True, repeat_num=1, batxhsize=1, image_height=384, image_width=128,
                      platform="CPU"):

    trans_train = get_trans_train()
    trans_test = get_trans_test()
    market_train = Market(dataset_path)
    market_test = Market(dataset_path)

    if platform == "Ascend":

        rank_size = int(os.getenv("RANK_SIZE", '1'))
        rank_id = int(os.getenv("RANK_ID", '0'))
        if rank_size == 1:
            data_set = ds.GeneratorDataset(market_train, column_names=['image', 'label'],
                                           sampler=samplers.RandomSampler(),
                                           num_parallel_workers=8, shuffle=None)
        else:
            data_set = ds.GeneratorDataset(market_test, column_names=['image', 'label'],
                                           sampler=samplers.RandomSampler(),
                                           num_parallel_workers=8, shuffle=None)
    elif platform == "GPU":
        if do_train:

            data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True)
        else:
            data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True)
    elif platform == "CPU":
        data_set = ds.ImageFolderDataset(dataset_dir=dataset_path, num_parallel_workers=1, shuffle=True)

    buffer_size = 1000

    decode_op = C.Decode()

    rotation = C.RandomRotation(5.0)

    horizontal_flip_op = C.RandomHorizontalFlip(prob=0.5)

    resize_op = C.Resize((image_height, image_width))

    normalize_op = C.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                               std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    # random_erasing = py.RandomErasing(value='random')
    change_swap_op = C.HWC2CHW()

    if do_train:
        # trans = [decode_op, rotation, resize_op, horizontal_flip_op, normalize_op, change_swap_op]
        # trans = [decode_op, py_vision.ToPIL()]
        trans = trans_train
    else:
        # trans = [decode_op, resize_op, normalize_op, change_swap_op]
        # trans = [decode_op]
        trans = trans_test

    type_cast_op = C2.TypeCast(mstype.int32)

    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)

    # apply shuffle operations
    data_set = data_set.shuffle(buffer_size=buffer_size)

    # apply batch operations
    data_set = data_set.batch(batxhsize, drop_remainder=True)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set
