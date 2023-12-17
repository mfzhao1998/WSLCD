import os
import PIL
from PIL import Image
import numpy as np

from torch.utils import data

from datasets.data_utils import CDDataAugmentation
from torch.utils.data.dataset import Dataset
from os.path import join
import torch
import torchvision.transforms as transforms
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png','.tif', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)
    return result
def get_transform(convert=True, normalize=False):
    transform_list = []
    if convert:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
        # transform_list += [transforms.Normalize((0.485, 0.456, 0.406),
        #                                         (0.229, 0.224, 0.225))]
    return transforms.Compose(transform_list)
class LoadDatasetFromFolder(Dataset):
    def __init__(self, suffix, hr1_path, hr2_path, lab_path, img_size=256, is_train=False):
        super(LoadDatasetFromFolder, self).__init__()
        # 获取图片列表
        datalist = [name for name in os.listdir(hr1_path) for item in suffix if
                      os.path.splitext(name)[1] == item]
        self.hr1_filenames = [join(hr1_path, x) for x in datalist if is_image_file(x)]
        self.hr2_filenames = [join(hr2_path, x) for x in datalist if is_image_file(x)]
        self.lab_filenames = [join(lab_path, x) for x in datalist if is_image_file(x)]
        self.img_size=img_size
        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_random_rot=True
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.img_size
            )

    def __getitem__(self, index):
        hr1_img = np.asarray(Image.open(self.hr1_filenames[index]).convert('RGB'))
        hr2_img = np.asarray(Image.open(self.hr2_filenames[index]).convert('RGB'))
        label = np.asarray(Image.open(self.lab_filenames[index]))
        if label.max()>0:
            cl_label=1
        else:
            cl_label=0
        cl_label=torch.tensor(cl_label, dtype=torch.long)
        label = label // 255
        [hr1_img, hr2_img], [label] = self.augm.transform([hr1_img, hr2_img], [label], to_tensor=True)
        image_name =  self.hr1_filenames[index].split('\\', -1)[1]
        return hr1_img, hr2_img, cl_label, make_one_hot(label.long().unsqueeze(0),2).squeeze(0), image_name

    def __len__(self):
        return len(self.hr1_filenames)

class LoadMSFDatasetFromFolder(LoadDatasetFromFolder):
    def __init__(self, suffix, hr1_path, hr2_path, lab_path,scales,img_size):
        super().__init__(suffix, hr1_path, hr2_path, lab_path)
        self.scales=scales
        self.transform = get_transform(convert=True, normalize=True)  # convert to tensor and normalize to [-1,1]
        self.label_transform = get_transform()  # only convert to tensor
        self.img_size=img_size
    def __getitem__(self, index):
        hr1_img = Image.open(self.hr1_filenames[index]).convert('RGB')
        hr2_img = Image.open(self.hr2_filenames[index]).convert('RGB')
        label=Image.open(self.lab_filenames[index])
        if hr1_img.size[0]!=self.img_size:
            hr1_img = hr1_img.resize((self.img_size,self.img_size), resample=PIL.Image.CUBIC)
            hr2_img = hr2_img.resize((self.img_size,self.img_size), resample=PIL.Image.CUBIC)
            label = label.resize((self.img_size,self.img_size), resample=PIL.Image.NEAREST)   
        label_np = np.asarray(label)
        if label_np.max()>0:
            cl_label=1
        else:
            cl_label=0
        cl_label=torch.tensor(cl_label, dtype=torch.long)
        seg_label = self.label_transform(label)
        seg_label = make_one_hot(seg_label.unsqueeze(0).long(), 2).squeeze(0)

        image_name =  self.hr1_filenames[index].split('\\', -1)[1]

        ms_img1_list = []
        ms_img2_list = []
        for s in self.scales:
            target_size = (round(hr1_img.size[0]*s),
                           round(hr2_img.size[1]*s))
            s_img1 = hr1_img.resize(target_size, resample=PIL.Image.CUBIC)
            s_img2 = hr2_img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img1_list.append(s_img1)
            ms_img2_list.append(s_img2)

        msf_img1_list = []
        msf_img2_list = []
        for i in range(len(ms_img1_list)):
            msf_img1_list.append(self.transform(ms_img1_list[i]))
            msf_img1_list.append(self.transform(np.flip(ms_img1_list[i], -1).copy()))
            msf_img2_list.append(self.transform(ms_img2_list[i]))
            msf_img2_list.append(self.transform(np.flip(ms_img2_list[i], -1).copy()))

        return msf_img1_list,  msf_img2_list, cl_label, seg_label, image_name

    def __len__(self):
        return len(self.hr1_filenames)