import torch
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
from pathlib import Path
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random


def denormImg(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def denormSeg(x):
    x = x / 18
    x = x.unsqueeze(1)
    x = x.repeat(1, 3, 1, 1)
    return x


class PairedDataToTensor:
    def __call__(self, imgs):
        img = imgs[0]
        seg = imgs[1]
        img = TF.to_tensor(img)
        seg = TF.to_tensor(seg) * 255
        return img, seg


class PairedDataNormalization:
    def __call__(self, imgs):
        img = imgs[0]
        seg = imgs[1]
        img = TF.normalize(img, (.5), (.5))
        return img, seg


class PairedDataResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, imgs):
        img = imgs[0]
        seg = imgs[1]
        img = TF.resize(img, self.size)
        seg = TF.resize(seg, self.size)

        return img, seg


class PairedDataRandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, imgs):
        img = imgs[0]
        seg = imgs[1]

        if random.random() < self.prob:
            img = TF.hflip(img)
            seg = TF.hflip(seg)
        # return (img1, img2)
        return img, seg


class PairedDataRandomCrop:
    def __init__(self, prob=0.5, inter_size = None, output_size=(256, 256)):
        self.prob = prob
        self.output_size = output_size
        if inter_size is None:
            self.inter_size = output_size
        else:
            self.inter_size = inter_size

    def __call__(self, imgs):
        img = imgs[0]
        seg = imgs[1]
        if random.random() < self.prob:
            img = TF.resize(img, self.inter_size)
            seg = TF.resize(seg, self.inter_size)

            i, j, h, w = T.RandomCrop.get_params(img, output_size=self.output_size)
            img = TF.crop(img, i, j, h, w)
            seg = TF.crop(seg, i, j, h, w)

        else:
            img = TF.resize(img, self.output_size)
            seg = TF.resize(seg, self.output_size)

        return img, seg



class CelebADataset(Dataset):
    def __init__(self, img_path:str, seg_path:str, transforms):
        self.img_path = Path(img_path)
        self.seg_path = Path(seg_path)

        self.img_list = list(self.img_path.glob('*.jpg'))
        self.seg_list = list(self.seg_path.glob('*.png'))

        self.img_len = len(self.img_list)
        self.seg_len = len(self.seg_list)

        assert not len(self.img_list) != len(self.seg_list), 'data is not available'
        print(f"Found {self.img_len} image and {self.seg_len} segmentation")

        self.transforms = transforms

    def __len__(self):
        return self.img_len

    def __getitem__(self, item):
        img = self.img_list[item]
        seg = self.seg_path / img.name.replace('jpg', 'png')

        img = Image.open(img)
        seg = Image.open(seg)

        img, seg = self.transforms((img, seg))
        seg = seg.type(torch.long)
        seg = torch.squeeze(seg, 0)
        return img, seg



if __name__ == "__main__":

    transforms = T.Compose([
        PairedDataToTensor(),
        PairedDataNormalization(),
        PairedDataResize(512),
        PairedDataRandomHorizontalFlip(prob = .5),
        PairedDataRandomCrop(prob=.5,
                             inter_size = (512, 512),
                             output_size = (512, 512))
    ])

    dataset = CelebADataset(img_path='/home/sin/git/pytorch.segmentation/data/CelebAMask-HQ/CelebA-HQ-img/',
                            seg_path='/home/sin/git/pytorch.segmentation/data/preprocessing-mask/',
                            transforms=transforms)
    train_indicise = [i for i in range(21000)]
    test_indicise = [i for i in range(9000)]
    train_set = Subset(dataset, train_indicise)
    test_set = Subset(dataset, test_indicise)
    print(len(train_set))
    loader = DataLoader(train_set, batch_size=8)
    iters = iter(loader)
    img, seg = iters.next()

    print(img.size())


    # loader = DataLoader(train_set, batch_size=8)
    # iters = iter(loader)
    # img, seg = iters.next()

    from model import DoubleResNet
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    import torch

    # net = DoubleResNet(upsample_blocks = [4,4,4,4,4],
    #                 downsample_blocks = [3,3,3,3],
    #                 norm_layer = nn.BatchNorm2d)
    #
    # pred = net(img)
    # pred = pred.argmax(1)
    # pred = denormSeg(pred)
    # print(pred.size())
    # loss = F.cross_entropy(pred, seg)
    # print(loss)

    # print(img.shape)
    # print(seg.shape)
    # print(img.min(), img.max())
    # print(seg.min(), seg.max())
    # img = denormImg(img)
    # seg = denormSeg(seg)
    # print(img.shape)
    # print(seg.shape)
    # print(img.min(), img.max())
    # print(seg.min(), seg.max())

    # print(img.min(), img.max())
    # print(seg.min(), seg.max())
    #
    # x_concat = torch.cat([img, seg], dim=2)
    # gimg = make_grid(x_concat)
    # plt.imshow(gimg.permute(1,2,0))
    # plt.show()
