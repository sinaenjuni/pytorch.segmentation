import torch
from torchvision.utils import make_grid, save_image
import torch.nn as nn
from model import DoubleResNet
from dataset.CelebA import *
import matplotlib.pyplot as plt
import torch.nn.functional as F

if __name__ == '__main__':

    transforms = T.Compose([
        PairedDataToTensor(),
        PairedDataNormalization(),
        PairedDataRandomHorizontalFlip(prob=.5),
        # PairedDataResize(256)
        PairedDataRandomCrop(prob=.5,
                             inter_size=(528, 528),
                             output_size=(512, 512))
        # PairedDataResize(512)
    ])

    dataset = CelebADataset(img_path='/home/sin/git/pytorch.segmentation/data/CelebAMask-HQ/CelebA-HQ-img/',
                            seg_path='/home/sin/git/pytorch.segmentation/data/preprocessing-mask/',
                            transforms=transforms)

    train_indicise = [i for i in range(21000)]
    test_indicise = [i for i in range(9000)]
    train_set = Subset(dataset, train_indicise)
    test_set = Subset(dataset, test_indicise)

    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)


    iters = iter(test_loader)
    img, seg = iters.next()

    net = DoubleResNet(upsample_blocks = [4,4,4,4,4],
                        downsample_blocks = [3,3,3,3],
                        norm_layer = nn.BatchNorm2d)
    net.to(torch.device('cuda'))
    net.load_state_dict(torch.load('/home/sin/git/pytorch.segmentation/DoubleResNetV2/8.23_17:31:36/models/29000-model:0.2208298146724701.pt'))
    net.eval()


    img = img.to(torch.device('cuda'))
    pred = net(img)
    pred = pred.argmax(1)
    print(pred.size())

    # pred = denormSeg(pred)
    # img = denormImg(img)
    # seg = denormSeg(seg)
    # target = torch.cat([img.data.cpu(),
    #                     seg.data.cpu(),
    #                     pred.data.cpu()], 2)
    #
    # save_image(target, './seg.png')

    onehot_pred = F.one_hot(pred, num_classes = 19).cpu()
    onehot_seg = F.one_hot(seg, num_classes = 19).cpu()

    added_
    for i in range()

    print((onehot_pred + onehot_seg).size())

    # print(onehot_seg.size(), onehot_seg.size())

    # print(onehot_seg.eq(onehot_pred).size())

    #
    #
    # import torch.nn as nn
    # ul = nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear')
    # plt.imshow(img[0].permute(1,2,0).data.cpu())
    # plt.show()
    # plt.imshow(ul(img)[0].permute(1,2,0).data.cpu())
    # plt.show()
    #
    # pred = net(img)
    # pred = pred.argmax(1)
    #
    # print(pred.size(), label.size())
    # print(pred.unique(), label.unique())
    #
    # onehot_pred = F.one_hot(pred, num_classes=19)
    # onehot_pred = onehot_pred.squeeze(0)
    #
    # onehot_label = F.one_hot(label, num_classes=19)
    # onehot_label = onehot_label.squeeze(0)
    #
    # print(onehot_label.unique())
    # plt.imshow(onehot_pred[...,1].data.cpu())
    # plt.show()
    #
    # plt.imshow(onehot_label[..., 1].data.cpu())
    # plt.show()
    #
    #
    #
    # img = denormImg(img)
    # pred = denormSeg(pred)
    # label = denormSeg(label)
    #
    # x_concat = torch.cat((img, label, pred), dim=2)
    # grid = make_grid(x_concat.data.cpu())
    # plt.imshow(grid.permute(1,2,0))
    # plt.show()
