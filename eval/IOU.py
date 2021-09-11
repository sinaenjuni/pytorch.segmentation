import torch.nn as nn
from models.model import DoubleResNet
from dataset.CelebA import *
import torch.nn.functional as F
from dataset.preprocessing import classes


class IOU:
    def __init__(self):
        self.classes = {v: k for k, v in classes.items()}

    def __call__(self, preds, labels):
        oh_pred = F.one_hot(preds, num_classes=len(self.classes))
        oh_labels = F.one_hot(labels, num_classes=len(self.classes))

        union = torch.logical_or(oh_pred, oh_labels)
        intersection = torch.logical_and(oh_pred, oh_labels)

        union = union.sum((1, 2))
        intersection = intersection.sum((1, 2))

        IOUs = (intersection / union).nan_to_num(nan=1).mean(0)
        mIOU = IOUs.mean(0)
        # for i in range(len(self.classes)):
        #     print(self.classes[i], IOU[..., i])
        # print(mIOU)
        return mIOU, IOUs



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

    train_loader = DataLoader(train_set, batch_size=8, shuffle=False)

    iters = iter(train_loader)
    inputs, labels = iters.next()

    net = DoubleResNet(upsample_blocks = [4,4,4,4,4],
                        downsample_blocks = [3,3,3,3],
                        norm_layer = nn.BatchNorm2d)
    net.to(torch.device('cuda'))
    net.load_state_dict(torch.load('/home/sin/git/pytorch.segmentation/DoubleResNetV2/8.23_17:31:36/models/29000-model:0.2208298146724701.pt'))
    net.eval()

    inputs = inputs.to(torch.device('cuda'))
    labels = labels.to(torch.device('cuda'))
    pred = net(inputs)
    pred = pred.argmax(1)
    print(pred.size())
    print(labels.size())

    iou = IOU()
    iou(pred, labels)

    # pred = denormSeg(pred)
    # img = denormImg(img)
    # seg = denormSeg(seg)
    # target = torch.cat([img.data.cpu(),
    #                     seg.data.cpu(),
    #                     pred.data.cpu()], 2)
    #
    # save_image(target, './seg.png')

    # classes = {v:k for k, v in classes.items()}
    # print(classes)
    #
    # oh_pred = F.one_hot(pred, num_classes = 19)
    # oh_labels = F.one_hot(labels, num_classes = 19)
    #
    # # plt.imshow(oh_pred[...,0].permute(1,2,0))
    # # plt.show()
    # #
    # # plt.imshow(oh_labels[...,0].permute(1,2,0))
    # # plt.show()
    #
    # union = torch.logical_or(oh_pred, oh_labels)
    # intersection = torch.logical_and(oh_pred, oh_labels)
    #
    # union = union.sum((1, 2))
    # intersection = intersection.sum((1, 2))
    #
    # IOU = (intersection/union).nan_to_num(nan=1).mean(0)
    # # IOU = (intersection/union)
    # mIOU = IOU.mean(0)
    # for i in range(len(classes)):
    #     print(classes[i], IOU[...,i])
    # print(mIOU)




    # print(oh_union.size())
    # for i in range(18):
    #     plt.imshow(oh_intersection[...,i].permute(1,2,0))
    #     plt.show()
    #     plt.imshow(oh_union[...,i].permute(1,2,0))
    #     plt.show()
    #
    # print(torch.logical_or(oh_pred[..., 0], oh_labels[..., 0]).size())

    # oh_pred = oh_pred.permute(0,3,1,2)
    # oh_labels = oh_labels.permute(0,3,1,2)
    #
    # print(oh_pred.size(), oh_labels.size())




    # added_
    # for i in range()
    #
    # print((onehot_pred + onehot_seg).size())

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
