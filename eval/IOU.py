
from torchvision.utils import make_grid
import torch.nn as nn
from model import DoubleResNet
from dataset.CelebA import *
import matplotlib.pyplot as plt
import torch.nn.functional as F

if __name__ == '__main__':

    net = DoubleResNet(upsample_blocks = [4,4,4,4,4],
                        downsample_blocks = [3,3,3,3],
                        norm_layer = nn.BatchNorm2d)
    net.to(torch.device('cuda'))
    net.load_state_dict(torch.load('/home/sin/git/pytorch.segmentation/DoubleResNetV2/8.23_17:31:36/models/29000-model:0.2208298146724701.pt'))
    net.eval()

    transforms = T.Compose([
        PairedDataToTensor(),
        PairedDataNormalization(),
        PairedDataRandomHorizontalFlip(prob=.5),
        PairedDataRandomCrop(prob=.5,
                             inter_size=(512, 512),
                             output_size=(256, 256))
        # PairedDataResize(512)

    ])
    dataset = CelebADataset(img_path='/home/sin/git/pytorch.segmentation/data/CelebAMask-HQ/CelebA-HQ-img/',
                            seg_path='/home/sin/git/pytorch.segmentation/data/preprocessing-mask/',
                            transforms=transforms)

    train_indicise = [i for i in range(21000)]
    test_indicise = [i for i in range(9000)]
    train_set = Subset(dataset, train_indicise)
    test_set = Subset(dataset, test_indicise)

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    iters = iter(test_loader)
    img, label = iters.next()
    img = img.to(torch.device('cuda'))
    label = label.to(torch.device('cuda'))
    print(img.size(), label.size())

    pred = net(img)
    pred = pred.argmax(1)

    print(pred.size(), label.size())
    print(pred.unique(), label.unique())

    onehot_pred = F.one_hot(pred, num_classes=19)
    onehot_pred = onehot_pred.squeeze(0)

    onehot_label = F.one_hot(label, num_classes=19)
    onehot_label = onehot_label.squeeze(0)

    print(onehot_label.unique())
    plt.imshow(onehot_pred[...,1].data.cpu())
    plt.show()

    plt.imshow(onehot_label[..., 1].data.cpu())
    plt.show()



    img = denormImg(img)
    pred = denormSeg(pred)
    label = denormSeg(label)

    x_concat = torch.cat((img, label, pred), dim=2)
    grid = make_grid(x_concat.data.cpu())
    plt.imshow(grid.permute(1,2,0))
    plt.show()
    # print(net)
    # print('True')