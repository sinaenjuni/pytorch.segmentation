from torch import nn
from torch.nn import Sequential
from torchvision import models
# from torchsummary import summary
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor

def conv3(in_channel:int, out_channel:int, stride:int=1, groups:int=1, dilation:int=1)->nn.Conv2d:
    return nn.Conv2d(in_channel,
                     out_channel,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)

def conv1(in_channel:int, out_channel:int, stride:int=1)->nn.Conv2d:
    return nn.Conv2d(in_channel,
                     out_channel,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class DownsampleBlock(nn.Module):
    def __init__(self,
                 in_channel:int,
                 out_channel:int,
                 stride:int = 1,
                 norm_layer:Optional[Callable[..., nn.Module]] = None,
                 skip_module: Optional[nn.Module] = None):
        super().__init__()
        self.skip_module = skip_module
        self.stride = stride

        self.conv1 = conv3(in_channel, out_channel, stride)
        self.bn1 = norm_layer(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3(out_channel, out_channel)
        self.bn2 = norm_layer(out_channel)

    def forward(self, x: Tensor) -> Tensor:
        skip_connection = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.skip_module is not None:
            skip_connection = self.skip_module(x)

        out += skip_connection
        out = self.relu(out)

        return out

# down = DownsampleBlock(64, 64, norm_layer=nn.BatchNorm2d).cuda()
# summary(down, input_size=(64, 256, 256), device='cuda')


class UpsampleBlock(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 stride:int=1,
                 norm_layer:Optional[Callable[..., nn.Module]] = None,
                 skip_module: Optional[nn.Module] = None):
        super(UpsampleBlock, self).__init__()
        self.skip_module = skip_module

        self.upConv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv1 = conv3(in_channel, out_channel)
        self.bn1 = norm_layer(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3(out_channel, out_channel)
        self.bn2 = norm_layer(out_channel)

    def forward(self, x):
        if self.skip_module is not None:
            x = self.upConv(x)

        skip_connection = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.skip_module is not None:
            skip_connection = self.skip_module(x)

        out += skip_connection
        out = self.relu(out)

        return out

# tt = UpsampleLayer(64, 32)
# print(tt)
# summary(tt, input_size=(64, 32, 32), device='cpu')

class DoubleResNet(nn.Module):
    def __init__(self,
                 upsample_blocks:list=None,
                 downsample_blocks:list=None,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()

        assert upsample_blocks is not None, "upsample_block should not be None"
        assert downsample_blocks is not None, "downsample_blocks should not be None"

        self._norm_layer = norm_layer
        self.in_channel = 64
        self.dilation = 1

        # self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.in_channel)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.encoder0 = self._make_layer(DownsampleBlock, 3, 64, downsample_blocks[0], stride=2)

        self.encoder1 = self._make_layer(DownsampleBlock, 64, 64, downsample_blocks[0], stride=2)
        self.encoder2 = self._make_layer(DownsampleBlock, 64, 128, downsample_blocks[1], stride=2)
        self.encoder3 = self._make_layer(DownsampleBlock, 128, 256, downsample_blocks[2], stride=2)
        self.encoder4 = self._make_layer(DownsampleBlock, 256, 512, downsample_blocks[3], stride=2)

        self.decoder5 = self._make_layer(UpsampleBlock, 512, 256, upsample_blocks[0], is_upsample=True)
        self.decoder4 = self._make_layer(UpsampleBlock, 256, 128, upsample_blocks[1], is_upsample=True)
        self.decoder3 = self._make_layer(UpsampleBlock, 128, 64, upsample_blocks[2], is_upsample=True)
        self.decoder2 = self._make_layer(UpsampleBlock, 64, 32, upsample_blocks[3], is_upsample=True)
        self.decoder1 = self._make_layer(UpsampleBlock, 32, 19, upsample_blocks[4], is_upsample=True)


        self.laset_conv = nn.Conv2d(19, 19, kernel_size=1, bias=False)


    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        x = self.encoder0(x)

        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)

        x = self.decoder5(x)
        x = self.decoder4(x)
        x = self.decoder3(x)
        x = self.decoder2(x)
        x = self.decoder1(x)

        x = self.laset_conv(x)
        return x

    def _make_layer(self,
                    block: Type[Union[UpsampleBlock, DownsampleBlock]],
                    in_channel:int,
                    out_channel: int,
                    blocks: int,
                    stride: int = 1,
                    is_upsample:bool = False,
                    dilate: bool = False) -> nn.Sequential:

        norm_layer = self._norm_layer
        skip_module = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1:
            skip_module = nn.Sequential(
                conv1(in_channel, out_channel, stride),
                norm_layer(out_channel),
            )

        if is_upsample == True:
            skip_module = nn.Sequential(
                conv1(in_channel, out_channel),
                norm_layer(out_channel),
            )

        layers = []
        layers.append(block(in_channel=in_channel,
                            out_channel=out_channel,
                            stride=stride,
                            norm_layer=norm_layer,
                            skip_module=skip_module))

        for _ in range(1, blocks):
            layers.append(block(in_channel=out_channel,
                                out_channel=out_channel,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


DRNet = DoubleResNet(upsample_blocks = [2,2,2,2,2],
                    downsample_blocks = [2,2,2,2],
                    norm_layer = nn.BatchNorm2d).cuda()
# summary(DRNet, input_size=(3, 512, 512), device='cuda')

if __name__ == '__main__':
    import sys

    sys.path.insert(0, '/home/sin/code/Face_sag/ours/')

    from data.O_dataset import ODataset
    from torchvision.transforms import Compose, ToTensor, Resize, RandomCrop
    from torch.utils.data import random_split
    from torch.utils.data import DataLoader

    from loss.O_loss import OhemCELoss
    from torchsummary import summary

    dataset = ODataset('/home/sin/dataset/celebA/O_img',
                       '/home/sin/dataset/celebA/O_mask',
                       transforms=Compose([ToTensor(),
                                           Resize(512),
                                           # RandomCrop(400),
                                           # CenterCrop(300)
                                           ]))

    train_set, test_set = random_split(dataset, [21000, 9000])
    lo = DataLoader(train_set, batch_size=16)
    it = iter(lo)
    img, mask = next(it)
    model = DoubleResNet(upsample_blocks = [2,2,2,2,2],
                    downsample_blocks = [2,2,2,2],
                    norm_layer = nn.BatchNorm2d).to('cuda:1')

    # summary(model, input_size=(3, 512, 512), device='cuda')
    # criteria1 = OhemCELoss(thresh=0.7, n_min=512 * 512).cuda()
    pred = model(img.to('cuda:1'))
    mask = mask.to('cuda:1')
    # loss = criteria1(pred, mask)
    loss = F.cross_entropy(pred, mask)
    print(loss)
