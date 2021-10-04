import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchsummary import summary
import torch.onnx

class EM(nn.Module):
    def __init__(self, in_channel, out_channel, downsample, dilation, groups):
        super().__init__()
        self.isResConv = in_channel != out_channel

        moodules = [nn.Conv2d(in_channel,
                              out_channel,
                              kernel_size=3,
                              stride= 2 if downsample else 1,
                              padding=dilation,
                              groups=groups,
                              bias=False,
                              dilation=dilation),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(),
                    nn.Conv2d(out_channel, out_channel,
                                           kernel_size=3,
                                           stride=1,
                                           padding=dilation,
                                           groups=groups,
                                           bias=False,
                                           dilation=dilation),
                    nn.BatchNorm2d(out_channel)]

        self.skip_conv = nn.Conv2d(in_channel,
                                   out_channel,
                                   kernel_size=1,
                                   stride=2 if downsample else 1,
                                   bias=False)

        self.block = nn.Sequential(*moodules)

    
    def forward(self, x: Tensor) -> Tensor:
        if self.isResConv:
            return F.relu(self.block(x) + self.skip_conv(x))
        else:
            return F.relu(self.block(x) + x)

if __name__ == '__main__':
    module = [EM(3, 32, False, 1, 1).to(torch.device('cuda'))]
    module += [EM(32, 32, False, 1, 1).to(torch.device('cuda'))]
    module += [EM(32, 64, True, 1, 1).to(torch.device('cuda'))]
    module += [EM(64, 64, False, 1, 1).to(torch.device('cuda'))]
    model = nn.Sequential(*module)
    # summary(model, (3, 32, 32), device='cuda')
    x = model(torch.empty(1, 3, 32, 32, dtype = torch.float32).to(torch.device('cuda')))
    print(model, x.size())
    # print(module)

    for i in model.modules():
        print(i)

    # dummy_data = torch.empty(1, 3, 32, 32, dtype = torch.float32).to(torch.device('cuda'))
    # torch.onnx.export(model, dummy_data, "./output.onnx")

# if __name__ == '__main__':
#     module = nn.Sequential(EM(3,  32, False, 1, 1).to(torch.device('cuda')),
#                            EM(32, 32, False, 1, 1).to(torch.device('cuda')))
#     summary(module, (3, 32, 32), device='cuda')
#     x = module(torch.randn((1, 3, 32, 32)).to(torch.device('cuda')))
#     print(module, x.size())
#     # print(module)