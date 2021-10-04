import torch
import torch.nn as nn
from torchsummary import summary

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Encoder(nn.Module):
    def __init__(self, in_channel, padding_type='reflect', norm_layer=nn.BatchNorm2d, n_downsampling=2, fcc=64, n_blocks=9, use_dropout=False):
        super(Encoder, self).__init__()
        if norm_layer == nn.BatchNorm2d:
            use_bias = False
        self.model = [nn.ReflectionPad2d(3),
                    nn.Conv2d(in_channel,
                              fcc,
                              kernel_size=7,
                              padding=0,
                              bias=use_bias),
                    norm_layer(fcc),
                    nn.ReLU(True)]

        self.n_downsampling=n_downsampling
        for i in range(n_downsampling):
            mult = 2 ** i  # 0:1, 1:2
            self.model += [nn.Conv2d(fcc * mult, # 0:64, 1:128
                                     fcc * mult * 2, # 0:128, 1:256
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     bias=use_bias),
                      norm_layer(fcc * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling # 4 : 256
        for i in range(n_blocks):
            self.model += [ResnetBlock(fcc * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        out = self.model(x)
        return out


if __name__ != '__main__':
    encoder = Encoder(in_channel=3, n_downsampling=2, n_blocks=4).to(torch.device('cuda'))
    print(encoder)
    dummy_data = torch.empty(1, 3, 32, 32, dtype = torch.float32).to(torch.device('cuda'))
    # print(dummy_data)
    # print(encoder(dummy_data))
    # torch.onnx.export(encoder, dummy_data, "./output.onnx")
    summary(encoder, (3,256,256))


class Decorder(nn.Module):
    def __init__(self, out_channel, n_downsampling, norm_layer, fcc, use_bias):
        super(Decorder, self).__init__()
        model = []
        for i in range(n_downsampling): # 2
            mult = 2 ** (n_downsampling - i) # 0:4 1:2
            model += [nn.ConvTranspose2d(fcc * mult,
                                         fcc * mult // 2,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         output_padding=1,
                                         bias=use_bias),
                      norm_layer(fcc * mult // 2),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(fcc, out_channel, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out

if __name__ != '__main__':
    decoder = Decorder(19, 2, nn.BatchNorm2d, 64, False).to(torch.device('cuda'))
    dummy_data = torch.empty(1, 256, 8, 8, dtype = torch.float32).to(torch.device('cuda'))
    torch.onnx.export(decoder, dummy_data, "./decoder.onnx")
    # print(decoder(dummy_data))
    # print(decoder)


class Model(nn.Module):
    def __init__(self, in_channel=3, out_channel=19, n_downsampling=2, n_blocks=4, fcc=64, norm_layer=nn.BatchNorm2d):
        super(Model, self).__init__()
        self.encoder = Encoder(in_channel=in_channel,
                               n_downsampling=2,
                               fcc=fcc,
                               n_blocks=n_blocks)
        self.decoder = Decorder(out_channel=out_channel,
                                n_downsampling=n_downsampling,
                                norm_layer=norm_layer,
                                fcc=fcc,
                                use_bias=False)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out

if __name__ != '__main__':
    model=Model(in_channel=3, out_channel=19, n_downsampling=2, n_blocks=4, fcc=64, norm_layer=nn.BatchNorm2d).to(torch.device('cuda'))
    dummy_data = torch.empty(1, 3, 256, 256, dtype = torch.float32).to(torch.device('cuda'))
    torch.onnx.export(model, dummy_data, "./model.onnx")
    # print(decoder(dummy_data))
    # print(decoder)
