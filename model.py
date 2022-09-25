import torch
from torch import nn

def init_conv(array):
    xp = torch.tensor(array)
    array[...] = xp.random.normal(loc=0.0, scale=0.02, size=array.shape)


def init_bn(array):
    xp = torch.tensor(array)
    array[...] = xp.random.normal(loc=1.0, scale=0.02, size=array.shape)


class DCGAN_G(nn.Module):
    def __init__(self, isize, nc, ngf, nBottleneck, conv_init=None, bn_init=None):
        super().__init__()
        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2
        layers = []
        layers.append(nn.ConvTranspose2d(nBottleneck, cngf, kernel_size=4, stride=1, padding=0, bias=False))
        
        layers.append(nn.BatchNorm2d(cngf))
        layers.append(nn.ReLU())
        csize, cndf = 4, cngf
        while csize < isize // 2:
            layers.append(nn.ConvTranspose2d(cngf, cngf // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(cngf // 2))
            layers.append(nn.ReLU())
            cngf = cngf // 2
            csize = csize * 2
        layers.append(nn.ConvTranspose2d(cngf, nc, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def __call__(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x


class DCGAN_D(nn.Module):
    def __init__(self, isize, ndf, nz=1, conv_init=None, bn_init=None):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.LeakyReLU())
        csize, cndf = isize / 2, ndf
        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            layers.append(nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU())

            cndf = cndf * 2
            csize = csize / 2
        # image state size: K x 4 x 4
        layers.append(nn.Conv2d(out_feat, nz, kernel_size=4, stride=1, padding=0, bias=False))
        self.layers = nn.Sequential(*layers)

    def encode(self, x):
        x = self.layers(x)
        return x

    def __call__(self, x):
        x = self.encode(x)
        x = torch.sum(x, axis=0) / x.shape[0]
        return torch.squeeze(x)


class EncoderDecoder(nn.Module):
    def __init__(self, nef, ngf, nc, nBottleneck, image_size=64, conv_init=None, bn_init=None):
        super().__init__()
        self.encoder=DCGAN_D(image_size, nef, nBottleneck, conv_init, bn_init)
        self.bn=nn.BatchNorm2d(nBottleneck)
        self.decoder=DCGAN_G(image_size, nc, ngf, nBottleneck, conv_init, bn_init)
        self.leakyrelu = nn.LeakyReLU()

    def encode(self, x):
        h = self.encoder.encode(x)
        h = self.leakyrelu(self.bn(h))

        return h

    def decode(self, x):
        h = self.decoder(x)

        return h

    def __call__(self, x):
        h = self.encode(x)
        h = self.decode(h)
        return h
    
