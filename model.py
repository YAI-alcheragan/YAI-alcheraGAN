import torch
import torchvision
from torch import nn

def init_conv(array):
#     xp = cuda.get_array_module(array)
    xp = torch.tensor(array) # 오류 날 수 있음
    array[...] = xp.random.normal(loc=0.0, scale=0.02, size=array.shape)


def init_bn(array):
#     xp = cuda.get_array_module(array)
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
        # input is Z, going into a convolution
#         layers.append(L.Deconvolution2D(None, cngf, ksize=4, stride=1, pad=0, initialW=conv_init, nobias=True))
        layers.append(nn.ConvTranspose2d(nBottleneck, cngf, kernel_size=4, stride=1, padding=0, bias=False))
        
        layers.append(nn.BatchNorm2d(cngf))
        layers.append(nn.ReLU())
        csize, cndf = 4, cngf
        while csize < isize // 2:
#             layers.append(L.Deconvolution2D(None, cngf // 2, ksize=4, stride=2, pad=1, initialW=conv_init, nobias=True))
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
        # x = self.layers(x)
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
        # state size. K x 4 x 4
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
    
# import chainer
# import chainer.functions as F
# import chainer.links as L
# from chainer import cuda


# def init_conv(array):
#     xp = cuda.get_array_module(array)
#     array[...] = xp.random.normal(loc=0.0, scale=0.02, size=array.shape)


# def init_bn(array):
#     xp = cuda.get_array_module(array)
#     array[...] = xp.random.normal(loc=1.0, scale=0.02, size=array.shape)


# class ReLU(chainer.Chain):
#     def __init__(self):
#         super(ReLU, self).__init__()

#     def __call__(self, x):
#         return F.relu(x)


# class Tanh(chainer.Chain):
#     def __init__(self):
#         super(Tanh, self).__init__()

#     def __call__(self, x):
#         return F.tanh(x)


# class LeakyReLU(chainer.Chain):
#     def __init__(self):
#         super(LeakyReLU, self).__init__()

#     def __call__(self, x):
#         return F.leaky_relu(x)


# class DCGAN_G(chainer.ChainList):
#     def __init__(self, isize, nc, ngf, conv_init=None, bn_init=None):
#         cngf, tisize = ngf // 2, 4
#         while tisize != isize:
#             cngf = cngf * 2
#             tisize = tisize * 2

#         layers = []
#         # input is Z, going into a convolution
#         layers.append(L.Deconvolution2D(None, cngf, ksize=4, stride=1, pad=0, initialW=conv_init, nobias=True))
#         layers.append(L.BatchNormalization(cngf, initial_gamma=bn_init))
#         layers.append(ReLU())
#         csize, cndf = 4, cngf
#         while csize < isize // 2:
#             layers.append(L.Deconvolution2D(None, cngf // 2, ksize=4, stride=2, pad=1, initialW=conv_init, nobias=True))
#             layers.append(L.BatchNormalization(cngf // 2, initial_gamma=bn_init))
#             layers.append(ReLU())
#             cngf = cngf // 2
#             csize = csize * 2
#         layers.append(L.Deconvolution2D(None, nc, ksize=4, stride=2, pad=1, initialW=conv_init, nobias=True))
#         layers.append(Tanh())

#         super(DCGAN_G, self).__init__(*layers)

#     def __call__(self, x):
#         for i in range(len(self)):
#             x = self[i](x)

#         return x


# class DCGAN_D(chainer.ChainList):
#     def __init__(self, isize, ndf, nz=1, conv_init=None, bn_init=None):
#         layers = []
#         layers.append(L.Convolution2D(None, ndf, ksize=4, stride=2, pad=1, initialW=conv_init, nobias=True))
#         layers.append(LeakyReLU())
#         csize, cndf = isize / 2, ndf
#         while csize > 4:
#             in_feat = cndf
#             out_feat = cndf * 2
#             layers.append(L.Convolution2D(None, out_feat, ksize=4, stride=2, pad=1, initialW=conv_init, nobias=True))
#             layers.append(L.BatchNormalization(out_feat, initial_gamma=bn_init))
#             layers.append(LeakyReLU())

#             cndf = cndf * 2
#             csize = csize / 2
#         # state size. K x 4 x 4
#         layers.append(L.Convolution2D(None, nz, ksize=4, stride=1, pad=0, initialW=conv_init, nobias=True))

#         super(DCGAN_D, self).__init__(*layers)

#     def encode(self, x):
#         for i in range(len(self)):
#             x = self[i](x)

#         return x

#     def __call__(self, x):
#         x = self.encode(x)
#         x = F.sum(x, axis=0) / x.shape[0]
#         return F.squeeze(x)


# class EncoderDecoder(chainer.Chain):
#     def __init__(self, nef, ngf, nc, nBottleneck, image_size=64, conv_init=None, bn_init=None):
#         super(EncoderDecoder, self).__init__(
#             encoder=DCGAN_D(image_size, nef, nBottleneck, conv_init, bn_init),
#             bn=L.BatchNormalization(nBottleneck, initial_gamma=bn_init),
#             decoder=DCGAN_G(image_size, nc, ngf, conv_init, bn_init)
#         )

#     def encode(self, x):
#         h = self.encoder.encode(x)
#         h = F.leaky_relu(self.bn(h))

#         return h

#     def decode(self, x):
#         h = self.decoder(x)

#         return h

#     def __call__(self, x):
#         h = self.encode(x)
#         h = self.decode(h)
#         return h
