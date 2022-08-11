import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self, img_shape=28):
        super().__init__()
        self.layers = nn.Sequential(
            self.discriminator_block(in_ch=1, out_ch=32, num_features=32, kernel_size=4, stride=2, padding=1),
            self.discriminator_block(in_ch=32, out_ch=64, num_features=64, kernel_size=4, stride=2, padding=1),
            self.discriminator_block(in_ch=64, out_ch=128, num_features=128, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=7, stride=1, padding=2, bias=False),
            nn.Sigmoid())
        
    def discriminator_block(self, in_ch, out_ch, num_features, kernel_size=3, stride=2, padding=0):
        layer = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
                              nn.BatchNorm2d(num_features),
                              nn.LeakyReLU(0.2, inplace=True)
                             )
        return layer
    
    def forward(self, image):
        x = self.layers(image)
        return x.view(-1, 1)
    

class Generator(nn.Module):
    def __init__(self, img_shape = 28, z=100):
        super().__init__()
        self.z = z
        self.layers = nn.Sequential(
            self.generator_block(in_ch=z, out_ch=256, num_features=256, kernel_size=7, stride=1, padding=0),
            self.generator_block(in_ch=256, out_ch=128, num_features=128, kernel_size=4, stride=2, padding=1),
            self.generator_block(in_ch=128, out_ch=64, num_features=64, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh())
        
    def generator_block(self, in_ch, out_ch, num_features, kernel_size=4, stride=2, padding=0):
        layer = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
                              nn.BatchNorm2d(num_features),
                              nn.ReLU(inplace=True)
                             )
                                 
        return layer
    
    def forward(self, image):
        x = image.view(-1, self.z, 1, 1)
        x = self.layers(x)
        return x
    
    
class Prior(nn.Module):
    def __init__(self, img_shape=28, z=100):
        super().__init__()
        self.z = z
        self.layers = nn.Sequential(
            self.discriminator_block(in_ch=1, out_ch=img_shape*4, num_features=img_shape*4, kernel_size=4, stride=2, padding=1),
            self.discriminator_block(in_ch=img_shape*4, out_ch=img_shape*8, num_features=img_shape*8, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(in_channels=img_shape*8, out_channels=z, kernel_size=7, stride=2, padding=0, bias=False),
            nn.Sigmoid())
        
    def discriminator_block(self, in_ch, out_ch, num_features, kernel_size=3, stride=2, padding=0):
        layer = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
                              nn.BatchNorm2d(num_features),
                              nn.LeakyReLU(0.2, inplace=True)
                             )
        return layer
    
    def forward(self, image):
        x = self.layers(image)
        return x.squeeze(1)