import torch.nn.functional as F
import numpy as np
import torch
import os
import argparse
import random
from model_stylegan import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NGPU = torch.cuda.device_count()
num_workers = 4 * NGPU
from torchsummary.torchsummary import summary
import cv2
from torchvision.utils import save_image

class Trainer:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Train Blending GAN')
        # parser.add_argument('--nef', type=int, default=64, help='# of base filters in encoder')
        # parser.add_argument('--ngf', type=int, default=64, help='# of base filters in decoder')
        # parser.add_argument('--nc', type=int, default=3, help='# of output channels in decoder')
        # parser.add_argument('--nBottleneck', type=int, default=4000, help='# of output channels in encoder')
        # parser.add_argument('--ndf', type=int, default=64, help='# of base filters in D')

        parser.add_argument('--lr_d', type=float, default=0.0002, help='Learning rate for Critic, default=0.0002')
        parser.add_argument('--lr_g', type=float, default=0.002, help='Learning rate for Generator, default=0.002')
        parser.add_argument('--beta1', type=float, default=0.5, help='Beta for Adam, default=0.5')
        parser.add_argument('--l2_weight', type=float, default=0.999, help='Weight for l2 loss, default=0.999')

        parser.add_argument('--gpu', type=int, default=0, help='GPU ID (negative value indicates CPU)')
        parser.add_argument('--n_epoch', type=int, default=25, help='# of epochs to train for')

        parser.add_argument('--data_root', help='Path to dataset')
        parser.add_argument('--load_size', type=int, default=64, help='Scale image to load_size')
        parser.add_argument('--ratio', type=float, default=0.5, help='Ratio for center square size v.s. image_size')
        parser.add_argument('--val_ratio', type=float, default=0.05, help='Ratio for validation set v.s. data set')

        parser.add_argument('--d_iters', type=int, default=5, help='# of D iters per each G iter')
        parser.add_argument('--clamp_lower', type=float, default=-0.01, help='Lower bound for clipping')
        parser.add_argument('--clamp_upper', type=float, default=0.01, help='Upper bound for clipping')

        parser.add_argument('--experiment', default='encoder_decoder_blending_result',
                            help='Where to store samples and models')
        parser.add_argument('--test_folder', default='samples', help='Where to store test results')
        parser.add_argument('--workers', type=int, default=4, help='# of data loading workers')
        parser.add_argument('--batch_size', type=int, default=64, help='Input batch size')
        parser.add_argument('--test_size', type=int, default=64, help='Batch size for testing')

        parser.add_argument('--train_samples', type=int, default=150000, help='# of training examples')
        parser.add_argument('--test_samples', type=int, default=256, help='# of testing examples')

        parser.add_argument('--manual_seed', type=int, default=5, help='Manul seed')

        parser.add_argument('--resume', default='', help='Resume the training from snapshot')
        parser.add_argument('--snapshot_interval', type=int, default=1, help='Interval of snapshot (epochs)')
        parser.add_argument('--print_interval', type=int, default=1,
                            help='Interval of printing log to console (iteration)')
        parser.add_argument('--plot_interval', type=int, default=10, help='Interval of plot (iteration)')
        parser.add_argument('--save_result', type=int, default=0, help='1 for save else 0')
        parser.add_argument("--dst_path", type='str', default="./", help='save folder')

        parser.add_argument("--size", type=int, default=256, help="image sizes for the model")
        parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor for the model. config-f = 2, else = 1")

        self.args = parser.parse_args()

        self.args.latent = 512
        self.args.n_mlp = 8
        

        random.seed(self.args.manual_seed)
        torch.manual_seed(7)
        np.random.seed(7)

        if not os.path.isdir(self.args.dst_path):
            os.mkdir(self.args.dst_path)

        '''loss'''
        self.criterion = F.mse_loss

        '''Model'''
        self.generator = EncoderDecoder(self.args.size, self.args.latent, self.args.n_mlp, self.args.channel_multiplier).to(device)
        self.discriminator = StyleGAN_D(self.args.size).to(device)

        '''optimizer'''
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr_d)
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr_g)

        if (device.type == "cuda") and (NGPU > 1):
            print("Multi Gpu activate")
            self.generator = torch.nn.DataParallel(self.generator, device_ids=list(range(NGPU)))
            self.discriminator = torch.nn.DataParallel(self.discriminator, device_ids=list(range(NGPU)))

        '''Data Load'''

    def save_img(self, img, dst_path, idx=0):
        save_image(img, "%s/%d.png"%(dst_path, idx), nrow=4)

    def show_tensor(self, tensor, save=False, index=0):
        x = tensor.permute((0, 2, 3, 1))
        if tensor.shape[0] != 0: # if has batch, show only first image
            x = x[0]
        x = torch.squeeze(x, axis=0)
        x = x.to('cpu').detach().numpy()
        if x.shape[2] == 3:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        if save:
            cv2.imwrite("./%d.jpg"%(index), x*255.0)
        cv2.imshow("tensor", x)
        cv2.waitKey()

    def train(self):
        total_iter = int(len(self.loader)/self.args.batch_size)
        cnt_iter = 1
        best_loss_g  = 10000
        for copy_paste, bg_cropped in self.loader:
            if cnt_iter < 25 or cnt_iter % 500 ==0:
                Diters = 100
            else:
                Diters = self.args.d_iters

            '''train discriminator first'''
            self.generator.requires_grad_(False)
            self.discriminator.requires_grad_(True)
            for _ in range(Diters):
                fake_img = self.generator(copy_paste)
                err_real = self.discriminator(bg_cropped)
                err_fake = self.discriminator(fake_img)
                loss_d = err_real - err_fake
                self.optimizer_d.zero_grad()
                loss_d.backward()
                self.optimizer_d.step()

            '''train generator'''
            self.generator.requires_grad_(True)
            self.discriminator.requires_grad_(False)
            fake_img = self.generator(copy_paste)
            err = self.discriminator(fake_img)
            loss_g = (1-self.args.l2_weight) * err + \
                     self.args.l2_weight*self.criterion(fake_img, bg_cropped)
            self.optimizer_g.zero_grad()
            loss_g.backward()
            self.optimizer_g.step()

            cnt_iter+=1
            print("Training GAN...(%d/%d) loss_D: %.3f, loss_G: %.3f"%(cnt_iter, total_iter, loss_d, loss_g))

            '''save generator result'''
            if self.args.save_img and (loss_g < best_loss_g):
                best_loss_g = loss_g
                self.save_img(fake_img, self.args.dst_path, idx=cnt_iter)

if __name__ == "__main__":
    root = "./saved_img/"
    batch_size = 1
    trainer = Trainer()
    summary(trainer.discriminator, (3, 224, 224))
    trainer.train()



