

import numpy as np
import os
import argparse
import random
from torch.utils.data import DataLoader
from blend_dataset import BlendingDataset
from model import *
from torch.nn import functional as F
from torchvision.utils import save_image
# import vessl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NGPU = torch.cuda.device_count()
# vessl.init()

class Trainer:
    def __init__(self):
        '''Command Line Arguments'''
        parser = argparse.ArgumentParser(description='Train Blending GAN')
        parser.add_argument('--nef', type=int, default=64, help='# of base filters in encoder')
        parser.add_argument('--ngf', type=int, default=64, help='# of base filters in decoder')
        parser.add_argument('--nc', type=int, default=3, help='# of output channels in decoder')
        parser.add_argument('--nBottleneck', type=int, default=4000, help='# of output channels in encoder')
        parser.add_argument('--ndf', type=int, default=64, help='# of base filters in D')

        parser.add_argument('--lr_d', type=float, default=0.0002, help='Learning rate for Critic, default=0.0002')
        parser.add_argument('--lr_g', type=float, default=0.002, help='Learning rate for Generator, default=0.002')
        parser.add_argument('--beta1', type=float, default=0.5, help='Beta for Adam, default=0.5')
        parser.add_argument('--l2_weight', type=float, default=0.999, help='Weight for l2 loss, default=0.999')

        parser.add_argument('--gpu', type=int, default=0, help='GPU ID (negative value indicates CPU)')
        parser.add_argument('--n_epoch', type=int, default=25, help='# of epochs to train for')

        parser.add_argument('--data_root', required=True, help='Path to dataset')
        parser.add_argument('--bg_dir', default='bg', help='background folder')
        parser.add_argument('--obj_dir', default='smoke', help='object folder')

        parser.add_argument("--save_img_path", default="./saved_imgs", help='save folder')
        parser.add_argument("--save_model_path", default="./save_weights")
        parser.add_argument('--mode', default='train', help='train or test')

        parser.add_argument('--load_size', type=int, default=64, help='Scale image to load_size')
        parser.add_argument('--ratio', type=float, default=0.5, help='Ratio for center square size v.s. image_size')
        parser.add_argument('--val_ratio', type=float, default=0.05, help='Ratio for validation set v.s. data set')

        parser.add_argument('--d_iters', type=int, default=5, help='# of D iters per each G iter')
        parser.add_argument('--clamp_lower', type=float, default=-0.01, help='Lower bound for clipping')
        parser.add_argument('--clamp_upper', type=float, default=0.01, help='Upper bound for clipping')

        parser.add_argument('--batch_size', type=int, default=64, help='Input batch size')
        parser.add_argument('--test_size', type=int, default=64, help='Batch size for testing')

        parser.add_argument('--manual_seed', type=int, default=5, help='Manul seed')

        parser.add_argument("--size", type=int, default=256, help="image sizes for the model")
        parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor for the model. config-f = 2, else = 1")
        parser.add_argument("--pretrained",  default=False, help="use pretrained model")

        self.args = parser.parse_args()

        '''generated image save folder'''
        if not os.path.isdir(self.args.save_img_path):
            os.mkdir(self.args.save_img_path)
        '''trained model save folder'''
        if not os.path.isdir(self.args.save_model_path):
            os.mkdir(self.args.save_model_path)
            os.mkdir(os.path.join(self.args.save_model_path, "g"))
            os.mkdir(os.path.join(self.args.save_model_path, "d"))
        
        '''random seed'''
        random.seed(self.args.manual_seed)
        torch.manual_seed(self.args.manual_seed)
        np.random.seed(self.args.manual_seed)

        '''loss'''
        self.criterion = F.mse_loss

        '''model'''
        self.generator = EncoderDecoder(self.args.nef, self.args.ngf, self.args.nc, self.args.nBottleneck,
                                        image_size=self.args.size, conv_init=init_conv, bn_init=init_bn).to(device)
        #load pretrained model
        if self.args.pretrained:
            print("use pretrained model")
            self.load_weights(self.generator, "./blending_gan.npz")
        self.discriminator = DCGAN_D(self.args.size, self.args.ndf, conv_init=init_conv, bn_init=init_bn).to(device)

        '''optimizer'''
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr_d)
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr_g)

        if (device.type == "cuda") and (NGPU > 1):
            print("Multi Gpu activate")
            self.generator = torch.nn.DataParallel(self.generator, device_ids=list(range(NGPU)))
            self.discriminator = torch.nn.DataParallel(self.discriminator, device_ids=list(range(NGPU)))

        '''dataloader'''
        self.dataset = BlendingDataset(self.args.data_root, bg_folder=self.args.bg_dir, obj_folder=self.args.obj_dir,
                                       mode=self.args.mode, load_size=self.args.size)
        self.loader = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    def save_img(self, img, dst_path, idx=0):
        save_image(img, "%s/%d.png"%(dst_path, idx), nrow=4)

    def load_weights(self, net, path):
        params = net.state_dict()
        pretrained_weights = np.load(path, allow_pickle=True)
        with torch.no_grad(): 
            for key in params:
                if "weight" == key[-6:]:
                    npkey = key[:-7].split('.')
                    if npkey[0] == 'bn':
                        npkey = '/'.join(npkey)
                    else:
                        npkey.pop(1)
                        npkey = '/'.join(npkey)
                    npkey1 = npkey + '/W'
                    npkey2 = npkey + '/gamma'
                    # npkey = bytes(npkey, "utf-8")
                    if npkey1 in pretrained_weights.keys():
                        print("Weight found for " + npkey1 + " layer")
                        params[key].copy_(torch.from_numpy(pretrained_weights[npkey1]).type(torch.Tensor))
                    if npkey2 in pretrained_weights.keys():
                        print("Weight found for " + npkey2 + " layer")
                        params[key].copy_(torch.from_numpy(pretrained_weights[npkey2]).type(torch.Tensor))
                if "running_var" == key[-11:]:
                    npkey = key[:-12].split('.')
                    if npkey[0] == 'bn':
                        npkey = '/'.join(npkey)
                    else:
                        npkey.pop(1)
                        npkey = '/'.join(npkey)
                    npkey = npkey + '/avg_var'
                    # npkey = bytes(npkey, "utf-8")
                    if npkey in pretrained_weights.keys():
                        print("Weight found for " + npkey + " layer")
                        params[key].copy_(torch.from_numpy(pretrained_weights[npkey]).type(torch.Tensor))
                if "running_mean" == key[-12:]:
                    npkey = key[:-13].split('.')
                    if npkey[0] == 'bn':
                        npkey = '/'.join(npkey)
                    else:
                        npkey.pop(1)
                        npkey = '/'.join(npkey)
                    npkey = npkey + '/avg_mean'
                    # npkey = bytes(npkey, "utf-8")
                    if npkey in pretrained_weights.keys():
                        print("Weight found for " + npkey + " layer")
                        params[key].copy_(torch.from_numpy(pretrained_weights[npkey]).type(torch.Tensor))
                if "bias" == key[-4:]:
                    npkey = key[:-5].split('.')
                    if npkey[0] == 'bn':
                        npkey = '/'.join(npkey)
                    else:
                        npkey.pop(1)
                        npkey = '/'.join(npkey)
                    npkey = npkey + '/beta'
                    # npkey = bytes(npkey, "utf-8")
                    if npkey in pretrained_weights.keys():
                        print("Weight found for " + npkey + " layer")
                        params[key].copy_(torch.from_numpy(pretrained_weights[npkey]).type(torch.Tensor))

    '''train'''
    def train(self):
        total_iter = int(len(self.loader.dataset)/self.args.batch_size)
        cnt_iter = 1
        best_loss  = 10000
        show_imgs = []
        for epoch in range(self.args.n_epoch):
            for copy_paste, bg_cropped in self.loader:
                copy_paste = copy_paste.detach().to(device)
                bg_cropped = bg_cropped.detach().to(device)
                if cnt_iter < 25 or cnt_iter % 500 ==0:
                   Diters = 10
                else:
                   Diters = 1

                '''train discriminator first'''
                self.generator.requires_grad_(False)
                self.discriminator.requires_grad_(True)

                for _ in range(Diters):
                    fake_img = self.generator(copy_paste)
                    real_pred = self.discriminator(bg_cropped)

                    fake_pred = self.discriminator(fake_img)
                    #loss_d = real_pred- fake_pred
                    loss_d = F.softplus(-real_pred).mean() + F.softplus(fake_pred).mean()
                    self.optimizer_d.zero_grad()
                    loss_d.backward()
                    self.optimizer_d.step()

                '''train generator'''
                self.generator.requires_grad_(True)
                self.discriminator.requires_grad_(False)
                fake_img = self.generator(copy_paste)
                err = self.discriminator(fake_img)
                err = F.softplus(-err).mean()
                loss_g = (1-self.args.l2_weight) * err + \
                         self.args.l2_weight*self.criterion(fake_img, bg_cropped)
                self.optimizer_g.zero_grad()
                loss_g.backward()
                self.optimizer_g.step()
                random_i = random.randint(0, len(copy_paste) - 1)
                cnt_iter+=1
                print("Training GAN...(%d/%d) loss_D: %.3f, loss_G: %.3f"%(cnt_iter, total_iter, loss_d, loss_g))
                # show_imgs.append(vessl.Image(fake_img[random_i], caption="fake img"))

                '''save generator result'''
                if (loss_g+loss_d < best_loss):
                    best_loss = loss_g+loss_d
                    #self.save_img(fake_img, self.args.save_img_path, idx=cnt_iter)
                    torch.save(self.generator.state_dict(), os.path.join(self.args.save_model_path,"g")+"/"+str(cnt_iter)+".pt")
                    #torch.save(self.discriminator.state_dict(), os.path.join(self.args.save_model_path, "d")+"/"+str(cnt_iter)+".pt")
                # vessl.log(
                #     step=epoch,
                #     payload={'loss_D': loss_d.item(),
                #              'loss_G': loss_g.item()}
                # )
                # vessl.log({
                #     'fake img': show_imgs
                # })

if __name__ == "__main__":
    root = "./saved_img/"
    trainer = Trainer()
    # summary(trainer.discriminator, (3, 224, 224))
    trainer.train()