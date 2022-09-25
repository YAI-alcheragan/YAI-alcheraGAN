import argparse
import os
import numpy as np

# import chainer
# from chainer import cuda, serializers
import torch
from torch import Tensor

from skimage import img_as_float
from skimage.io import imread, imsave

from gp_gan import gp_gan
from model import EncoderDecoder
from blend_dataset import BlendingDataset
import cv2


basename = lambda path: os.path.splitext(os.path.basename(path))[0]

"""
    Note: source image, destination image and mask image have the same size.
"""

def show_tensor(tensor, save=False, index=0):
    """
    check image
    """
    x = tensor.permute((0, 2, 3, 1))
    if tensor.shape[0] != 0:  # if has batch, show only first image
        x = x[0]
    x = torch.squeeze(x, axis=0)
    x = x.to('cpu').detach().numpy()
    if x.shape[2] == 3:
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    if save:
        cv2.imwrite("./%d.jpg" % (index), x * 255.0)
    cv2.imshow("tensor", x)
    cv2.waitKey()



def load_weights(net, path):
    """
    load pretrained/finetuned model weights from 'path' to 'net'
    """
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
                    params[key].copy_(torch.from_numpy(pretrained_weights[npkey1]).type(Tensor))
                if npkey2 in pretrained_weights.keys():
                    print("Weight found for " + npkey2 + " layer")
                    params[key].copy_(torch.from_numpy(pretrained_weights[npkey2]).type(Tensor))
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
                    params[key].copy_(torch.from_numpy(pretrained_weights[npkey]).type(Tensor))
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
                    params[key].copy_(torch.from_numpy(pretrained_weights[npkey]).type(Tensor))
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
                    params[key].copy_(torch.from_numpy(pretrained_weights[npkey]).type(Tensor))

def main():
    '''Command Line Arguments'''
    parser = argparse.ArgumentParser(description='Gaussian-Poisson GAN for high-resolution image blending')
    parser.add_argument('--nef', type=int, default=64, help='# of base filters in encoder')
    parser.add_argument('--ngf', type=int, default=64, help='# of base filters in decoder or G')
    parser.add_argument('--nc', type=int, default=3, help='# of output channels in decoder or G')
    parser.add_argument('--nBottleneck', type=int, default=4000, help='# of output channels in encoder')
    parser.add_argument('--ndf', type=int, default=64, help='# of base filters in D')

    parser.add_argument('--image_size', type=int, default=64, help='The height / width of the input image to network')

    parser.add_argument('--color_weight', type=float, default=1, help='Color weight')
    parser.add_argument('--sigma', type=float, default=0.5,
                        help='Sigma for gaussian smooth of Gaussian-Poisson Equation')
    parser.add_argument('--gradient_kernel', type=str, default='normal', help='Kernel type for calc gradient')
    parser.add_argument('--smooth_sigma', type=float, default=1, help='Sigma for gaussian smooth of Laplacian pyramid')

    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--g_path', default='./blending_gan.npz', help='Path for pretrained Blending GAN model')
    parser.add_argument('--unsupervised_path', default='models/unsupervised_blending_gan.npz',
                        help='Path for pretrained unsupervised Blending GAN model')
    parser.add_argument('--list_path', default='',
                        help='File for input list in csv format: obj_path;bg_path;mask_path in each line')

    parser.add_argument('--root', default='', help='Path for images : root/test/bg , root/test/smoke')
    parser.add_argument('--blended_folder', default='', help='Where to save blended image')

    args = parser.parse_args()

    print('Input arguments:')
    for key, value in vars(args).items():
        print('\t{}: {}'.format(key, value))
    print('')

    '''Device Setting'''
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    '''Load Pretrained/Finetuned Model'''
    G = EncoderDecoder(args.nef, args.ngf, args.nc, args.nBottleneck, image_size=args.image_size)
    print('Load pretrained Blending GAN model from {} ...'.format(args.g_path))
    # Load Pretrained Model
    load_weights(G, args.g_path)
    # Load Finetuned Model
    # G.load_state_dict(torch.load(args.g_path))
    G.to(device)

    '''Generated Image Save Folder'''
    if not os.path.isdir(args.blended_folder):
        os.makedirs(args.blended_folder)
    print('Result will save to {} ...\n'.format(args.blended_folder))

    '''Data Loader'''
    bg_folder, obj_folder = "bg", "smoke"       # obj1.jpg, obj1.json ... in /root/test/obj_folder
    test_loader = torch.utils.data.DataLoader(BlendingDataset(root= args.root, bg_folder=bg_folder, obj_folder=obj_folder, mode="test", load_size=128),
                                            batch_size=1, 
                                            shuffle=False,
                                            drop_last=False)

    '''Evaluation'''
    with torch.no_grad():
        G.eval()

        for idx, batch in enumerate(test_loader):
            # show_tensor(batch["obj"])
            # show_tensor(batch["mask"])
            # show_tensor(batch["cp"])
            print('Processing {}/{} ...'.format(idx + 1, len(test_loader)))
            obj, bg, mask, cp = torch.squeeze(batch["obj"]).cpu().numpy(), \
            torch.squeeze(batch["bg"]).cpu().numpy(), torch.squeeze(batch["mask"]).cpu().numpy(), \
                                torch.squeeze(batch["cp"]).cpu().numpy()
            obj, bg, mask, cp = np.transpose(obj,(1,2,0)), np.transpose(bg,(1,2,0)) , np.transpose(mask,(1,2,0)),\
                            np.transpose(cp, (1, 2, 0))
            blended_im = gp_gan(obj, bg, mask[:,:,0], cp, G, args.image_size, args.gpu, color_weight=args.color_weight,
                                sigma=args.sigma,
                                gradient_kernel=args.gradient_kernel, smooth_sigma=args.smooth_sigma)

            imsave('{}/obj_{}.png'.format(args.blended_folder, idx),blended_im)



if __name__ == '__main__':
    main()