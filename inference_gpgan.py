import argparse
import os
import numpy as np
import sys
import cv2
import torch
from torch import Tensor
from torchvision.utils import save_image
from skimage import img_as_float
from skimage.io import imread, imsave
from gp_gan import gp_gan
from model import EncoderDecoder
from inference_blend_dataset import BlendingDataset4
# import chainer
# from chainer import cuda, serializers


basename = lambda path: os.path.splitext(os.path.basename(path))[0]

"""
    Note: source image, destination image and mask image have the same size.

          root(./datasets) have  ./datasets/confirmed  , ./datasets/skipped  

          Data structure

              RGB/
        ├─ G0_2022-03-01_2022-06-30/
        │  ├─ confirmed/
        │  │  ├─ 2/
        │  │  │  ├─ images/
        │  │  │  │  ├─ cur/
        │  │  │  │  │  ├─ (file_name)
        │  │  │  ├─ labels/
        │  │  │  │  ├─ (file_name)
        │  │  │  ├─ masks/
        │  │  │  │  ├─ (file_name)
        │  │  ├─ 4/
        │  │  ├─ 3/
        │  │  ├─ .../
        │  ├─ skipped/
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
    parser = argparse.ArgumentParser(description='Gaussian-Poisson GAN for high-resolution image blending')

    ############ For training GP GAN ##################
    parser.add_argument('--nef', type=int, default=64, help='# of base filters in encoder')
    parser.add_argument('--ngf', type=int, default=64, help='# of base filters in decoder or G')
    parser.add_argument('--nc', type=int, default=3, help='# of output channels in decoder or G')
    parser.add_argument('--nBottleneck', type=int, default=4000, help='# of output channels in encoder')
    parser.add_argument('--ndf', type=int, default=64, help='# of base filters in D')
    parser.add_argument('--image_size', type=int, default=64, help='The height / width of the input image to GPGAN network')
    parser.add_argument('--color_weight', type=float, default=1, help='Color weight')
    parser.add_argument('--sigma', type=float, default=0.5,
                        help='Sigma for gaussian smooth of Gaussian-Poisson Equation')
    parser.add_argument('--gradient_kernel', type=str, default='normal', help='Kernel type for calc gradient')
    parser.add_argument('--smooth_sigma', type=float, default=1, help='Sigma for gaussian smooth of Laplacian pyramid')

    parser.add_argument('--supervised', type=lambda x: x == 'True', default=True,
                        help='Use unsupervised Blending GAN if False')
    parser.add_argument('--nz', type=int, default=100, help='Size of the latent z vector')
    parser.add_argument('--n_iteration', type=int, default=1000, help='# of iterations for optimizing z')

    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--g_path', default='./blending_gan.npz', help='Path for pretrained Blending GAN model')
    parser.add_argument('--unsupervised_path', default='models/unsupervised_blending_gan.npz',
                        help='Path for pretrained unsupervised Blending GAN model')

    ############ For inference ##################
    parser.add_argument('--root', default='./datasets', help='Path where confirmed and skipped folder are located at : e.g.  ./datasets/confirmed ')
    parser.add_argument('--result_folder', default='experiment_blending_result', help='Name for folder storing results')
    parser.add_argument('--crop_size', default=200, help='gp_gan function input image size(how much pixels to be cropped around bbox centor coordinate and re-attached to target image)')
    parser.add_argument('--n_output', default=50, help='How many images to be generated')

    args = parser.parse_args()


    ############ For debugging ##################
    print('-'*50, '\nInput arguments:')
    for key, value in vars(args).items():
        print('\t{}: {}'.format(key, value))

    print('-'*50)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    G = EncoderDecoder(args.nef, args.ngf, args.nc, args.nBottleneck, image_size=args.image_size)
    print('\nLoad pretrained Blending GAN model from {} ...'.format(args.g_path))
    load_weights(G, args.g_path)      # G.load_state_dict(torch.load(args.g_path))
    G.to(device)

    if not os.path.isdir(args.result_folder):
        os.makedirs(args.result_folder)
    print('\nResult will save to {} ...\n'.format(args.result_folder))
    import torchvision.transforms as T                                       
    temp_tf = T.Compose([
                T.ToPILImage(),
                T.Resize([256, 256]),
                T.ToTensor(),
              ])




    ############ For image blending ##################
    bg_folder, obj_folder = "skipped", "confirmed"      
    test_loader = torch.utils.data.DataLoader(BlendingDataset4(root= args.root, bg_folder=bg_folder, obj_folder=obj_folder,  load_size=256, \
                                                crop_size = args.crop_size),
                                            batch_size=1, 
                                            shuffle=False,
                                            drop_last=False)

    with torch.no_grad():
        G.eval()
        for idx, batch in enumerate(test_loader):

            print('Processing {}/{} ...'.format(idx + 1, len(test_loader)))

            ############ Transform torch image into numpy image ##################
            gp_obj, gp_bg, gp_mask, gp_cp = torch.squeeze(batch["gpgan_obj"]).cpu().numpy().astype(np.float32), \
            torch.squeeze(batch["gpgan_bg"]).cpu().numpy().astype(np.float32), torch.squeeze(batch["gpgan_mask"]).cpu().numpy().astype(np.float32), \
                                torch.squeeze(batch["gpgan_cp"]).cpu().numpy().astype(np.float32)
            gp_obj, gp_bg, gp_mask, gp_cp =np.transpose(gp_obj,(1,2,0)), np.transpose(gp_bg,(1,2,0)) , np.transpose(gp_mask,(1,2,0)),\
                            np.transpose(gp_cp, (1,2,0))
            gp_mask[gp_mask>0]=1


            ############ Image blending using GP GAN ##################
            blended_im = gp_gan(gp_obj, gp_bg, gp_mask[:,:,0], gp_cp, G, args.image_size, args.gpu, color_weight=args.color_weight,
                                sigma=args.sigma,
                                gradient_kernel=args.gradient_kernel, smooth_sigma=args.smooth_sigma)
            blended_torch = temp_tf(blended_im)

            # show result
            result_before1 = torch.cat([
                    batch["obj"].data.cpu()[0],
                    batch["mask_old"].data.cpu()[0], 
                    batch["cropped_obj"].data.cpu()[0],
                    batch["cropped_mask"].data.cpu()[0]], 2)

            result_before2 = torch.cat([
                    batch["bg"].data.cpu()[0],
                    batch["mask"].data.cpu()[0],
                    batch["cp"].data.cpu()[0], 
                    blended_torch.data.cpu()], 2)       

            result_before3 = torch.cat([
                    batch["mask"].data.cpu()[0],
                    batch["cp_old"].data.cpu()[0],
                    batch["cp"].data.cpu()[0], 
                    blended_torch.data.cpu()], 2)       

            result_total = torch.cat([result_before1,result_before2, result_before3], 1)

            ############ Undo cropping and reconstruct original image ##################
            gp_x1, gp_x2, gp_y1, gp_y2 = batch["region"]
            gp_x1, gp_x2, gp_y1, gp_y2 = gp_x1.item(), gp_x2.item(), gp_y1.item(), gp_y2.item()
            original_img = np.array(batch["bg_final"][0])
            original_img[gp_y1:gp_y2 , gp_x1:gp_x2] = blended_im



            ############ For debugging and visualize the results ##################
            print("gp gan inputs : " , gp_obj.shape, gp_bg.shape, gp_mask.shape)
            print("gp gan output : " , blended_im.shape)
            print(f"assert {gp_y2}-{gp_y1} = {args.crop_size}, {gp_x2}-{gp_x1}= {args.crop_size}")
            print(f"replacing region... : {original_img.shape}[{gp_y1}:{gp_y2},{gp_x1}:{gp_x2}] = {blended_im.shape}" )
            blended_torch = temp_tf(blended_im) 
            final_torch = temp_tf(original_img)
            print("visualize ",temp_tf(batch["gpgan_bg"].data.cpu()[0]).shape, \
            batch["gpgan_cp"].data.cpu()[0].shape, blended_torch.shape,final_torch.shape)

            result_before1 = torch.cat([
                    batch["obj"].data.cpu()[0],
                    batch["mask_old"].data.cpu()[0], 
                    batch["cropped_obj"].data.cpu()[0],
                    batch["cropped_mask"].data.cpu()[0]], 2)

            result_before2 = torch.cat([
                    batch["bg"].data.cpu()[0],
                    batch["mask"].data.cpu()[0],
                    batch["cp_old"].data.cpu()[0], 
                    batch["cp"].data.cpu()[0]], 2)       
            
            result_gp1 = torch.cat([
                    temp_tf(batch["gpgan_bg"].data.cpu()[0]),
                    temp_tf(batch["gpgan_obj"].data.cpu()[0]),
                    temp_tf(batch["gpgan_mask"].data.cpu()[0]),
                    temp_tf(batch["gpgan_cp"].data.cpu()[0])],2)

            result_gp2 = torch.cat([
                    temp_tf(batch["gpgan_bg"].data.cpu()[0]),
                    temp_tf(batch["gpgan_cp"].data.cpu()[0]),
                    blended_torch,
                    final_torch],2)  

            result_total = torch.cat([result_before1,result_before2,result_gp1,result_gp2], 1)             


            ############ Save the result images ##################
            if args.result_folder:
                save_image(result_total,'%s/total_%s.png' % (args.result_folder,idx))
            else:
                print("\nError : args.result_folder is not valid\n")
                break

            if idx == 50 :
              print(f"reach MAX_n_ouput {args.n_output}, end GPGAN inference")
              break

if __name__ == '__main__':
    main()