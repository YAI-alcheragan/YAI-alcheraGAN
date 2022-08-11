from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
import torch
import os
import time
import datetime
import wandb
import traceback
from torch import nn
from torch import optim
from torchsummary import summary
from GAN import Discriminator, Generator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Select GPU device number
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

NOTES = '7_GAN'

# Define network hyperparameters:
HPARAMS = {
    'BATCH_SIZE': 100,
    'NUM_WORKERS': 16,
    'EPOCHS_NUM': 50,
    'LR_G': 0.001,
    'LR_D': 0.0003
}

TPARAMS = {}

image_shape = (28, 28)

START_DATE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

wandb.init(project='basic',
           entity='oisl',
           config=HPARAMS,
           name=START_DATE,
           mode='disabled',
           notes=NOTES)

class LossFunction():
    """Loss function class for multiple loss function."""

    def __init__(self):
        self.criterion = nn.BCELoss()
    
    def forward(self, output, target):
        loss = self.criterion(output, target)
        return loss


def train(train_parameters):
    train_parameters['dis'].train()
    train_parameters['gen'].train()
    result = {}
    result['G_loss_sum'] = 0
    result['D_loss_sum'] = 0
    for (real, _) in train_parameters['trainset_loader']:
        real = real.to(DEVICE)
        batch_size = real.size()[0]
        z = train_parameters['gen'].z
        target_real = torch.ones(batch_size, 1).to(DEVICE)
        target_fake = torch.zeros(batch_size, 1).to(DEVICE)
        noise_d = torch.randn((batch_size, z)).to(DEVICE)
        noise_g = torch.randn((batch_size, z)).to(DEVICE)
        f_image_d = train_parameters['gen'](noise_d)
        f_image_g = train_parameters['gen'](noise_g)

        train_parameters['D_optimizer'].zero_grad()
        dis_real = train_parameters['dis'](real)
        dis_out_d = train_parameters['dis'](f_image_d)
        D_loss_real = train_parameters['D_loss'].forward(dis_real, target_real)
        D_loss_fake = train_parameters['D_loss'].forward(dis_out_d, target_fake)
        D_loss = D_loss_real + D_loss_fake
        D_loss.backward()
        train_parameters['D_optimizer'].step()
        
        train_parameters['G_optimizer'].zero_grad()
        dis_out_g = train_parameters['dis'](f_image_g)
        G_loss = train_parameters['G_loss'].forward(dis_out_g, target_real)
        G_loss.backward()
        train_parameters['G_optimizer'].step()
        
        result['G_loss_sum'] += G_loss.item()
        result['D_loss_sum'] += D_loss.item()
        result['output'] = f_image_g
    return result


def test(test_parameters):
    test_parameters['dis'].eval()
    test_parameters['gen'].eval()
    result = {}
    result['G_loss_sum'] = 0
    result['D_loss_sum'] = 0
    with torch.no_grad():
        for (real, _) in test_parameters['testset_loader']:
            real = real.to(DEVICE)
            batch_size = real.size()[0]
            target_real = torch.ones(batch_size, 1).to(DEVICE)
            target_fake = torch.zeros(batch_size, 1).to(DEVICE)

            noise_d = get_noise(n_samples = batch_size,
                                z_dim=test_parameters['gen'].z,
                                device=DEVICE)
            noise_g = get_noise(n_samples = batch_size,
                                z_dim=test_parameters['gen'].z,
                                device=DEVICE)
            f_image_d = test_parameters['gen'](noise_d)
            f_image_g = test_parameters['gen'](noise_g)

            dis_real = test_parameters['dis'](real)
            dis_out_d = test_parameters['dis'](f_image_d)
            dis_out_g = test_parameters['dis'](f_image_g)

            D_loss_real = test_parameters['D_loss'].forward(dis_real, target_real)
            D_loss_fake = test_parameters['D_loss'].forward(dis_out_d, target_fake)
            D_loss = D_loss_real + D_loss_fake
            G_loss = test_parameters['G_loss'].forward(dis_out_g, target_real)
            result['G_loss_sum'] += G_loss.item()
            result['D_loss_sum'] += D_loss.item()
            result['output'] = f_image_g
    return result


def main():
    TPARAMS['gen'] = Generator(img_shape=28, z=64)
    TPARAMS['gen'] = TPARAMS['gen'].to(DEVICE)
    TPARAMS['dis'] = Discriminator(img_shape=28)
    TPARAMS['dis'] = TPARAMS['dis'].to(DEVICE)
    
    summary(TPARAMS['gen'], input_size=(64, 1))
    summary(TPARAMS['dis'], input_size=(1, 28, 28))

    transformer = transforms.Compose([transforms.ToTensor(),
                                      transforms.RandomAffine(degrees=(30, 70),
                                                              translate=(0.1, 0.2),
                                                              scale=(0.75, 1)),
                                      transforms.Normalize(0.5, 0.5)
                                     ])

    train_data = datasets.MNIST(root='/data/deep_learning_study',
                                train=True,
                                download=False,
                                transform=transformer)
    test_data = datasets.MNIST(root='/data/deep_learning_study',
                               train=False,
                               download=False,
                               transform=transformer)

    TPARAMS['trainset_loader'] = torch.utils.data.DataLoader(train_data,
                                                          batch_size=HPARAMS['BATCH_SIZE'],
                                                          num_workers=HPARAMS['NUM_WORKERS'],
                                                          shuffle=True,
                                                          )
    TPARAMS['testset_loader'] = torch.utils.data.DataLoader(test_data,
                                                         batch_size=HPARAMS['BATCH_SIZE'],
                                                         num_workers=HPARAMS['NUM_WORKERS'],
                                                         shuffle=True,
                                                         )

    TPARAMS['G_loss'] = LossFunction()
    TPARAMS['D_loss'] = LossFunction()

    TPARAMS['G_optimizer'] = optim.Adam(TPARAMS['gen'].parameters(), lr=HPARAMS['LR_G'])
    TPARAMS['D_optimizer'] = optim.Adam(TPARAMS['dis'].parameters(), lr=HPARAMS['LR_D'])

    # Training and test
    for epoch in range(HPARAMS['EPOCHS_NUM']):
        start_time = time.time()
        train_result = train(TPARAMS)
        test_result = test(TPARAMS)

        save_loss_G = train_result['G_loss_sum'] / len(TPARAMS['trainset_loader'])
        save_loss_D = train_result['D_loss_sum'] / len(TPARAMS['trainset_loader'])
        save_test_loss_G = test_result['G_loss_sum'] / len(TPARAMS['testset_loader'])
        save_test_loss_D = test_result['D_loss_sum'] / len(TPARAMS['testset_loader'])
        
        print('Train :: epoch: {}. Gloss: {:.5}. Dloss: {:.5}. time: {:.5}s.'.format(epoch, save_loss_G, save_loss_D, time.time() - start_time))
        print('Valid :: epoch: {}. Gloss: {:.5}. Dloss: {:.5}.'.format(epoch, save_test_loss_G, save_test_loss_D))

        # Test result 
        train_output = train_result['output'][0:5].cpu()
        test_output = test_result['output'][0:5].cpu()
        log_test_output = wandb.Image(test_output)
        log_train_output = wandb.Image(train_output)

        wandb.log({
            "Train Result": log_train_output,
            "Train GLoss": save_loss_G,
            "Train DLoss": save_loss_D,
            "Test Result": log_test_output,
            "Test GLoss": save_test_loss_G,
            "Test DLoss": save_test_loss_D,
        }, step=epoch)
    torch.save(TPARAMS['gen'].state_dict(), './gen.pth')
    torch.save(TPARAMS['dis'].state_dict(), './dis.pth')
        

if __name__ == "__main__":
    main()