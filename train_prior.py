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
from GAN import Discriminator, Generator, Prior

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Select GPU device number
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

NOTES = '7_GAN_inpaint'

# Define network hyperparameters:
HPARAMS = {
    'BATCH_SIZE': 100,
    'NUM_WORKERS': 16,
    'EPOCHS_NUM': 50,
    'LR_P': 0.0003,
}

TPARAMS = {}

image_shape = (28, 28)

START_DATE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

wandb.init(project='basic',
           config=HPARAMS,
           name=START_DATE,
#            mode='disabled',
           notes=NOTES)

class LossFunction():
    """Loss function class for multiple loss function."""

    def __init__(self):
        self.criterion = nn.CrossEntropyLoss().to(DEVICE)
    
    def forward(self, output, target):
        loss = self.criterion(output, target)
        return loss


def train(train_parameters):
    train_parameters['dis'].eval()
    train_parameters['gen'].eval()
    train_parameters['prior'].train()
    result = {}
    result['loss_sum'] = 0
    for (real, _) in train_parameters['trainset_loader']:
        train_parameters['P_optimizer'].zero_grad()
        real = real.to(DEVICE)
        mask = torch.ones((1,28,28))
        mask[:, 8:20, 5:18] = 0
        mask = mask.to(DEVICE)
        masked_real = real * mask
        batch_size = real.size()[0]
        target_fake = torch.zeros(batch_size, 1).to(DEVICE)
        target_fake = torch.ones(batch_size, 1).to(DEVICE)
        output_P = train_parameters['prior'](masked_real)
        output_G = train_parameters['gen'](output_P)
        output_D = train_parameters['dis'](output_G)
        masked_G = output_G * mask
#         loss_P = train_parameters['P_loss'].forward(masked_real, masked_G.long().squeeze(1))
        loss_P = train_parameters['P_loss'].forward(masked_real, masked_G)
        loss_D =  train_parameters['D_loss'].forward(target_fake, output_D)
        loss = loss_P + loss_D
        loss.backward()
        train_parameters['P_optimizer'].step()
        
        result['output'] = output_G
        result['input'] = masked_real
        result['loss_sum'] += loss
    return result


def test(test_parameters):
    test_parameters['dis'].eval()
    test_parameters['gen'].eval()
    test_parameters['prior'].eval()
    result = {}
    result['loss_sum'] = 0
    with torch.no_grad():
        for (real, _) in test_parameters['testset_loader']:
            real = real.to(DEVICE)
            mask = torch.ones((1,28,28))
            mask[:, 8:20, 5:18] = 0
            mask = mask.to(DEVICE)
            masked_real = real * mask
            batch_size = real.size()[0]
            target_fake = torch.zeros(batch_size, 1).to(DEVICE)
            output_P = test_parameters['prior'](masked_real)
            output_G = test_parameters['gen'](output_P)
            output_D = test_parameters['dis'](output_G)

            masked_G = output_G * mask
#             loss_P = test_parameters['P_loss'].forward(masked_real, masked_G.long().squeeze(1))
            loss_P = test_parameters['P_loss'].forward(masked_real, masked_G)
            loss_D = test_parameters['D_loss'].forward(target_fake, output_D)
            loss = loss_P + loss_D

            result['output'] = output_G
            result['input'] = masked_real
            result['loss_sum'] += loss
    return result


def main():
    TPARAMS['gen'] = Generator(img_shape=28, z=64)
    TPARAMS['gen'].load_state_dict(torch.load('./gen.pth'))
    TPARAMS['gen'] = TPARAMS['gen'].to(DEVICE)
    TPARAMS['dis'] = Discriminator(img_shape=28)
    TPARAMS['dis'].load_state_dict(torch.load('./dis.pth'))
    TPARAMS['dis'] = TPARAMS['dis'].to(DEVICE)
    
    TPARAMS['prior'] = Prior(z=64)
    TPARAMS['prior'] = TPARAMS['prior'].to(DEVICE)
    
    summary(TPARAMS['gen'], input_size=(64, 1))
    summary(TPARAMS['dis'], input_size=(1, 28, 28))
    summary(TPARAMS['prior'], input_size=(1, 28, 28))

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

#     TPARAMS['P_loss'] = LossFunction()
    TPARAMS['P_loss'] = nn.MSELoss()
    TPARAMS['D_loss'] = nn.MSELoss()
#     TPARAMS['D_loss'] = LossFunction()

    TPARAMS['P_optimizer'] = optim.Adam(TPARAMS['prior'].parameters(), lr=HPARAMS['LR_P'])

    # Training and test
    for epoch in range(HPARAMS['EPOCHS_NUM']):
        start_time = time.time()
        train_result = train(TPARAMS)
        test_result = test(TPARAMS)

        save_loss = train_result['loss_sum'] / len(TPARAMS['trainset_loader'])
        save_test_loss = test_result['loss_sum'] / len(TPARAMS['testset_loader'])
        
        print('Train :: epoch: {}. loss: {:.5}. time: {:.5}s.'.format(epoch, save_loss, time.time() - start_time))
        print('Valid :: epoch: {}. loss: {:.5}.'.format(epoch, save_test_loss))

        # Test result 
        train_output = train_result['output'][0:5].cpu()
        test_output = test_result['output'][0:5].cpu()
        train_input = train_result['input'][0:5].cpu()
        test_input = test_result['input'][0:5].cpu()
        log_test_output = wandb.Image(test_output)
        log_train_output = wandb.Image(train_output)
        log_test_input = wandb.Image(test_input)
        log_train_input = wandb.Image(train_input)

        wandb.log({
            "Train Input": log_train_input,
            "Train Result": log_train_output,
            "Train Loss": save_loss,
            "Test Input": log_test_input,
            "Test Result": log_test_output,
            "Test Loss": save_test_loss,
        }, step=epoch)
        

if __name__ == "__main__":
    main()