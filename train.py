from dataset import SmokeDataset
from model import Swinv2
from utils import cal_correct

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from tqdm import tqdm
import os


if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    EPOCH = 100
    
    #### Define Dataset, Dataloader ###
    dataset = SmokeDataset('YAI_RGB_2203_2206/RGB/G0_2022-03-01_2022-06-30')
    dataset_size = len(dataset)
    # train_size, val_size = int(dataset_size * 0.8), int(dataset_size*0.1)
    train_size, val_size = int(dataset_size * 0.001), int(dataset_size*0.001)
    test_size = dataset_size-train_size-val_size
    # split dataset 
    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
    # define dataloader
    train_batch_size, val_batch_size, test_batch_size = 8, 4, 4
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=val_batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True, drop_last=True)
    
    ### Define Model ###
    model = Swinv2()
    model.to(device)
    
    ### Define Train Option ###
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    ### Make Save Dir, log.txt ###
    model_name = 'Swinv2_window16_no_pretrained'
    save_dir = os.path.join('savings', model_name)
    os.makedirs(save_dir, exist_ok=True)

    log_file = open(os.path.join(save_dir, "log.txt"), "w");
    


    best_train_acc = 0.0
    best_val_acc = 0.0

    ## Train
    for epoch in range(EPOCH):
        epoch_iter = 0
        
        lr_scheduler.step()
        model.train()
        
        running_loss = 0.0
        running_correct = 0
        epoch_correct = 0

        for i, (images, labels) in tqdm(enumerate(train_loader)):
      
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            correct_num = cal_correct(outputs, labels)
            running_correct += correct_num
            epoch_correct += correct_num

            
            if i % 200 == 199: # check training acc and val acc
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200 :.3f} acc : {running_correct/(200*train_batch_size):.3f}')
                log_file.write(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200 :.3f} acc : {running_correct/(200*train_batch_size):.3f}\n')
                running_loss, running_correct = 0.0, 0

                with torch.no_grad() :
                  val_total_loss = 0.0
                  val_acc = 0
                  for val_idx, (val_images, val_labels) in enumerate(val_loader):
                    val_outputs = model(val_images)
                    val_loss = criterion(outputs, labels)
                    val_total_loss += val_loss.item()
                    val_correct = cal_correct(val_outputs, val_labels)

                  val_acc = val_correct/val_size
                  print(f'## [{epoch + 1}, {i + 1:5d}] val loss: {val_total_loss / (val_idx+1) :.3f} val acc : {val_acc:.3f}')
                  log_file.write(f'## [{epoch + 1}, {i + 1:5d}] val loss: {val_total_loss / (val_idx+1) :.3f} val acc : {val_acc:.3f}')

                  if val_acc > best_val_acc :
                    torch.save(model.state_dict(), os.path.join(save_dir, f'best_val_acc.pt'))
                    print(f'#### Save Best Val Acc Model : {val_acc} ####')  
                    log_file.write(f'#### Save Best Acc Model : {val_acc} ####\n')
                    best_val_acc = val_acc


        epoch_acc = epoch_correct/train_size
        if epoch_acc > best_train_acc :
          torch.save(model.state_dict(), os.path.join(save_dir, f'best_train_acc.pt'))
          print(f'#### Save Best Train Acc Model : {epoch_acc} ####')  
          log_file.write(f'#### Save Best Train Acc Model : {epoch_acc} ####\n')
          best_train_acc = epoch_acc


        if epoch % 10 == 9 :
          torch.save(model.state_dict(), os.path.join(save_dir, f'epoch_{epoch+1}.pt'))
          print(f'#### Save Epoch : {epoch+1} ####')
          log_file.write(f'#### Save Epoch : {epoch+1} ####\n')
        
        
          

            




            
            
        
    