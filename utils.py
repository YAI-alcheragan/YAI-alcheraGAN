import torch

def cal_correct(outputs, labels):
  correct = 0
  with torch.no_grad():
    _, predictions = torch.max(outputs, 1)
    for label, prediction in zip(labels, predictions):
      if label==prediction:
        correct+=1
  
  return correct