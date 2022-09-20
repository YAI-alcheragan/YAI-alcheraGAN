import torch
from torch import nn
import timm

def Swinv2(class_num=2, pretrained=False) :
    model = timm.create_model('swinv2_base_window16_256', pretrained=pretrained)

    n_inputs = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(n_inputs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, class_num), #nosmoke, smoke 구분 -> smoke label이 1, no smoke label이 0
        nn.Softmax()
    )
    
    return model
