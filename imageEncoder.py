from torchvision import models
# from blocks import *
import torch
import torch.nn as nn

class image_encoder(nn.Module):
    def __init__(self, embedding_size=1000):
        super().__init__()
        self.encoder = models.inception_v3(pretrained=True)
        self.encoder.aux_logits = False

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        input_shape = self.encoder.fc.in_features
        self.encoder.fc = nn.Sequential(
            nn.Linear(input_shape, input_shape*2),
            nn.ReLU(),
            nn.BatchNorm1d(input_shape*2),
            nn.Dropout(0.3),
            nn.Linear(input_shape*2, input_shape),
            nn.ReLU(),
            nn.BatchNorm1d(input_shape),
            nn.Dropout(0.3),
            nn.Linear(input_shape, embedding_size*2),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_size*2),
            nn.Dropout(0.3),
            nn.Linear(embedding_size*2, embedding_size)
        )
    
    def forward(self, x):
        return self.encoder(x)
    
# image_encoder = image_encoder(1000)
# image_encoder.eval()
# x = torch.randn(1,3,299,299)
# output = image_encoder(x)

# print(output.shape)