from torchvision import models
from blocks import *
import torch
import torch.nn as nn
from eegEncoder import *
from imageEncoder import *
from PIL import Image
import torchvision.transforms as transforms

def compatibility(x1, x2):
        return (x1*x2).sum(axis=1)
    
def triplet_loss(pos, neg):
    return torch.maximum(torch.zeros_like(pos), neg-pos)

class siamese_model(nn.Module):
    def __init__(self, embedding_size=1000):
        super().__init__()
        self.image_encoder_positive = image_encoder(embedding_size)
        self.image_encoder_negative = image_encoder(embedding_size)
        self.eeg_encoder = eeg_encoder(embedding_size)

    def forward(self, eeg, positive_img, negative_img):
        eeg_embedding = self.eeg_encoder(eeg)
        pos_embedding = self.image_encoder_positive(positive_img)
        neg_embedding = self.image_encoder_negative(negative_img)
        pos = compatibility(eeg_embedding, pos_embedding)
        neg = compatibility(eeg_embedding, neg_embedding)
        return triplet_loss(pos, neg)
    
# eeg = torch.randn(2,1,128,512)
# pos = torch.randn(2,3,299,299)
# neg = torch.randn(2,3,299,299)

def transform(x):
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(x)
torch.manual_seed(42)
with torch.inference_mode():
    checkpoint = torch.load('./dreamdiffusion/exps_siamese/16-04-2024-18-44-22/checkpoints/checkpoint.pth')
    model = siamese_model(checkpoint['config'].embedding_size)
    model.load_state_dict(checkpoint['model'])
    model.to(torch.device('cuda'))
    model.image_encoder_positive.eval()
    data = torch.load("./DreamDiffusion/datasets/eeg_55_95_std.pth")
    x = data['dataset'][0]['eeg'].unsqueeze(0).unsqueeze(0)
    x_img = Image.open(f"./DreamDiffusion/datasets/imageNet_images/{data['labels'][data['dataset'][0]['label']]}/{data['images'][data['dataset'][0]['image']]}.JPEG").convert('RGB')
    # x_img = Image.open('./DreamDiffusion/datasets/imageNet_images/n13054560/n13054560_10.JPEG').convert('RGB')
    x_img = transform(x_img)
    x_img = x_img.unsqueeze(0)
    pred = model.eeg_encoder(x)
    pred_img = model.image_encoder_positive(x_img)
    # print((((pred*pred).sum())**(0.5)*((pred_img*pred_img).sum())**(0.5)), (pred*pred_img).sum())
    print(model.state_dict()==checkpoint['model'])