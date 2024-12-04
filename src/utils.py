import torch
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
from PIL import Image

def load_model(kind:str):
    model = deeplabv3_mobilenet_v3_large(weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
    if kind == 'Anime':
        model.classifier[4] = torch.nn.Conv2d(256,2,kernel_size=1)
        model.load_state_dict(torch.load('src/anime_seg1.pth',map_location='cpu',weights_only=False))
    elif kind =='Potrait':
        model.classifier[4] = torch.nn.Conv2d(256,2,kernel_size=1)
        model.load_state_dict(torch.load('src/human_seg.pth',map_location='cpu',weights_only=False))
    return model

def load_image(upload):
    return Image.open(upload).convert('RGB')