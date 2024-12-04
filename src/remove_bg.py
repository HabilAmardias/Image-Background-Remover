import torch
from torchvision.transforms import v2
from PIL import Image
import numpy as np

transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32,scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])


def get_mask(model,image:Image,threshold:float=0.5):
    preprocessed = transforms(image).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred = model(preprocessed)['out']
        probs = torch.nn.functional.softmax(pred,dim=1)
    mask = probs[0,0]
    mask = 1-mask
    mask = (mask > threshold).float() * 255.0
    return mask.cpu().numpy().astype(np.uint8)

def remove_bg(model,image:Image)->Image:
    mask = get_mask(model,image)
    rgba = image.convert('RGBA')
    mask_pil = Image.fromarray(mask,mode='L')
    rgba.putalpha(mask_pil)
    return rgba

