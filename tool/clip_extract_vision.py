import torch
import clip
from PIL import Image
import numpy as np

def get_vision_clip(img):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    if isinstance(img, str):
        image = preprocess(Image.open(img)).unsqueeze(0).to(device)
    else:
        image = Image.fromarray(np.uint8(img))
        image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
    return  image_features

if __name__ == '__main__':
    img = '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/0/0.png' 
    result = get_vision_clip(img)
    print(result.shape)
    



