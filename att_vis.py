import torch
import cv2
import json
import numpy as np

from lavis.models import load_model_and_preprocess
from PIL import Image
from skimage import transform as skimage_transform
from scipy.ndimage import filters
from matplotlib import pyplot as plt

    
def getAttMap(img, attMap, blur = True, overlap = True):
    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap /= attMap.max()
    attMap = skimage_transform.resize(attMap, (img.shape[:2]), order = 3, mode = 'constant')
    if blur:
        attMap = filters.gaussian_filter(attMap, 0.02*max(img.shape[:2]))
        attMap -= attMap.min()
        if attMap.max() > 0:
            attMap /= attMap.max()
    cmap = plt.get_cmap('jet')
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)
    if overlap:
        attMap = 1*(1-attMap**0.7).reshape(attMap.shape + (1,))*img + (attMap**0.7).reshape(attMap.shape+(1,)) * attMapV
    return attMap

def AttMap(model, image_path, image_id):
    # image_path = "./bird.jpg"
    # print(model)
    model.text_decoder.bert.encoder.layer[-1].crossattention.self.save_attention = False
    raw_image = Image.open(image_path)
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    caption = model.generate({"image": image})

    model.eval()
    model.text_decoder.bert.encoder.layer[-1].crossattention.self.save_attention = True

    sample = {"image": image, "text_input": caption}
    out = model(sample)

    model.zero_grad()
    out.loss_lm.backward()

    with torch.no_grad():
        cams = model.text_decoder.bert.encoder.layer[-1].crossattention.self.get_attention_map()
        grads = model.text_decoder.bert.encoder.layer[-1].crossattention.self.get_attn_gradients()

        cams = cams[:, :, :, 1:].reshape(1, 12, -1, 24, 24)
        grads = grads[:, :, :, 1:].clamp(0).reshape(1, 12, -1, 24, 24)
        
        gradcam = cams * grads
        # gradcam = cams
        gradcam = gradcam[0].mean(0).cpu().detach().numpy()

    tokenizer = model.tokenizer
    text_input = tokenizer(caption)
    # print(text_input.input_ids[0])
    num_image = len(text_input.input_ids[0])
    print(caption)
    print(num_image)
    fig, ax = plt.subplots((num_image-1)//5 + 1, 5, figsize=(50, 5*((num_image-1)//5 + 1)))

    rgb_image = cv2.imread(image_path)[:, :, ::-1]
    rgb_image = np.float32(rgb_image) / 255

    ax[0][0].imshow(rgb_image)
    ax[0][0].set_yticks([])
    ax[0][0].set_xticks([])
    ax[0][0].set_xlabel("Image")

    for i,token_id in enumerate(text_input.input_ids[0][1:]):
        word = tokenizer.decode([token_id])
        gradcam_image = getAttMap(rgb_image, gradcam[i])

        ax[(i+1)//5][(i+1)%5].imshow(gradcam_image)
        ax[(i+1)//5][(i+1)%5].set_yticks([])
        ax[(i+1)//5][(i+1)%5].set_xticks([])
        ax[(i+1)//5][(i+1)%5].set_xlabel(word)

    plt.savefig(f'out/finetune/{image_id}.png') # change output dir

if __name__ == '__main__':
    annotations = "test.json" 
    with open(annotations) as f:
        data = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="large_coco", is_eval=False, device=device) # model
        
    for img in data:
        AttMap(model, f"../../dataset/cub_caption/{img['image']}", img['image_id'])